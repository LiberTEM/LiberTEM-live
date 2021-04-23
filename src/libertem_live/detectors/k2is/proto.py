import multiprocessing as mp
import os
import time
import socket
import threading

import numpy as np

from libertem.io.dataset.k2is import DataBlock
from libertem.common.buffers import bytes_aligned, zeros_aligned
from libertem_live.utils.net import mcast_socket
from libertem_live.detectors.k2is.decode import decode_uint12_le
from libertem_live.detectors.common import StoppableThreadMixin
from libertem.io.dataset.base.tiling import TilingScheme, DataTile
from libertem.common import Shape, Slice
from libertem.io.dataset.base import Partition, DataSetMeta
from libertem.udf.base import UDFRunner
from libertem.executor.base import Environment

GROUP = '225.1.1.1'


class SyncState:
    """
    Shared state for synchronization between processes
    """

    def __init__(self, num_processes):
        self.num_processes = num_processes
        self.first_frame_id = mp.Value('I', 0)
        self.sync_msg_seen = mp.Value('i', 0)
        self.sync_done = mp.Event()

    def set_first_frame_id(self, frame_id):
        """
        Each process should set the `frame_id` of the first full frame they observe
        """
        if self.sync_done.is_set():
            raise ValueError("sync already done")
        with self.first_frame_id.get_lock(), self.sync_msg_seen.get_lock():
            self.first_frame_id.value = max(self.first_frame_id.value, frame_id)
            self.sync_msg_seen.value += 1
            if self.sync_msg_seen.value == self.num_processes:
                self.sync_done.set()

    def get_first_frame_id(self):
        if not self.sync_done.is_set():
            raise RuntimeError("get_first_frame_id called with incomplete sync procedure")
        return self.first_frame_id.value


ts = TilingScheme.make_for_shape(
    tileshape=Shape((1, 930, 16), sig_dims=2),
    dataset_shape=Shape((1, 2*930, 16*8*16), sig_dims=2)
)


class PlaceholderPartition(Partition):
    def __init__(
        self, meta: DataSetMeta, partition_slice: Slice, tiles, start_frame: int, num_frames: int,
    ):
        self._tiles = tiles
        self._start_frame = start_frame
        self._num_frames = num_frames
        super().__init__(
            meta=meta,
            partition_slice=partition_slice,
            io_backend=None,
        )

    def get_tiles(self, tiling_scheme, dest_dtype=np.float32, roi=None):
        assert roi is None

        # FIXME: stop after processing `num_frames`
        for tile in self._tiles:
            yield tile

    def need_decode(self, read_dtype, roi, corrections):
        return True

    def get_base_shape(self, roi):
        return (930, 16)

    def adjust_tileshape(self, tileshape, roi):
        return tileshape  # FIXME

    def set_corrections(self, corrections):
        self._corrections = corrections


class MsgReaderThread(StoppableThreadMixin, threading.Thread):
    def __init__(
        self, idx, port, affinity_set, sync_state, udfs, out_queue,
        local_addr='0.0.0.0', iface='enp193s0f0', timeout=0.1,
        *args, **kwargs
    ):
        self.idx = idx
        self.port = port
        self.affinity_set = affinity_set
        self.iface = iface
        self.local_addr = local_addr
        self.timeout = timeout
        self.sync_state = sync_state
        self.sync_timeout = 1  # TODO: make this a parameter?
        self.e = threading.Event()
        self.udfs = udfs
        self.out_queue = out_queue
        super().__init__(*args, **kwargs)

    def read_loop(self, s):
        # NOTE: non-IS data is truncated - we only read the first 0x5758 bytes of the message
        buf = bytes_aligned(0x5758)
        s.settimeout(self.timeout)
        while True:
            if self.is_stopped():
                return
            try:
                p = s.recvmsg_into([buf])
                assert p[0] == 0x5758
            except socket.timeout:
                continue

            yield (buf, p[1])

    def sync(self, read_iter):
        """
        Syncronize all sectors
        """
        frame_ids = set()

        # get the frame ids for the first 32 blocks:
        for i in range(32):
            p = next(read_iter)
            h = np.frombuffer(p[0], dtype=DataBlock.header_dtype, count=1, offset=0)
            frame_ids.add(int(h['frame_id']))

        # we send the highest frame_id we see in those 32 blocks:
        self.sync_state.set_first_frame_id(max(frame_ids))

        # ... and wait, until all processes have sent their frame_id:
        if not self.sync_state.sync_done.wait(timeout=self.sync_timeout):
            raise RuntimeError("timed out waiting for sync")
        return self.sync_state.get_first_frame_id()

    @property
    def x_offset(self):
        return self.idx * 256

    def get_tiles(self, read_iter, first_frame_id):
        tileshape = Shape((1, 930, 16), sig_dims=2)
        buf = zeros_aligned((1, 930, 16), dtype=np.uint16)
        buf_flat = buf.reshape((-1,))

        x_offset = self.x_offset

        origin_to_idx = {}
        for idx, slice_ in ts.slices:
            origin_to_idx[slice_.origin] = idx

        for p in read_iter:
            decode_uint12_le(inp=p[0][40:], out=buf_flat)
            h = np.frombuffer(p[0], dtype=DataBlock.header_dtype, count=1, offset=0)
            frame_idx = int(h['frame_id']) - first_frame_id

            sig_origin = (
                int(h['pixel_y_start']),
                int(h['pixel_x_start']) + x_offset
            )

            tile_slice = Slice(
                origin=(frame_idx,) + sig_origin,
                shape=tileshape,
            )
            scheme_idx = origin_to_idx[sig_origin]
            dt = DataTile(
                buf,
                tile_slice=tile_slice,
                scheme_idx=scheme_idx,
            )
            yield dt

    def run(self):
        print(f"thread {threading.get_native_id()}")
        os.sched_setaffinity(0, self.affinity_set)
        self.e.wait()
        print(f"listening on {self.local_addr}:{self.port}/{GROUP} on {self.iface}")

        with mcast_socket(self.port, GROUP, self.local_addr, self.iface) as s:
            print("entry MsgReaderThread, waiting for first packet(s)")

            first_frame_id = None
            read_iter = self.read_loop(s)

            first_frame_id = self.sync(read_iter)

            print(f"synced to {first_frame_id}")

            tiles = self.get_tiles(read_iter, first_frame_id)

            # FIXME: partitioning
            # frames_per_partition = 400

            num_frames = 4000  # less than 10 seconds

            meta = DataSetMeta(
                shape=Shape((num_frames, 1860, 2048), sig_dims=2),
                image_count=num_frames,
                raw_dtype=np.uint16,
            )

            partition_slice = Slice(
                origin=(0, 0, self.x_offset),
                shape=Shape((num_frames, 1860, 256), sig_dims=2),
            )

            # let's first create single partition per sector, with size >= what
            # we expect during 10 seconds of runtime
            partition = PlaceholderPartition(
                meta=meta,
                partition_slice=partition_slice,
                tiles=tiles,
                start_frame=0,
                num_frames=num_frames,
            )

            env = Environment(threads_per_worker=2)  # FIXME?
            runner = UDFRunner(self.udfs)
            result = runner.run_for_partition(
                partition=partition,
                corrections=None,
                roi=None,
                env=env,
            )
            print(result)
            self.out_queue.put(result)
            print(f"stored result into q {self.out_queue}")


def get_settings_for_sector(idx):
    return {
        'idx': idx,  # zero-based index of sector
        'local_addr': '225.1.1.1',
        'port': 2001 + idx,
        'affinity_set': {8 + idx},
        'iface': 'enp193s0f0' if idx < 4 else 'enp193s0f1',
    }


class MySubProcess(mp.Process):
    def __init__(self, idx, sync_state, udfs, out_queue, acqtime=10, *args, **kwargs):
        self.idx = idx
        self.sync_state = sync_state
        self.udfs = udfs
        self.acqtime = acqtime
        self.out_queue = out_queue
        super().__init__(*args, **kwargs)

    def run(self):
        warmup_buf_out = zeros_aligned((930, 16), dtype=np.uint16).reshape((-1,))
        warmup_buf_inp = zeros_aligned(0x5758, dtype=np.uint8)

        decode_uint12_le(inp=warmup_buf_inp[40:], out=warmup_buf_out)

        try:
            settings = get_settings_for_sector(self.idx)
            settings.update({
                'sync_state': self.sync_state,
                'udfs': self.udfs,
                'out_queue': self.out_queue,
            })
            t = MsgReaderThread(**settings)
            t.start()
            # time.sleep(30) # → uncomment for tracing purposes
            # for debugging, we can delay the start of the actual work in the
            # thread using this event:
            t.e.set()
            time.sleep(self.acqtime)  # TTL: how long should we acquire data?
        finally:
            t.stop()
            t.join()
