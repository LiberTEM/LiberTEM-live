import multiprocessing as mp
import uuid
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
from libertem.udf.base import UDFRunner, UDFMeta, UDFResults
from libertem.executor.base import Environment
from libertem.common.buffers import BufferWrapper
from ..common import SerializedQueue
from .state import EventReplicaClient, CamConnectedEvent

GROUP = '225.1.1.1'


class FakeDataSet:
    def __init__(self):
        self.shape = Shape((num_frames, 1860, 2048), sig_dims=2)
        self.dtype = np.uint16


class K2ISTask:
    def __init__(self, sector_idx, udfs):
        self.idx = sector_idx
        self.udfs = udfs


def make_udf_tasks(udfs, dataset, roi, corrections, backends):
    assert roi is None
    assert corrections is None or not corrections.have_corrections()

    # in case of a k2is live dataset, we need to create "tasks" for each
    # partition, so for each sector:
    return [
        K2ISTask(idx, udfs)
        for idx in range(8)
    ]


class FakeExecutor:
    def run_tasks(self, tasks, cancel_id):
        ss = SyncState(num_processes=len(tasks))
        processes = []
        oqs = []
        try:
            for task in tasks:
                oq = SerializedQueue()
                p = MySubProcess(
                    idx=task.idx,
                    sync_state=ss,
                    out_queue=oq
                )
                p.start()
                processes.append(p)
                oqs.append(oq)
            for idx, q in enumerate(oqs):
                print(f"getting result from q {idx}")
                part_results = q.get()
                for timing in part_results.timings:
                    print(f"{idx}: {timing}")
                yield part_results, idx
        finally:
            for p in processes:
                p.join()

    def ensure_sync(self):
        return self


def _get_dtype(udfs, dtype, corrections):
    if corrections is not None and corrections.have_corrections():
        tmp_dtype = np.result_type(np.float32, dtype)
    else:
        tmp_dtype = dtype
    for udf in udfs:
        tmp_dtype = np.result_type(
            udf.get_preferred_input_dtype(),
            tmp_dtype
        )
    return tmp_dtype


def _prepare_run_for_dataset(
    udfs, dataset, executor, roi, corrections, backends, dry
):
    meta = UDFMeta(
        partition_shape=None,
        dataset_shape=dataset.shape,
        roi=roi,
        dataset_dtype=dataset.dtype,
        input_dtype=_get_dtype(udfs, dataset.dtype, corrections),
        corrections=corrections,
    )
    for udf in udfs:
        udf.set_meta(meta)
        udf.init_result_buffers()
        udf.allocate_for_full(dataset, roi)

        if hasattr(udf, 'preprocess'):
            udf.set_views_for_dataset(dataset)
            udf.preprocess()
    if dry:
        tasks = []
    else:
        tasks = list(make_udf_tasks(udfs, dataset, roi, corrections, backends))
    return tasks


def _partition_by_idx(idx):
    # num_frames = 1800  # less than 10 seconds

    meta = DataSetMeta(
        shape=Shape((num_frames, 1860, 2048), sig_dims=2),
        image_count=num_frames,
        raw_dtype=np.uint16,
    )

    x_offset = 256 * idx

    partition_slice = Slice(
        origin=(0, 0, x_offset),
        shape=Shape((num_frames, 1860, 256), sig_dims=2),
    )

    # let's first create single partition per sector, with size >= what
    # we expect during 10 seconds of runtime
    return PlaceholderPartition(
        meta=meta,
        partition_slice=partition_slice,
        tiles=[],
        start_frame=0,
        num_frames=num_frames,
    )


def run_for_dataset_sync(udfs, dataset, executor,
                    roi=None, progress=False, corrections=None, backends=None, dry=False):
    tasks = _prepare_run_for_dataset(
        udfs, dataset, executor, roi, corrections, backends, dry
    )
    cancel_id = str(uuid.uuid4())

    if progress:
        from tqdm import tqdm
        t = tqdm(total=len(tasks))

    executor = executor.ensure_sync()

    damage = BufferWrapper(kind='nav', dtype=bool)
    damage.set_shape_ds(dataset.shape, roi)
    damage.allocate()
    if tasks:
        for part_results, task in executor.run_tasks(tasks, cancel_id):
            if progress:
                t.update(1)
            for results, udf in zip(part_results.buffers, udfs):
                udf.set_views_for_partition(_partition_by_idx(task))
                udf.merge(
                    dest=udf.results.get_proxy(),
                    src=results.get_proxy()
                )
                udf.clear_views()
            v = damage.get_view_for_partition(_partition_by_idx(task))
            v[:] = True
            yield UDFResults(
                buffers=tuple(
                    udf._do_get_results()
                    for udf in udfs
                ),
                damage=damage
            )
    else:
        # yield at least one result (which should be empty):
        for udf in udfs:
            udf.clear_views()
        yield UDFResults(
            buffers=tuple(
                udf._do_get_results()
                for udf in udfs
            ),
            damage=damage
        )

    if progress:
        t.close()


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
        self, idx, port, affinity_set, sync_state, out_queue,
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
        self.out_queue = out_queue
        self.replica = EventReplicaClient()
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

    @property
    def nav_shape(self):
        return self.replica.store.state.nav_shape

    def run(self):
        print(f"thread {threading.get_native_id()}")
        os.sched_setaffinity(0, self.affinity_set)
        print(f"listening on {self.local_addr}:{self.port}/{GROUP} on {self.iface}")

        with mcast_socket(self.port, GROUP, self.local_addr, self.iface) as s:
            print("entry MsgReaderThread, waiting for first packet(s)")

            self.replica.do_events()

            first_frame_id = None
            read_iter = self.read_loop(s)

            first_frame_id = self.sync(read_iter)

            print(f"synced to {first_frame_id}")

            self.replica.dispatch(CamConnectedEvent())

            tiles = self.get_tiles(read_iter, first_frame_id)

            # very simple state machine: either we are running a UDF, or we are just
            # reading from the sockets and throw away the data

            while self.replica.store.state is None and not self.is_stopped():
                self.replica.do_events(timeout=100)

            while not self.replica.store.state.udfs and not self.is_stopped():
                self.replica.do_events(timeout=100)

            if self.is_stopped():
                return

            frames_per_partition = 400

            while True:
                num_frames = np.prod(self.nav_shape)

                meta = DataSetMeta(
                    shape=Shape((num_frames, 1860, 2048), sig_dims=2),
                    image_count=num_frames,
                    raw_dtype=np.uint16,
                )

                partition_slice = Slice(
                    origin=(0, 0, self.x_offset),
                    shape=Shape((frames_per_partition, 1860, 256), sig_dims=2),
                )

                # let's first create single partition per sector, with size >= what
                # we expect during 10 seconds of runtime
                partition = PlaceholderPartition(
                    meta=meta,
                    partition_slice=partition_slice,
                    tiles=tiles,
                    start_frame=0,
                    num_frames=frames_per_partition,
                )

                env = Environment(threads_per_worker=2)  # FIXME?
                runner = UDFRunner(self.replica.store.state.udfs)

                result = runner.run_for_partition(
                    partition=partition,
                    corrections=None,
                    roi=None,
                    env=env,
                )
                self.out_queue.put(result)  # FIXME: replace with a zmq socket
            self.stop()


def get_settings_for_sector(idx):
    return {
        'idx': idx,  # zero-based index of sector
        'local_addr': '225.1.1.1',
        'port': 2001 + idx,
        'affinity_set': {8 + idx},
        'iface': 'veth2',
        # 'iface': 'enp193s0f0' if idx < 4 else 'enp193s0f1',
    }


class MySubProcess(mp.Process):
    def __init__(self, idx, sync_state, out_queue, *args, **kwargs):
        self.idx = idx
        self.sync_state = sync_state
        self.out_queue = out_queue
        super().__init__(*args, **kwargs)

    def run(self):
        warmup_buf_out = zeros_aligned((930, 16), dtype=np.uint16).reshape((-1,))
        warmup_buf_inp = zeros_aligned(0x5758, dtype=np.uint8)

        decode_uint12_le(inp=warmup_buf_inp[40:], out=warmup_buf_out)

        settings = get_settings_for_sector(self.idx)
        settings.update({
            'sync_state': self.sync_state,
            'out_queue': self.out_queue,
        })
        t = MsgReaderThread(**settings)

        try:
            t.start()
            print(f"MySubProcess {self.idx} waiting for results")
            while not t.is_stopped():
                time.sleep(1)
            print(f"MySubProcess {self.idx} done waiting for results, joining thread")
        finally:
            t.stop()
            t.join()
            print(f"MySubPRocess {self.idx} closing out_queue")
            self.out_queue.close()
        print(f"MySubProcess {self.idx} end of run()")
