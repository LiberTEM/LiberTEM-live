import multiprocessing as mp
import os
import time
import socket
import threading
import contextlib
import logging

import numpy as np
import zmq

from libertem.io.dataset.k2is import DataBlock, SHUTTER_ACTIVE_MASK
from libertem.common.buffers import bytes_aligned, zeros_aligned
from libertem_live.utils.net import mcast_socket
from libertem_live.detectors.k2is.decode import decode_bulk_uint12_le
from libertem.io.dataset.base.tiling import TilingScheme, DataTile
from libertem.common import Shape, Slice
from libertem.io.dataset.base import Partition, DataSetMeta
from libertem.udf.base import UDFRunner
from libertem.executor.base import Environment
from ..common import ErrThreadMixin, send_serialized
from .state import (
    EventReplicaClient,
    CamConnectionState, ProcessingState,
    CamConnectedEvent, CamDisconnectedEvent,
)

GROUP = '225.1.1.1'

logger = logging.getLogger(__name__)


def warmup():
    warmup_buf_inp = bytes_aligned(0x5758*32)
    for dtype in [np.uint16, np.float32, np.float64]:
        warmup_buf_out = zeros_aligned((32, 930, 16), dtype=dtype).reshape((-1,))
        decode_bulk_uint12_le(inp=warmup_buf_inp[40:], out=warmup_buf_out, num_packets=32)


class FakeEnvironment(Environment):
    @contextlib.contextmanager
    def enter(self):
        yield


class SyncState:
    """
    Shared state for synchronization between processes
    """

    def __init__(self, num_processes):
        self.num_processes = num_processes
        self.first_frame_id = mp.Value('I', 0)
        self.sync_msg_seen = mp.Value('i', 0)
        self.sync_done = mp.Event()
        self.initial_sync = mp.Value('i', 0)
        self.reset_barrier = mp.Barrier(num_processes)

    def set_initial_sync(self):
        """
        Call this function to indicate that data from a sector
        has been received:
        """
        with self.initial_sync.get_lock():
            self.initial_sync.value += 1
            return self.initial_sync.value

    def set_first_frame_id(self, frame_id):
        """
        Each process should set the `frame_id` of the first full frame they observe
        """
        with self.first_frame_id.get_lock(), self.sync_msg_seen.get_lock():
            if self.sync_done.is_set():
                raise ValueError("sync already done")
            self.first_frame_id.value = max(self.first_frame_id.value, frame_id)
            self.sync_msg_seen.value += 1
            if self.sync_msg_seen.value == self.num_processes:
                self.sync_done.set()

    def get_first_frame_id(self):
        if not self.sync_done.is_set():
            raise RuntimeError("get_first_frame_id called with incomplete sync procedure")
        return self.first_frame_id.value

    def reset(self):
        # NOTE: blocking here, as timeout breaks the barrier state
        # this is the only place we are blocking for a while
        logger.info("SyncState.reset before barrier")
        ret = self.reset_barrier.wait(timeout=10)
        logger.info(f"SyncState.reset after barrier {ret}")
        if ret == 0:
            logger.info("SyncState.reset ret==0")
            with self.first_frame_id.get_lock(),\
                    self.sync_msg_seen.get_lock(),\
                    self.initial_sync.get_lock():
                self.first_frame_id.value = 0
                self.sync_msg_seen.value = 0
                self.initial_sync.value = 0
                self.sync_done.clear()
                self.reset_barrier.reset()


# FIXME later...
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

        # FIXME: how stop after processing `num_frames`?
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


class MsgReaderThread(ErrThreadMixin, threading.Thread):
    def __init__(
        self, idx, port, affinity_set, sync_state,
        local_addr='0.0.0.0', iface='enp193s0f0', timeout=0.1, pdb=False,
        profile=False,
        *args, **kwargs
    ):
        # TODO: separate this class... way too much state!
        self.idx = idx
        self.port = port
        self.affinity_set = affinity_set
        self.iface = iface
        self.local_addr = local_addr
        self.timeout = timeout
        self.sync_state = sync_state
        self.sync_timeout = 1  # TODO: make this a parameter?
        self.replica = EventReplicaClient()
        self.first_frame_id = None
        self.recent_frame_id = None  # the last frame_id we have seen
        self.buffered_tile = None  # a bufferd DataTile between calls of `get_tiles`
        self.sync_barrier = mp.Barrier(8)  # FIXME: hardcoded number of participants
        self.zctx = None
        self.result_socket = None
        self.is_connected = False
        self.packet_counter = 0
        self.pdb = pdb
        self.profile = profile

        super().__init__(*args, **kwargs)

    def run(self):
        profiler = None
        if self.profile:
            from line_profiler import LineProfiler
            profiler = LineProfiler()
            profiler.add_function(self.read_loop)
            profiler.add_function(self.get_tiles)
            profiler.add_function(self.main_loop)
            profiler.add_function(UDFRunner.run_for_partition)
            profiler.add_function(UDFRunner._run_udfs)
            profiler.add_function(UDFRunner._run_tile)
            try:
                profiler.runcall(self._real_run)
            finally:
                import io
                out = io.StringIO()
                profiler.print_stats(stream=out)
                out.seek(0)
                with open("/tmp/profile-%d.txt" % os.getpid(), "w") as f:
                    print("writing profile")
                    f.write(out.read())
        else:
            self._real_run()

    def _real_run(self):
        self.zctx = zmq.Context.instance()
        self.result_socket = self.zctx.socket(zmq.PUSH)
        self.result_socket.connect("tcp://127.0.0.1:7205")
        logger.info(f"thread {threading.get_native_id()}")
        os.sched_setaffinity(0, self.affinity_set)
        warmup()
        try:
            # FIXME: benchmark number of threads
            env = Environment(threads_per_worker=1)
            with mcast_socket(self.port, GROUP, self.local_addr, self.iface) as s, env.enter():
                logger.info(f"listening on {self.local_addr}:{self.port}/{GROUP} on {self.iface}")
                while True:
                    if self.is_stopped():
                        break
                    self.main_loop(s)
        except Exception as e:
            return self.error(e)
        self.stop()

    def update_first_frame_id(self, new_frame_id):
        self.first_frame_id = new_frame_id

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
                self.packet_counter += 1
                self.is_connected = True
            except socket.timeout:
                # On timeout, disconnect and make sure cancellation works by
                # processing events. Also reset synchronization, to make sure
                # we re-sync when data starts to come in again
                if self.state_is_connected():
                    logger.warn("read_loop timeout while connected, resetting")
                    self.replica.dispatch(CamDisconnectedEvent())
                    self.is_connected = False
                    self.sync_state.reset()
                self.replica.do_events()
                continue

            yield (buf, p[1])

    def read_loop_bulk(self, s, num_packets=128):
        """
        Read `num_packets` at once
        """
        # NOTE: non-IS data is truncated - we only read the first 0x5758 bytes of the message
        packet_size = 0x5758
        buf = bytes_aligned(packet_size * num_packets)
        buf_arr = np.frombuffer(buf)
        s.settimeout(self.timeout)
        idx = 0

        # first, sync up to `self.first_frame_id`:
        buf_part = buf[idx*packet_size:(idx + 1) * packet_size]
        while True:
            p = s.recvmsg_into([buf_part])
            h = np.frombuffer(buf_part, dtype=DataBlock.header_dtype, count=1, offset=0)
            if int(h['frame_id']) >= self.first_frame_id:
                idx = 1
                break

        while True:
            if self.is_stopped():
                return
            try:
                buf_part = buf[idx*packet_size:(idx + 1) * packet_size]
                p = s.recvmsg_into([buf_part])
                assert p[0] == packet_size
                idx += 1
                self.packet_counter += 1
                self.is_connected = True
                if idx == num_packets:
                    yield (buf, p[1])
                    idx = 0
                    buf_arr[:] = 0
            except socket.timeout:
                # On timeout, disconnect and make sure cancellation works by
                # processing events. Also reset synchronization, to make sure
                # we re-sync when data starts to come in again
                if self.state_is_connected():
                    logger.warn("read_loop timeout while connected, resetting")
                    self.replica.dispatch(CamDisconnectedEvent())
                    self.is_connected = False
                    self.sync_state.reset()
                self.replica.do_events()
                idx = 0
                buf_arr[:] = 0
                continue

    def sync(self, read_iter):
        """
        Initial synchronization of all sectors
        """
        first_frame_id = -1

        # get the frame ids for the first 32 blocks, to discard possibly incomplete
        # frames at the beginning:
        for i in range(32):
            p = next(read_iter)
            h = np.frombuffer(p[0], dtype=DataBlock.header_dtype, count=1, offset=0)
            first_frame_id = max(first_frame_id, int(h['frame_id']))

        if self.sync_state.set_initial_sync() >= 8:
            self.replica.dispatch(CamConnectedEvent())

        # discard packets until the shutter is active:
        while h['flags'] & SHUTTER_ACTIVE_MASK != 1 and False:
            p = next(read_iter)
            h = np.frombuffer(p[0], dtype=DataBlock.header_dtype, count=1, offset=0)
            first_frame_id = max(first_frame_id, int(h['frame_id']))

        # we send the highest frame_id we have seen:
        return self.sync_to_frame_id(first_frame_id)

    def sync_to_frame_id(self, frame_id: int) -> int:
        """
        Synchronize with the specified :code:`frame_id`. The largest :code:`frame_id`
        sent by any sector will be returned.

        Parameters
        ----------
        frame_id : int
            The :code:`frame_id` that should be sent to the synchonization process

        Returns
        -------
        int
            The resulting :code:`frame_id` that won the "election".

        Raises
        ------
        RuntimeError
            If the election fails to resolve in the given timeout
        """
        # send the frame_id to the synchronizer
        # (increment by 1, because we have read at least one block of this frame)
        self.sync_state.set_first_frame_id(frame_id + 1)

        # ... and wait, until all processes have sent their frame_id:
        if not self.sync_state.sync_done.wait(timeout=self.sync_timeout):
            raise RuntimeError("timed out waiting for sync")
        self.packet_counter = 0
        return self.sync_state.get_first_frame_id()

    @property
    def x_offset(self):
        return self.idx * 256

    def get_tiles(self, read_iter, end_after_idx, num_packets=128):
        assert num_packets % 32 == 0
        num_frames = num_packets // 32
        tileshape = Shape((num_frames, 1860, 256), sig_dims=2)
        buf = zeros_aligned(tileshape, dtype=np.uint16)

        x_offset = self.x_offset

        origin_to_idx = {}
        for idx, slice_ in ts.slices:
            origin_to_idx[slice_.origin] = idx

        logger.info(
            "get_tiles: end_after_idx=%d, num_packets=%d, buf.shape=%s",
            end_after_idx, num_packets, buf.shape,
        )

        # the first tile may be left over from the last run of this function:
        if self.buffered_tile is not None:
            frame_id, dt = self.buffered_tile
            frame_idx = frame_id - self.first_frame_id
            dt = DataTile(
                np.array(dt),
                tile_slice=Slice(
                    origin=(frame_idx,) + dt.tile_slice.origin[1:],
                    shape=dt.tile_slice.shape,
                ),
                scheme_idx=dt.scheme_idx,
            )
            yield dt
            self.buffered_tile = None

        sig_origin = (0, x_offset)
        for p in read_iter:
            try:
                meta_out = np.zeros((num_packets, 3), dtype=np.uint32)
                decode_bulk_uint12_le(
                    inp=p[0], out=buf, num_packets=num_packets, meta_out=meta_out,
                )
            except RuntimeError:
                logger.info(
                    "called decode bulk w/ len(p[0])=%d, buf.shape=%s, num_packets=%d",
                    len(p[0]), buf.shape, num_packets,
                )
                logger.info(
                    "meta_out=%r", meta_out,
                )
                raise
            # FIXME: assumption on packet ordering
            frame_id = meta_out[0, 0]  # first frame id
            frame_idx = frame_id - self.first_frame_id
            if frame_idx < 0:
                # Detect wraparound either a) because we are replaying a finite
                # input file, or b) because we reached the range of the integer type.
                # We select an arbitrary threshold of 10 frames to detect a)
                if frame_id == 0 or abs(frame_idx) > 10:
                    logging.info(
                        "wraparound? frame_id=%d frame_idx=%d", 
                        frame_id, frame_idx,
                    )
                    assert False, "TODO"
                    self.update_first_frame_id(frame_id)
                else:
                    # Discard any remaining packets from the sync procedure,
                    # which may include out-of-order packets:
                    continue
            self.recent_frame_id = frame_id

            # FIXME: border handling where tileshape[0] < num_frames should hold
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
            # FIXME: possibly split DataTile if partition is not divisible by num_frames
            if frame_idx > end_after_idx:
                logger.info(
                    "called bulk decode with len(inp)=%d, num_packets=%d",
                    len(p[0]), num_packets,
                )
                self.buffered_tile = (frame_id, dt)
                return
            else:
                yield dt

    @property
    def nav_shape(self):
        return self.replica.store.state.nav_shape

    def should_process(self):
        if self.replica.store.state is None:
            return False
        return self.replica.store.state.processing is ProcessingState.RUNNING

    def state_is_connected(self):
        return self.is_connected

    def main_loop(self, s):
        self.replica.do_events()

        logger.info("MsgReaderThread: waiting for first packet(s)")

        read_iter_sync = self.read_loop(s)

        while not self.should_process():
            # we are connected but aren't running a UDF yet, so we need to drain
            # the socket until we are told to run:
            if self.is_stopped():
                break
            _ = next(read_iter_sync)
            self.replica.do_events()

        logger.info("syncing")
        self.first_frame_id = self.sync(read_iter_sync)
        self.recent_frame_id = self.first_frame_id
        logger.info(f"synced to {self.first_frame_id}")

        # NOTE: we could make this dynamic, but that's for another day;
        # as currently the UDF needs to know how many frames there
        # are per partition. we could keep this rather small and
        # have another thread `merge` all the intermediate results,
        # and then sample _that_ intermediate result to the main process
        frames_per_partition = 100

        frame_counter = 0
        epoch = 0
        tiling_scheme = None

        read_iter = self.read_loop_bulk(s)

        while True:
            num_frames = np.prod(self.nav_shape, dtype=np.int64)
            frames_in_partition = min(frames_per_partition, num_frames - frame_counter)

            if frames_in_partition <= 0:
                # FIXME: if not in continuous mode, dispatch a stop event here
                frame_counter = 0
                epoch += 1
                self.first_frame_id = self.recent_frame_id
                continue

            self.replica.do_events()
            if self.is_stopped() or not self.should_process():
                return

            tiles = self.get_tiles(
                read_iter,
                end_after_idx=frame_counter + frames_in_partition - 1,
            )

            meta = DataSetMeta(
                shape=Shape((num_frames, 1860, 2048), sig_dims=2),
                image_count=num_frames,
                raw_dtype=np.uint16,
            )

            partition_slice = Slice(
                origin=(frame_counter, 0, self.x_offset),
                shape=Shape((frames_in_partition, 1860, 256), sig_dims=2),
            )

            partition = PlaceholderPartition(
                meta=meta,
                partition_slice=partition_slice,
                tiles=tiles,
                start_frame=frame_counter,
                num_frames=frames_in_partition,
            )

            # udfs = [
            #     udf.copy()
            #     for udf in self.replica.store.state.udfs
            # ]
            udfs = self.replica.store.state.udfs
            runner = UDFRunner(udfs)

            logger.debug(f"before run_for_partition {frame_counter}")
            result = runner.run_for_partition(
                partition=partition,
                corrections=None,
                roi=None,
                env=FakeEnvironment(threads_per_worker=1),
                tiling_scheme=tiling_scheme,
                pdb_port=4444 + self.idx if self.pdb else None,
            )
            tiling_scheme = result.tiling_scheme
            logger.info("sending some result for %r (sector %d)", partition_slice, self.idx)
            self.send_result(partition_slice, result, epoch, self.packet_counter)
            frame_counter += frames_in_partition

    def send_result(self, partition_slice, result, epoch, packet_counter):
        send_serialized(self.result_socket, (partition_slice, result, epoch, packet_counter))


def get_settings_for_sector(idx):
    return {
        'idx': idx,  # zero-based index of sector
        'local_addr': '225.1.1.1',
        'port': 2001 + idx,
        'affinity_set': {8 + idx},
        # 'iface': 'veth2',
        'iface': 'enp193s0f0' if idx < 4 else 'enp193s0f1',
    }


class K2ListenerProcess(mp.Process):
    def __init__(self, idx, sync_state, enable_tracing, pdb, profile, *args, **kwargs):
        self.idx = idx
        self.sync_state = sync_state
        self.enable_tracing = enable_tracing
        self.pdb = pdb
        self.profile = profile
        self._stop_event = mp.Event()
        super().__init__(*args, **kwargs)

    def stop(self):
        self._stop_event.set()

    def is_stopped(self):
        return self._stop_event.is_set()

    def run(self):
        import gc
        gc.disable()
        if self.enable_tracing:
            import pytracing
            self._trace_f = open("/tmp/cam-server-%d.json" % os.getpid(), "wb")
            tp = pytracing.TraceProfiler(output=self._trace_f)
            try:
                with tp.traced():
                    self.main()
            finally:
                self._trace_f.close()
        else:
            self.main()
        gc.enable()

    def main(self):
        warmup()

        settings = get_settings_for_sector(self.idx)
        settings.update({
            'sync_state': self.sync_state,
            'pdb': self.pdb,
            'profile': self.profile,
        })
        t = MsgReaderThread(**settings)

        try:
            t.start()
            logger.info(f"MySubProcess {self.idx} started processing")
            while not (t.is_stopped() or self.is_stopped()):
                t.maybe_raise()
                time.sleep(1)
            logger.info(f"MySubProcess {self.idx} stopped processing")
            t.maybe_raise()
        finally:
            t.stop()
            t.join()
            self.stop()
        logger.info(f"MySubProcess {self.idx} end of run()")
