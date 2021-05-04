import multiprocessing as mp
import uuid
import os
import time
import socket
import threading
import contextlib
import logging

import numpy as np

from libertem.io.dataset.k2is import DataBlock, SHUTTER_ACTIVE_MASK
from libertem.common.buffers import bytes_aligned, zeros_aligned
from libertem_live.utils.net import mcast_socket
from libertem_live.detectors.k2is.decode import decode_uint12_le
from libertem.io.dataset.base.tiling import TilingScheme, DataTile
from libertem.common import Shape, Slice
from libertem.io.dataset.base import Partition, DataSetMeta
from libertem.udf.base import UDFRunner, UDFMeta, UDFResults
from libertem.executor.base import Environment
from libertem.common.buffers import BufferWrapper
from ..common import SerializedQueue, ErrThreadMixin
from .state import (
    EventReplicaClient,
    CamConnectionState, ProcessingState,
    CamConnectedEvent, CamDisconnectedEvent, StopProcessingEvent,
)

GROUP = '225.1.1.1'

logger = logging.getLogger(__name__)


def warmup():
    warmup_buf_inp = bytes_aligned(0x5758)
    for dtype in [np.uint16, np.float32, np.float64]:
        warmup_buf_out = zeros_aligned((930, 16), dtype=dtype).reshape((-1,))
        decode_uint12_le(inp=warmup_buf_inp[40:], out=warmup_buf_out)


class FakeEnvironment(Environment):
    @contextlib.contextmanager
    def enter(self):
        yield


class FakeDataSet:
    def __init__(self, nav_shape):
        self.nav_shape = nav_shape
        self.shape = Shape(nav_shape + (1860, 2048), sig_dims=2)
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
                p = K2ListenerProcess(
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
        self.first_frame_id = None
        self.last_frame_id = None  # the last frame_id we have seen
        self.buffered_tile = None  # a bufferd DataTile between calls of `get_tiles`
        self.sync_barrier = mp.Barrier(8)  # FIXME: hardcoded number of participants
        super().__init__(*args, **kwargs)

    def run(self):
        logger.info(f"thread {threading.get_native_id()}")
        os.sched_setaffinity(0, self.affinity_set)
        warmup()
        try:
            # FIXME: benchmark number of threads
            env = Environment(threads_per_worker=1)
            with mcast_socket(self.port, GROUP, self.local_addr, self.iface) as s, env.enter():
                logger.info(f"listening on {self.local_addr}:{self.port}/{GROUP} on {self.iface}")
                read_iter = self.read_loop(s)
                while True:
                    if self.is_stopped():
                        break
                    self.main_loop(s, read_iter)
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
            except socket.timeout:
                # On timeout, disconnect and make sure cancellation works by
                # processing events. Also reset synchronization, to make sure
                # we re-sync when data starts to come in again
                if self.state_is_connected():
                    logger.warn("read_loop timeout while connected, resetting")
                    self.replica.dispatch(CamDisconnectedEvent())
                    self.sync_state.reset()
                self.replica.do_events()
                continue

            yield (buf, p[1])

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
        while h['flags'] & SHUTTER_ACTIVE_MASK != 1:
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
        return self.sync_state.get_first_frame_id()

    @property
    def x_offset(self):
        return self.idx * 256

    def get_tiles(self, read_iter, end_after_idx):
        tileshape = Shape((1, 930, 16), sig_dims=2)
        buf = zeros_aligned((1, 930, 16), dtype=np.uint16)
        buf_flat = buf.reshape((-1,))

        x_offset = self.x_offset

        origin_to_idx = {}
        for idx, slice_ in ts.slices:
            origin_to_idx[slice_.origin] = idx

        # the first tile may be left over from the last run of this function:
        if self.buffered_tile is not None:
            yield self.buffered_tile
            self.buffered_tile = None

        for p in read_iter:
            decode_uint12_le(inp=p[0][40:], out=buf_flat)
            h = np.frombuffer(p[0], dtype=DataBlock.header_dtype, count=1, offset=0)
            frame_id = int(h['frame_id'])
            frame_idx = frame_id - self.first_frame_id
            if frame_idx < 0:
                # Detect wraparound either a) because we are replaying a finite
                # input file, or b) because we reached the range of the integer type.
                # We select an arbitrary threshold of 10 frames to detect a)
                if frame_id == 0 or abs(frame_idx) > 10:
                    self.update_first_frame_id(frame_id)
                else:
                    # Discard any remaining packets from the sync procedure,
                    # which may include out-of-order packets:
                    continue
            self.last_frame_id = frame_id

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
            if frame_idx > end_after_idx:
                self.buffered_tile = dt
                return
            else:
                yield dt

    @property
    def nav_shape(self):
        return self.replica.store.state.nav_shape

    def state_is_running(self):
        if self.replica.store.state is None:
            return False
        return self.replica.store.state.processing is ProcessingState.RUNNING

    def state_is_connected(self):
        if self.replica.store.state is None:
            return False
        return self.replica.store.state.cam_connection is CamConnectionState.CONNECTED

    def main_loop(self, s, read_iter):
        self.replica.do_events()

        logger.info("MsgReaderThread: waiting for first packet(s)")

        while not self.state_is_running():
            # we are connected but aren't running a UDF yet, so we need to drain
            # the socket until we are told to run:
            if self.is_stopped():
                break
            _ = next(read_iter)
            self.replica.do_events()

        self.first_frame_id = self.sync(read_iter)
        self.last_frame_id = self.first_frame_id
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

        while True:
            num_frames = np.prod(self.nav_shape)
            frames_in_partition = min(frames_per_partition, num_frames - frame_counter)

            if frames_in_partition <= 0:
                # FIXME: if not in continuous mode, dispatch a stop event here
                frame_counter = 0
                epoch += 1
                self.first_frame_id = self.last_frame_id + 1
                continue

            self.replica.do_events()
            if self.is_stopped() or not self.state_is_running():
                return

            tiles = self.get_tiles(
                read_iter,
                end_after_idx=frame_counter + frames_in_partition
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

            runner = UDFRunner(self.replica.store.state.udfs)

            logger.debug(f"before run_for_partition {frame_counter}")
            result = runner.run_for_partition(
                partition=partition,
                corrections=None,
                roi=None,
                env=FakeEnvironment(threads_per_worker=1),
                tiling_scheme=tiling_scheme,
            )
            tiling_scheme = result.tiling_scheme
            logger.info("sending some result for %r (sector %d)", partition_slice, self.idx)
            self.out_queue.put(result)  # FIXME: replace with a zmq socket
            frame_counter += frames_in_partition


def get_settings_for_sector(idx):
    return {
        'idx': idx,  # zero-based index of sector
        'local_addr': '225.1.1.1',
        'port': 2001 + idx,
        'affinity_set': {8 + idx},
        'iface': 'veth2',
        # 'iface': 'enp193s0f0' if idx < 4 else 'enp193s0f1',
    }


class K2ListenerProcess(mp.Process):
    def __init__(self, idx, sync_state, out_queue, enable_tracing, *args, **kwargs):
        self.idx = idx
        self.sync_state = sync_state
        self.out_queue = out_queue
        self.enable_tracing = enable_tracing
        self._stop_event = mp.Event()
        super().__init__(*args, **kwargs)

    def stop(self):
        self._stop_event.set()

    def is_stopped(self):
        return self._stop_event.is_set()

    def run(self):
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

    def main(self):
        warmup()

        settings = get_settings_for_sector(self.idx)
        settings.update({
            'sync_state': self.sync_state,
            'out_queue': self.out_queue,
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
            logger.info(f"MySubProcess {self.idx} closing out_queue")
            self.out_queue.close()
            self.stop()
        logger.info(f"MySubProcess {self.idx} end of run()")
