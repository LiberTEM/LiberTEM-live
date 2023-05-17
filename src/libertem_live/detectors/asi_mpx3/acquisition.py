from contextlib import contextmanager
import logging
import time
from typing import Tuple, Optional

import numpy as np
from opentelemetry import trace

from libertem.common import Shape, Slice
from libertem.common.math import prod
from libertem.common.executor import WorkerContext, TaskProtocol, WorkerQueue, TaskCommHandler
from libertem.io.dataset.base import (
    DataTile, DataSetMeta, BasePartition, Partition, DataSet, TilingScheme,
)
from libertem.corrections.corrset import CorrectionSet

from libertem_live.detectors.base.acquisition import AcquisitionMixin
from libertem_live.detectors.base.controller import AcquisitionController
from libertem_live.hooks import ReadyForDataEnv, Hooks
from .connection import AsiMpx3DetectorConnection, AsiMpx3PendingAcquisition

from libertem_asi_mpx3 import CamClient, FrameStackHandle

tracer = trace.get_tracer(__name__)
logger = logging.getLogger(__name__)


def get_frame_stacks(worker_context: WorkerContext):
    """
    Consume all FRAME_STACK messages from the request queue until we get an
    END_PARTITION message (which we also consume)
    """
    request_queue = worker_context.get_worker_queue()

    with request_queue.get() as msg:
        header, _ = msg
        header_type = header["type"]
        assert header_type == "BEGIN_TASK"
        socket_path = header["socket"]
        cam_client = CamClient(socket_path)
    try:
        while True:
            with request_queue.get() as msg:
                header, payload = msg
                header_type = header["type"]
                if header_type == "FRAME_STACK":
                    frame_stack = FrameStackHandle.deserialize(payload)
                    frames = cam_client.get_frames(handle=frame_stack)
                    try:
                        yield frames
                    finally:
                        cam_client.done(frame_stack)
                elif header_type == "END_PARTITION":
                    return
                else:
                    raise RuntimeError(
                        f"invalid header type {header['type']}; "
                        f"FRAME_STACK or END_PARTITION expected"
                    )
    finally:
        cam_client.close()


class AsiCommHandler(TaskCommHandler):
    def __init__(
        self,
        conn: "AsiMpx3DetectorConnection",
    ):
        self.conn = conn.get_conn_impl()

    def handle_task(self, task: TaskProtocol, queue: WorkerQueue):
        with tracer.start_as_current_span("AsiCommHandler.handle_task") as span:
            span.set_attribute(
                "libertem_live.detectors.asi:socket",
                self.conn.get_socket_path(),
            )
            queue.put({
                "type": "BEGIN_TASK",
                "socket": self.conn.get_socket_path(),
            })
            put_time = 0.0
            recv_time = 0.0
            # send the data for this task to the given worker
            partition = task.get_partition()
            slice_ = partition.slice
            start_idx = slice_.origin[0]
            end_idx = slice_.origin[0] + slice_.shape[0]
            span.set_attributes({
                "libertem.partition.start_idx": start_idx,
                "libertem.partition.end_idx": end_idx,
            })
            stack_size = 32
            current_idx = start_idx
            while current_idx < end_idx:
                current_stack_size = min(stack_size, end_idx - current_idx)

                t0 = time.perf_counter()
                frame_stack = self.conn.get_next_stack(
                    max_size=current_stack_size
                )
                assert len(frame_stack) <= current_stack_size,\
                    f"{len(frame_stack)} <= {current_stack_size}"
                t1 = time.perf_counter()
                recv_time += t1 - t0
                if frame_stack is None:
                    if current_idx != end_idx:
                        raise RuntimeError("premature end of frame iterator")
                    break

                t0 = time.perf_counter()
                serialized = frame_stack.serialize()
                queue.put({
                    "type": "FRAME_STACK",
                }, payload=serialized)
                t1 = time.perf_counter()
                put_time += t1 - t0

                current_idx += len(frame_stack)
            span.set_attributes({
                "total_put_time": put_time,
                "total_recv_time": recv_time,
            })

    def start(self):
        pass  # nothing to do here really

    def done(self):
        pass  # ... likewise here


class AsiMpx3Acquisition(AcquisitionMixin, DataSet):
    '''
    Acquisition from a ASI MPX3 detector

    Use :meth:`libertem_live.api.LiveContext.make_acquisition` to instantiate this
    class!

    Examples
    --------

    >>> with ctx.make_connection('asi_mpx3').open(
    ...     data_port=MPX3_PORT,
    ... ) as conn:
    ...     pending_aq = conn.wait_for_acquisition(1.0)
    ...     aq = ctx.make_acquisition(
    ...         conn=conn,
    ...         pending_aq=pending_aq,
    ...     )

    Parameters
    ----------
    conn
        An existing `AsiDetectorConnection` instance
    nav_shape
        The number of scan positions as a 2-tuple :code:`(height, width)`
    frames_per_partition
        A tunable for configuring the feedback rate - more frames per partition
        means slower feedback, but less computational overhead. Might need to be tuned
        to adapt to the dwell time.
    pending_aq
        A pending acquisition in passive mode, obtained from
        :meth:`AsiMpx3DetectorConnection.wait_for_acquisition`.
    hooks
        Acquisition hooks to react to certain events
    '''
    def __init__(
        self,

        conn: AsiMpx3DetectorConnection,

        hooks: Optional[Hooks] = None,

        # in passive mode, we get this:
        pending_aq: Optional[AsiMpx3PendingAcquisition] = None,

        # this is for future compatibility with an active mode:
        controller: Optional[AcquisitionController] = None,

        nav_shape: Optional[Tuple[int, ...]] = None,

        frames_per_partition: Optional[int] = None,
    ):
        assert pending_aq is not None, "only supporting passive mode for now"
        assert controller is None, "only supporting passive mode for now"
        if frames_per_partition is None:
            frames_per_partition = 4096
        super().__init__(
            conn=conn,
            frames_per_partition=frames_per_partition,
            nav_shape=nav_shape,
            controller=None,
            pending_aq=pending_aq,
            hooks=hooks,
        )
        sig_shape = pending_aq.sig_shape
        assert sig_shape is not None, "sig_shape must be known"
        self._sig_shape: Tuple[int, ...] = sig_shape

    def initialize(self, executor) -> "DataSet":
        ''
        self._meta = DataSetMeta(
            shape=Shape(self._nav_shape + self._sig_shape, sig_dims=2),
            raw_dtype=np.float32,  # this is a lie, the dtype can vary by frame!
            dtype=np.float32,
        )
        return self

    @property
    def dtype(self):
        return self._meta.dtype

    @property
    def raw_dtype(self):
        return self._meta.raw_dtype

    @property
    def shape(self):
        return self._meta.shape

    @property
    def meta(self):
        return self._meta

    def get_correction_data(self):
        ''
        return CorrectionSet()

    @contextmanager
    def _acquire_active(self):
        raise NotImplementedError("")

    @contextmanager
    def acquire(self):
        with tracer.start_as_current_span('acquire'):
            if self._pending_aq is None:  # active mode:
                with tracer.start_as_current_span("AsiAcquisition.on_ready_for_data"):
                    self._hooks.on_ready_for_data(ReadyForDataEnv(aq=self))
            else:
                pass  # passive mode
            yield

    def check_valid(self):
        ""
        pass

    def need_decode(self, read_dtype, roi, corrections):
        ""
        return True  # FIXME: we just do this to get a large tile size

    def adjust_tileshape(self, tileshape, roi):
        ""
        # depth = 512  # FIXME: hardcoded, hmm...
        # return (depth, *self.meta.shape.sig)
        # return Shape((self._end_idx - self._start_idx, 256, 256), sig_dims=2)
        return super().adjust_tileshape(tileshape, roi)

    def get_max_io_size(self):
        ""
        # return 12*256*256*8
        # FIXME magic numbers?
        return 1200000 * prod(self.meta.shape.sig) * 8

    def get_base_shape(self, roi):
        ""
        return (1, *self.meta.shape.sig)

    def get_partitions(self):
        ""
        # FIXME: only works for inline executor or similar, as we are using a zeromq socket
        # which is not safe to be passed to other threads
        num_frames = np.prod(self._nav_shape, dtype=np.uint64)
        num_partitions = int(num_frames // self._frames_per_partition)

        slices = BasePartition.make_slices(self.shape, num_partitions)

        for part_slice, start, stop in slices:
            yield AsiLivePartition(
                start_idx=start,
                end_idx=stop,
                meta=self._meta,
                partition_slice=part_slice,
            )

    def get_task_comm_handler(self) -> "AsiCommHandler":
        return AsiCommHandler(
            conn=self._conn,
        )


class AsiLivePartition(Partition):
    def __init__(
        self, start_idx, end_idx, partition_slice,
        meta,
    ):
        super().__init__(meta=meta, partition_slice=partition_slice, io_backend=None, decoder=None)
        self._start_idx = start_idx
        self._end_idx = end_idx

    def shape_for_roi(self, roi):
        return self.slice.adjust_for_roi(roi).shape

    @property
    def shape(self):
        return self.slice.shape

    @property
    def dtype(self):
        return self.meta.raw_dtype

    def set_corrections(self, corrections):
        self._corrections = corrections

    def set_worker_context(self, worker_context: WorkerContext):
        self._worker_context = worker_context

    def _preprocess(self, tile_data, tile_slice):
        if self._corrections is None:
            return
        self._corrections.apply(tile_data, tile_slice)

    def _get_tiles_straight(
        self,
        tiling_scheme: TilingScheme,
        dest_dtype="float32",
        roi=None,
        array_backend=None
    ):
        assert len(tiling_scheme) == 1, "only supports full frames tiling scheme for now"
        logger.debug("reading up to frame idx %d for this partition", self._end_idx)
        to_read = self._end_idx - self._start_idx
        depth = tiling_scheme.depth
        tile_start = self._start_idx
        sig_shape = tuple(self.shape.sig)
        stacks = get_frame_stacks(self._worker_context)
        buf = np.zeros((depth,) + tuple(tiling_scheme[0].shape), dtype=dest_dtype)
        buf_cursor = 0
        while to_read > 0:
            try:
                stack = next(stacks)
            except StopIteration:
                assert to_read == 0, f"we were still expecting to read {to_read} frames more!"
                break

            for mem, dtype in stack:
                dtype = dtype.as_string()
                frame_arr = np.frombuffer(mem, dtype=dtype).reshape(sig_shape)
                buf[buf_cursor] = frame_arr
                buf_cursor += 1
                to_read -= 1
                if buf_cursor == depth or to_read == 0:
                    frames_in_tile = buf_cursor
                    buf_cut = buf[:frames_in_tile]
                    tile_shape = Shape(
                        (frames_in_tile,) + tuple(tiling_scheme[0].shape),
                        sig_dims=2
                    )
                    tile_slice = Slice(
                        origin=(tile_start,) + (0, 0),
                        shape=tile_shape,
                    )
                    self._preprocess(buf_cut, tile_slice)
                    yield DataTile(
                        buf_cut,
                        tile_slice=tile_slice,
                        scheme_idx=0,
                    )
                    tile_start += frames_in_tile

        logger.debug("LivePartition.get_tiles: end of method")

    def get_tiles(self, tiling_scheme, dest_dtype="float32", roi=None, array_backend=None):
        yield from self._get_tiles_straight(tiling_scheme, dest_dtype, roi, array_backend)

    def __repr__(self):
        return f"<AsiLivePartition {self._start_idx}:{self._end_idx}>"
