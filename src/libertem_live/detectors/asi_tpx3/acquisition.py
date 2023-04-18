from contextlib import contextmanager
import logging
import time
from typing import Tuple, Optional

import scipy
import scipy.sparse
import numpy as np
from opentelemetry import trace
from sparseconverter import SCIPY_CSR

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
from .connection import AsiTpx3DetectorConnection, AsiTpx3PendingAcquisition

from libertem_asi_tpx3 import CamClient, ChunkStackHandle

tracer = trace.get_tracer(__name__)
logger = logging.getLogger(__name__)


def get_chunk_stacks(worker_context: WorkerContext):
    """
    Consume all CHUNK_STACK messages from the request queue until we get an
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
                if header_type == "CHUNK_STACK":
                    chunk_stack = ChunkStackHandle.deserialize(payload)
                    chunks = cam_client.get_chunks(handle=chunk_stack)
                    try:
                        yield chunks
                    finally:
                        cam_client.done(chunk_stack)
                elif header_type == "END_PARTITION":
                    return
                else:
                    raise RuntimeError(
                        f"invalid header type {header['type']}; "
                        f"CHUNK_STACK or END_PARTITION expected"
                    )
    finally:
        cam_client.close()


class AsiCommHandler(TaskCommHandler):
    def __init__(
        self,
        conn: "AsiTpx3DetectorConnection",
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
            stack_size = 1000000000000
            current_idx = start_idx
            while current_idx < end_idx:
                current_stack_size = min(stack_size, end_idx - current_idx)

                t0 = time.perf_counter()
                chunk_stack = self.conn.get_next_stack(
                    max_size=current_stack_size
                )
                assert len(chunk_stack) <= current_stack_size,\
                    f"{len(chunk_stack)} <= {current_stack_size}"
                t1 = time.perf_counter()
                recv_time += t1 - t0
                if chunk_stack is None:
                    if current_idx != end_idx:
                        raise RuntimeError("premature end of frame iterator")
                    break

                t0 = time.perf_counter()
                serialized = chunk_stack.serialize()
                queue.put({
                    "type": "CHUNK_STACK",
                }, payload=serialized)
                t1 = time.perf_counter()
                put_time += t1 - t0

                current_idx += len(chunk_stack)
            span.set_attributes({
                "total_put_time": put_time,
                "total_recv_time": recv_time,
            })

    def start(self):
        pass  # nothing to do here really

    def done(self):
        pass  # ... likewise here


class AsiTpx3Acquisition(AcquisitionMixin, DataSet):
    '''
    Acquisition from a ASI TPX3 detector

    Use :meth:`libertem_live.api.LiveContext.make_acquisition` to instantiate this
    class!

    Examples
    --------

    >>> with ctx.make_connection('asi_tpx3').open(
    ...     data_port=TPX3_PORT,
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
        :meth:`AsiTpx3DetectorConnection.wait_for_acquisition`.
    hooks
        Acquisition hooks to react to certain events
    '''
    def __init__(
        self,

        conn: AsiTpx3DetectorConnection,

        hooks: Optional[Hooks] = None,

        # in passive mode, we get this:
        pending_aq: Optional[AsiTpx3PendingAcquisition] = None,

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
        self._sig_shape: Tuple[int, ...] = ()
        self._acquisition_header = pending_aq.header

    def initialize(self, executor) -> "DataSet":
        ''
        self._sig_shape = self._acquisition_header.get_sig_shape()
        self._nav_shape = self._acquisition_header.get_nav_shape()
        self._meta = DataSetMeta(
            shape=Shape(self._nav_shape + self._sig_shape, sig_dims=2),
            array_backends=[SCIPY_CSR],
            raw_dtype=np.float32,  # this is a lie, the dtype can vary by chunk!
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
        depth = 512  # FIXME: hardcoded, hmm...
        return (depth, *self.meta.shape.sig)
        # return Shape((self._end_idx - self._start_idx, 256, 256), sig_dims=2)

    def get_max_io_size(self):
        ""
        # return 12*256*256*8
        # FIXME magic numbers?
        return 1200000*np.prod(self.meta.shape.sig)*8

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
        # FIXME: use depth?
        # depth = tiling_scheme.depth
        tile_start = self._start_idx
        stacks = get_chunk_stacks(self._worker_context)
        sig_shape = tuple(self.slice.shape.sig)
        sig_dims = len(sig_shape)
        buf = None
        while to_read > 0:
            try:
                stack = next(stacks)
            except StopIteration:
                assert to_read == 0, f"we were still expecting to read {to_read} frames more!"

            for (
                chunk_layout,
                chunk_indptr,
                chunk_indices,
                chunk_values
            ) in stack:
                chunk_values_arr = np.frombuffer(chunk_values, dtype=chunk_layout.get_value_dtype())
                chunk_indices_arr = np.frombuffer(
                    chunk_indices, dtype=chunk_layout.get_indices_dtype()
                )
                chunk_indptr_arr = np.frombuffer(
                    chunk_indptr, dtype=chunk_layout.get_indptr_dtype()
                )
                if chunk_layout.get_value_dtype() != np.dtype(dest_dtype):
                    if buf is None or buf.shape[0] < chunk_values_arr.shape[0]:
                        buf = np.zeros_like(chunk_values_arr, dtype=dest_dtype)
                    buf[:chunk_values_arr.shape[0]] = chunk_values_arr
                    chunk_values_arr = buf[:chunk_values_arr.shape[0]]
                arr = scipy.sparse.csr_matrix(
                    (chunk_values_arr, chunk_indices_arr, chunk_indptr_arr),
                    shape=(chunk_layout.get_nframes(), prod(self.slice.shape.sig)),
                )
                frames_in_tile = chunk_layout.get_nframes()
                to_read -= frames_in_tile

                tile_slice = Slice(
                    origin=(tile_start, ) + (0, ) * sig_dims,
                    shape=Shape((arr.shape[0], ) + sig_shape, sig_dims=sig_dims),
                )
                assert tile_slice.origin[0] < self._end_idx
                yield DataTile(
                    data=arr,
                    tile_slice=tile_slice,
                    scheme_idx=0,
                )

                tile_start += frames_in_tile
        logger.debug("LivePartition.get_tiles: end of method")

    def get_tiles(self, tiling_scheme, dest_dtype="float32", roi=None, array_backend=None):
        yield from self._get_tiles_straight(tiling_scheme, dest_dtype, roi, array_backend)

    def __repr__(self):
        return f"<AsiLivePartition {self._start_idx}:{self._end_idx}>"
