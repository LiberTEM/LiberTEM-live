import logging
from typing import Optional, NamedTuple
from collections.abc import Iterator
import numpy as np
from libertem.common import Shape
from libertem.common.executor import (
    WorkerContext,
)
from libertem.io.dataset.base import (
    DataSetMeta, BasePartition, Partition, DataSet, TilingScheme,
)
from sparseconverter import ArrayBackend, NUMPY, CUDA

from libertem_live.detectors.base.acquisition import (
    AcquisitionMixin, GenericCommHandler, GetFrames,
)
from libertem_live.hooks import Hooks, ReadyForDataEnv
from .data import AcquisitionHeader
from .connection import MerlinDetectorConnection, MerlinPendingAcquisition
from .controller import MerlinActiveController

from opentelemetry import trace
import libertem_qd_mpx

tracer = trace.get_tracer(__name__)
logger = logging.getLogger(__name__)


class QdGetFrames(GetFrames):
    CAM_CLIENT_CLS = libertem_qd_mpx.CamClient
    FRAME_STACK_CLS = libertem_qd_mpx.QdFrameStack


class AcqState(NamedTuple):
    acq_header: AcquisitionHeader


class MerlinAcquisition(AcquisitionMixin, DataSet):
    '''
    Acquisition from a Quantum Detectors Merlin camera.

    Use :meth:`libertem_live.api.LiveContext.make_acquisition`
    to create an instance!

    Parameters
    ----------

    conn
        An existing :class:`MerlinDetectorConnection` instance
    nav_shape
        The navigation shape as a tuple, for example :code:`(height, width)`
        for a 2D STEM scan.
    frames_per_partition
        Number of frames to process before performing a merge operation. Decreasing this number
        increases the update rate, but can decrease performance.
    pending_aq
        A pending acquisition in passive mode, obtained from
        :meth:`MerlinDetectorConnection.wait_for_acquisition`.
        If this is not provided, it's assumed that the detector should be
        actively armed and triggered.
    controller
        A `MerlinActiveController` instance, which can be obtained
        from :meth:`MerlinDetectorConnection.get_active_controller`.
        You can pass additional parameters to
        :meth:`MerlinDetectorConnection.get_active_controller` in order
        to change detector settings.
        If no controller is passed in, and `pending_aq` is also not
        given, then the acquisition will be started in active
        mode, leaving all detector settings unchanged.
    hooks
        Acquisition hooks to react to certain events
    '''
    _conn: MerlinDetectorConnection
    _pending_aq: Optional[MerlinPendingAcquisition]

    def __init__(
        self,
        conn: MerlinDetectorConnection,
        nav_shape: Optional[tuple[int, ...]] = None,
        frames_per_partition: Optional[int] = None,
        pending_aq: Optional[MerlinPendingAcquisition] = None,
        controller: Optional[MerlinActiveController] = None,
        hooks: Optional[Hooks] = None,
    ):
        if frames_per_partition is None:
            frames_per_partition = 256
        if controller is None and pending_aq is None:
            controller = conn.get_active_controller()

        if hooks is None:
            hooks = Hooks()

        self._acq_state: Optional[AcqState] = None

        super().__init__(
            conn=conn,
            nav_shape=nav_shape,
            frames_per_partition=frames_per_partition,
            pending_aq=pending_aq,
            controller=controller,
            hooks=hooks,
        )

    def initialize(self, executor):
        ''
        # FIXME: possibly need to have an "acquisition plan" object
        # so we know all relevant parameters beforehand
        sig_shape = self._conn.read_sig_shape()

        bitdepth = self._conn.read_bitdepth()
        if bitdepth in (1, 6):
            dtype = np.uint8
        elif bitdepth == 12:
            dtype = np.uint16
        elif bitdepth == 24:
            dtype = np.uint32
        else:
            raise RuntimeError(f"unknown COUNTERDEPTH {bitdepth}! should be in (1,6,16,24)")

        self._meta = DataSetMeta(
            shape=Shape(self._nav_shape + sig_shape, sig_dims=2),
            raw_dtype=dtype,
            dtype=dtype,
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

    def start_acquisition(self):
        if self._pending_aq is None:
            # active case, we manage the connection:
            self._conn.connect()
            with tracer.start_as_current_span("MerlinAcquisition.trigger"):
                self._hooks.on_ready_for_data(ReadyForDataEnv(aq=self))

            acq_header = self._conn.wait_for_acquisition()

            self._acq_state = AcqState(
                acq_header=acq_header,
            )
        else:
            # passive case, we don't manage the connection and we definitely
            # don't drain anything out of the socket!
            acq_header = self._pending_aq.header
            self._acq_state = AcqState(
                acq_header=acq_header,
            )

    def end_acquisition(self):
        # active case, we manage the connection:
        if self._pending_aq is None:
            self._conn.close()
        self._acq_state = None

    def check_valid(self):
        ''
        pass

    def need_decode(self, read_dtype, roi, corrections):
        ''
        return True  # FIXME: we just do this to get a large tile size

    def adjust_tileshape(self, tileshape, roi):
        ''
        depth = 24
        return (depth, *self.meta.shape.sig)
        # return Shape((self._end_idx - self._start_idx, 256, 256), sig_dims=2)

    def get_max_io_size(self):
        ''
        # return 12*256*256*8
        # FIXME magic numbers?
        return 24*np.prod(self.meta.shape.sig)*8

    def get_base_shape(self, roi):
        return (1, 1, self.meta.shape.sig[-1])

    def get_partitions(self):
        ''
        num_frames = np.prod(self._nav_shape, dtype=np.uint64)
        num_partitions = int(num_frames // self._frames_per_partition)

        slices = BasePartition.make_slices(self.shape, num_partitions)
        for part_slice, start, stop in slices:
            yield MerlinLivePartition(
                start_idx=start,
                end_idx=stop,
                meta=self._meta,
                partition_slice=part_slice,
            )

    def get_task_comm_handler(self) -> "MerlinCommHandler":
        return MerlinCommHandler(
            conn=self._conn,
        )


def _accum_partition(
    frames_iter: Iterator[tuple[np.ndarray, int]],
    tiling_scheme: TilingScheme,
    sig_shape: tuple[int, ...],
    dtype
) -> np.ndarray:
    """
    Accumulate frame stacks for a partition
    """
    out = np.zeros((tiling_scheme.depth,) + sig_shape, dtype=dtype)
    offset = 0
    for frame_stack, _ in frames_iter:
        out[offset:offset+frame_stack.shape[0]] = frame_stack
        offset += frame_stack.shape[0]
    assert offset == tiling_scheme.depth
    return out


class MerlinCommHandler(GenericCommHandler):
    def get_conn_impl(self):
        return self._conn.get_conn_impl()


class MerlinLivePartition(Partition):
    def __init__(
        self, start_idx, end_idx, partition_slice, meta,
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

    def _get_tiles_fullframe(self, tiling_scheme: TilingScheme, dest_dtype="float32", roi=None,
            array_backend: Optional["ArrayBackend"] = None):
        assert array_backend in (None, NUMPY, CUDA)
        # assert len(tiling_scheme) == 1
        tiling_scheme = tiling_scheme.adjust_for_partition(self)
        logger.debug("reading up to frame idx %d for this partition", self._end_idx)
        to_read = self._end_idx - self._start_idx

        with QdGetFrames(
            request_queue=self._worker_context.get_worker_queue(),
            dtype=dest_dtype,
            sig_shape=tuple(tiling_scheme[0].shape),
        ) as frames:
            yield from frames.get_tiles(
                to_read=to_read,
                start_idx=self._start_idx,
                tiling_scheme=tiling_scheme,
                roi=roi,
                array_backend=array_backend,
            )

    def get_tiles(self, tiling_scheme, dest_dtype="float32", roi=None,
            array_backend: Optional["ArrayBackend"] = None):
        yield from self._get_tiles_fullframe(
            tiling_scheme, dest_dtype, roi, array_backend=array_backend
        )

    def __repr__(self):
        return f"<MerlinLivePartition {self._start_idx}:{self._end_idx}>"
