import base64
import logging
from typing import Optional, TYPE_CHECKING
from collections.abc import Sequence

import numpy as np
from opentelemetry import trace

from libertem.common import Shape
from libertem.common.math import prod
from libertem.common.executor import (
    WorkerContext,
)
from libertem.io.dataset.base import (
    DataSetMeta, BasePartition, Partition, DataSet, TilingScheme,
)
from libertem.corrections.corrset import CorrectionSet
from sparseconverter import ArrayBackend, NUMPY, CUDA, CUPY

from libertem_live.detectors.base.acquisition import (
    AcquisitionMixin, GenericCommHandler, GetFrames
)
from libertem_live.hooks import Hooks, ReadyForDataEnv
from .controller import DectrisActiveController
from .common import AcquisitionParams, DetectorConfig
from .DEigerClient import DEigerClient

import libertem_dectris

if TYPE_CHECKING:
    from .connection import DectrisDetectorConnection
    from .common import DectrisPendingAcquisition

tracer = trace.get_tracer(__name__)
logger = logging.getLogger(__name__)


def get_darray(darray) -> np.ndarray:
    """
    Helper to decode darray-encoded arrays to numpy
    """
    data = darray['value']['data']
    if 'base64' not in darray['value']['filters']:
        raise RuntimeError("don't unterstand this encoding, bailing")
    shape = tuple(reversed(darray['value']['shape']))
    dtype = darray['value']['type']
    data = base64.decodebytes(data.encode("ascii"))
    return np.frombuffer(data, dtype=dtype).reshape(shape)


def dtype_from_frame(frame):
    dtype = np.dtype(frame.get_pixel_type()).newbyteorder(
        frame.get_endianess()
    )
    return dtype


def shape_from_frame(frame):
    return tuple(reversed(frame.get_shape()))


class DectrisGetFrames(GetFrames):
    CAM_CLIENT_CLS = libertem_dectris.CamClient
    FRAME_STACK_CLS = libertem_dectris.FrameStackHandle


class DectrisCommHandler(GenericCommHandler):
    def __init__(
        self,
        params: Optional[AcquisitionParams],
        conn: "DectrisDetectorConnection",
        controller: "Optional[DectrisActiveController]",
    ):
        super().__init__(conn=conn)
        self.params = params
        self.controller = controller

    def get_conn_impl(self):
        return self._conn.get_conn_impl()

    def start(self):
        if self.controller is not None and self.params is not None:
            logger.info(f"arming for acquisition with id={self.params.sequence_id}")
            self.controller.handle_start(self._conn, self.params.sequence_id)

    def done(self):
        if self.controller is not None:
            self.controller.handle_stop(self._conn)


# FIXME: naming: native `DetectorConnection` vs this is confusing?
class DectrisAcquisition(AcquisitionMixin, DataSet):
    '''
    Acquisition from a DECTRIS detector.
    Use :meth:`libertem_live.api.LiveContext.make_acquisition` to instantiate this
    class!

    Examples
    --------

    >>> with ctx.make_connection('dectris').open(
    ...     api_host='127.0.0.1',
    ...     api_port=DCU_API_PORT,
    ...     data_host='127.0.0.1',
    ...     data_port=DCU_DATA_PORT,
    ... ) as conn:
    ...     aq = ctx.make_acquisition(
    ...         conn=conn,
    ...         nav_shape=(128, 128),
    ...     )

    Parameters
    ----------
    conn
        An existing `DectrisDetectorConnection` instance
    nav_shape
        The navigation shape as a tuple, for example :code:`(height, width)`
        for a 2D STEM scan.
    frames_per_partition
        A tunable for configuring the feedback rate - more frames per partition
        means slower feedback, but less computational overhead. Might need to be tuned
        to adapt to the dwell time.
    pending_aq
        A pending acquisition in passive mode, obtained from
        :meth:`DectrisDetectorConnection.wait_for_acquisition`.
        If this is not provided, it's assumed that the detector should be
        actively armed and triggered.
    controller
        A `DectrisActiveController` instance, which can be obtained
        from :meth:`DectrisDetectorConnection.get_active_controller`.
        You can pass additional parameters to
        :meth:`DectrisDetectorConnection.get_active_controller` in order
        to change detector settings.
        If no controller is passed in, and `pending_aq` is also not
        given, then the acquisition will be started in active
        mode, leaving all detector settings unchanged.
    hooks
        Acquisition hooks to react to certain events
    '''
    _conn: "DectrisDetectorConnection"
    _controller: "DectrisActiveController"
    _pending_aq: "DectrisPendingAcquisition"

    def __init__(
        self,

        # this replaces the {api,data}_{host,port} parameters:
        conn: "DectrisDetectorConnection",

        nav_shape: Optional[tuple[int, ...]] = None,
        frames_per_partition: int = 128,

        # in passive mode, we get this:
        pending_aq: Optional["DectrisPendingAcquisition"] = None,

        # in passive mode, we don't pass the controller.
        # (in active mode, you _can_ pass it, but if you don't, a default
        # controller will be created):
        controller: Optional[DectrisActiveController] = None,

        hooks: Optional[Hooks] = None,
    ):
        if frames_per_partition is None:
            frames_per_partition = 512

        if controller is None and pending_aq is None:
            # default to active mode, with no settings to change:
            controller = conn.get_active_controller()

        if pending_aq is not None:
            if controller is not None:
                raise ValueError(
                    "Got both active `controller` and pending acquisition, please only provide one"
                )

        if hooks is None:
            hooks = Hooks()

        super().__init__(
            conn=conn,
            nav_shape=nav_shape,
            frames_per_partition=frames_per_partition,
            hooks=hooks,
            controller=controller,
            pending_aq=pending_aq,
        )

        self._sig_shape: tuple[int, ...] = ()
        self._acq_state: Optional[AcquisitionParams] = None

    def get_api_client(self) -> DEigerClient:
        '''
        Get an API client, which can be used to configure the detector.
        '''
        return self._conn.get_api_client()

    def get_detector_config(self) -> DetectorConfig:
        '''
        Get (very limited subset of) detector configuration
        '''
        ec = self._conn.get_api_client()
        # FIXME: initialize detector here, if not already initialized?
        shape_x = ec.detectorConfig("x_pixels_in_detector")['value']
        shape_y = ec.detectorConfig("y_pixels_in_detector")['value']
        bit_depth = ec.detectorConfig("bit_depth_image")['value']
        return DetectorConfig(
            x_pixels_in_detector=shape_x,
            y_pixels_in_detector=shape_y,
            bit_depth=bit_depth,
        )

    def initialize(self, executor) -> "DataSet":
        ''
        self._update_meta()
        return self

    def _update_meta(self):
        dc = self.get_detector_config()
        dtypes = {
            8: np.uint8,
            16: np.uint16,
            32: np.uint32,
        }
        try:
            dtype = dtypes[dc.bit_depth]
        except KeyError:
            raise Exception(f"unknown bit depth: {dc.bit_depth}")
        self._sig_shape = (dc.y_pixels_in_detector, dc.x_pixels_in_detector)
        self._meta = DataSetMeta(
            shape=Shape(self._nav_shape + self._sig_shape, sig_dims=2),
            array_backends=self.array_backends,
            raw_dtype=dtype,
            dtype=dtype,
        )

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

    @property
    def array_backends(self) -> Sequence[ArrayBackend]:
        return (NUMPY, CUDA, CUPY)

    def get_correction_data(self):
        ''
        if self._controller is None or not self._controller.enable_corrections:
            return CorrectionSet()
        ec = self._conn.get_api_client()
        mask = ec.detectorConfig("pixel_mask")
        mask_arr = get_darray(mask)
        excluded_pixels = mask_arr > 0
        return CorrectionSet(excluded_pixels=excluded_pixels)

    def start_acquisition(self):
        ''
        with tracer.start_as_current_span('start_acquisition'):
            if self._controller is not None:
                self._conn.prepare_for_active()
                self._controller.apply_file_writing()
                self._controller.apply_scan_settings(self._nav_shape)
                self._controller.apply_misc_settings()
                self._update_meta()
                sequence_id = self._controller.arm()
                nimages = prod(self.shape.nav)
                self._acq_state = AcquisitionParams(
                    sequence_id=sequence_id,
                    nimages=nimages,
                )
                with tracer.start_as_current_span("DectrisAcquisition.trigger"):
                    self._hooks.on_ready_for_data(ReadyForDataEnv(aq=self))
            else:
                nimages = prod(self.shape.nav)
                sequence_id = self._pending_aq.series
                self._acq_state = AcquisitionParams(
                    sequence_id=sequence_id,
                    nimages=nimages,
                )

    def end_acquisition(self):
        return

    def check_valid(self):
        ''
        pass

    def need_decode(self, read_dtype, roi, corrections):
        ''
        return True

    def get_base_shape(self, roi):
        ''
        return (1, *self.meta.shape.sig)

    @property
    def acquisition_state(self):
        ''
        return self._acq_state

    def get_partitions(self):
        ''
        num_frames = np.prod(self._nav_shape, dtype=np.uint64)
        num_partitions = int(num_frames // self._frames_per_partition)

        slices = BasePartition.make_slices(self.shape, num_partitions)

        for part_slice, start, stop in slices:
            yield DectrisLivePartition(
                start_idx=start,
                end_idx=stop,
                meta=self._meta,
                partition_slice=part_slice,
            )

    def get_task_comm_handler(self) -> "DectrisCommHandler":
        ''
        return DectrisCommHandler(
            params=self._acq_state,
            conn=self._conn,
            controller=self._controller,
        )


class DectrisLivePartition(Partition):
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

    def _get_tiles_fullframe(
        self,
        tiling_scheme: TilingScheme,
        dest_dtype="float32",
        roi=None,
        array_backend: Optional["ArrayBackend"] = None
    ):
        assert array_backend in (None, NUMPY, CUDA, CUPY)
        assert len(tiling_scheme) == 1, "only supports full frames tiling scheme for now"
        logger.debug("reading up to frame idx %d for this partition", self._end_idx)
        to_read = self._end_idx - self._start_idx
        with DectrisGetFrames(
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
            tiling_scheme, dest_dtype, roi,
            array_backend=array_backend
        )

    def __repr__(self):
        return f"<DectrisLivePartition {self._start_idx}:{self._end_idx}>"
