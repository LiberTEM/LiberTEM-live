from contextlib import contextmanager
import math
import os
import base64
import logging
import time
from typing import Optional, Tuple
import tempfile

import numpy as np
from opentelemetry import trace

from libertem.common import Shape, Slice
from libertem.common.math import prod
from libertem.common.executor import WorkerContext, TaskProtocol, WorkerQueue, TaskCommHandler
from libertem.io.dataset.base import (
    DataTile, DataSetMeta, BasePartition, Partition, DataSet, TilingScheme,
)
from libertem.corrections.corrset import CorrectionSet
from sparseconverter import ArrayBackend, NUMPY, CUDA

from libertem_live.detectors.base.acquisition import AcquisitionMixin
from libertem_live.detectors.base.connection import (
    PendingAcquisition, DetectorConnection,
)
from .DEigerClient import DEigerClient
from .controller import DectrisActiveController
from .common import AcquisitionParams, DetectorConfig, TriggerMode

import libertem_dectris

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


def get_frames(request_queue):
    """
    Consume all FRAMES messages from the request queue until we get an
    END_PARTITION message (which we also consume)
    """
    buf = None
    depth = 0
    with request_queue.get() as msg:
        header, _ = msg
        header_type = header["type"]
        assert header_type == "BEGIN_TASK"
        socket_path = header["socket"]
        cam_client = libertem_dectris.CamClient(socket_path)
    try:
        while True:
            with request_queue.get() as msg:
                header, payload = msg
                header_type = header["type"]
                if header_type == "FRAMES":
                    frame_stack = libertem_dectris.FrameStackHandle.deserialize(payload)
                    if buf is None or len(frame_stack) > buf.shape[0]:
                        depth = len(frame_stack)
                        buf = np.zeros((depth,) + header['shape'], header['dtype'])
                    cam_client.decompress_frame_stack(frame_stack, out=buf)
                    buf_reshaped = buf.reshape((depth,) + tuple(header['shape']))
                    try:
                        yield buf_reshaped[:len(frame_stack)]
                    finally:
                        cam_client.done(frame_stack)
                elif header_type == "END_PARTITION":
                    # print(f"partition {partition} done")
                    return
                else:
                    raise RuntimeError(
                        f"invalid header type {header['type']}; FRAME or END_PARTITION expected"
                    )
    finally:
        cam_client.close()


class DectrisCommHandler(TaskCommHandler):
    def __init__(
        self,
        params: Optional[AcquisitionParams],
        conn: "DectrisDetectorConnection",
        controller: "Optional[DectrisActiveController]",
    ):
        self.params = params
        self.conn = conn
        self.controller = controller

    def handle_task(self, task: TaskProtocol, queue: WorkerQueue):
        conn = self.conn.get_conn_impl()
        with tracer.start_as_current_span("DectrisCommHandler.handle_task") as span:
            span.set_attribute(
                "libertem_live.detectors.dectris:socket",
                conn.get_socket_path(),
            )
            queue.put({
                "type": "BEGIN_TASK",
                "socket": conn.get_socket_path(),
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
            chunk_size = 128
            current_idx = start_idx
            while current_idx < end_idx:
                current_chunk_size = min(chunk_size, end_idx - current_idx)

                t0 = time.perf_counter()
                frame_stack = conn.get_next_stack(
                    max_size=current_chunk_size
                )
                assert len(frame_stack) <= current_chunk_size,\
                    f"{len(frame_stack)} <= {current_chunk_size}"
                t1 = time.perf_counter()
                recv_time += t1 - t0
                if len(frame_stack) == 0:
                    if current_idx != end_idx:
                        raise RuntimeError("premature end of frame iterator")
                    break

                dtype = dtype_from_frame(frame_stack)
                shape = shape_from_frame(frame_stack)
                t0 = time.perf_counter()
                serialized = frame_stack.serialize()
                queue.put({
                    "type": "FRAMES",
                    "dtype": dtype,
                    "shape": shape,
                    "encoding": frame_stack.get_encoding(),
                }, payload=serialized)
                t1 = time.perf_counter()
                put_time += t1 - t0

                current_idx += len(frame_stack)
            span.set_attributes({
                "total_put_time": put_time,
                "total_recv_time": recv_time,
            })

    def start(self):
        if self.controller is not None and self.params is not None:
            print(f"arming for acquisition with id={self.params.sequence_id}")
            self.controller.handle_start(self.conn, self.params.sequence_id)

    def done(self):
        if self.controller is not None:
            self.controller.handle_stop(self.conn)


class DectrisPendingAcquisition(PendingAcquisition):
    def __init__(self, detector_config, series):
        self._detector_config = detector_config
        self._series = series

    @property
    def detector_config(self):
        return self._detector_config

    @property
    def series(self):
        return self._series

    def create_acquisition(self, *args, **kwargs):
        aq = DectrisAcquisition(pending_aq=self, *args, **kwargs)
        return aq

    def __repr__(self):
        return f"<DectrisPendingAcquisition series={self.series} config={self.detector_config}>"


# FIXME: naming: native `DetectorConnection` vs this is confusing?
class DectrisDetectorConnection(DetectorConnection):
    '''
    Connect to a DECTRIS DCU, both for detector configuration and for accessing
    the data stream.

    Parameters
    ----------
    api_host
        The hostname or IP address of the DECTRIS DCU for the REST API
    api_port
        The port of the REST API
    data_host
        The hostname or IP address of the DECTRIS DCU for the zeromq data stream
    data_port
        The zeromq port to use
    buffer_size
        The total receive buffer in MiB that is used to stream data to worker
        processes.
    bytes_per_frame
        Roughtly how many bytes should be reserved per frame

        If this is None, a rough guess will be calculated from the detector size.
        You can check the :py:attr:`~bytes_per_frame` property to see if the guess
        matches reality, and adjust this parameter if it doesn't.
    frame_stack_size
        How many frames should be stacked together and put into a shared memory slot?
        If this is chosen too small, it might cause slowdowns because of having
        to handle many small objects; if it is chosen too large, it again may be
        slower due to having to split frame stacks more often in boundary conditions.
        When in doubt, leave this value at it's default.
    huge_pages
        Set to True to allocate shared memory in huge pages. This can improve performance
        by reducing the page fault cost. Currently only available on Linux. Enabling this
        requires reserving huge pages, either at system start, or before connecting.

        For example, to reserve 10000 huge pages, you can run:

        :code:`echo 10000 | sudo tee /proc/sys/vm/nr_hugepages`

        See also the :code:`hugeadm` utility, especially :code:`hugeadm --explain`
        can be useful to check your configuration.
    '''
    def __init__(
        self,
        api_host: str,
        api_port: int,
        data_host: str,
        data_port: int,
        buffer_size: int = 1024,
        bytes_per_frame: Optional[int] = None,
        frame_stack_size: int = 24,
        huge_pages: bool = False,
    ):
        self._passive_started = False
        self._api_host = api_host
        self._api_port = api_port
        self._data_host = data_host
        self._data_port = data_port
        self._huge_pages = huge_pages
        self._frame_stack_size = frame_stack_size

        if bytes_per_frame is None:
            # estimate based on detector size:
            ec = self.get_api_client()
            shape_x = ec.detectorConfig("x_pixels_in_detector")['value']
            shape_y = ec.detectorConfig("y_pixels_in_detector")['value']
            bit_depth = ec.detectorConfig("bit_depth_image")['value']

            bpp = bit_depth // 8
            bytes_per_frame_uncompressed = bpp * shape_x * shape_y

            # rough guess, doesn't have to be exact:
            bytes_per_frame = bytes_per_frame_uncompressed // 8

        assert bytes_per_frame is not None, "should be set automatically if None"
        self._bytes_per_frame = bytes_per_frame

        buffer_size_bytes = buffer_size * 1024 * 1024
        num_slots = int(math.floor(buffer_size_bytes / (bytes_per_frame * frame_stack_size)))
        self._num_slots = num_slots
        self._conn: libertem_dectris.DectrisConnection = self._connect()

    def _connect(self):
        return libertem_dectris.DectrisConnection(
            uri=f"tcp://{self._data_host}:{self._data_port}",
            frame_stack_size=self._frame_stack_size,
            num_slots=self._num_slots,
            bytes_per_frame=self._bytes_per_frame,
            huge=self._huge_pages,
            handle_path=self._make_socket_path(),
        )

    def wait_for_acquisition(
        self, timeout: Optional[float] = None
    ) -> Optional[DectrisPendingAcquisition]:
        if not self._passive_started:
            self._conn.start_passive()
            self._passive_started = True
        self._ensure_basic_settings()
        config_series = self._conn.wait_for_arm(timeout)
        if config_series is None:
            return None
        config, series = config_series
        return DectrisPendingAcquisition(
            detector_config=config,
            series=series,
        )

    def get_active_controller(
        self,
        trigger_mode: Optional[TriggerMode] = None,
        count_time: Optional[float] = None,
        frame_time: Optional[float] = None,
        roi_mode: Optional[str] = None,  # disabled, merge2x2 etc.
        roi_y_size: Optional[int] = None,
        roi_bit_depth: Optional[int] = None,
        enable_file_writing: Optional[bool] = None,
        compression: Optional[str] = None,  # bslz4, lz4
        name_pattern: Optional[str] = None,
        nimages_per_file: Optional[int] = 0,
    ):
        '''
        Create a controller object that knows about the detector settings
        to apply when the acquisition starts.

        Any settings left out or set to None will be left unchanged.

        Parameters
        ----------
        trigger_mode
            The strings 'exte', 'inte', 'exts', 'ints', as defined in the manual
        count_time
            Exposure time per image in seconds
        frame_time
            The interval between start of image acquisitions in seconds
        roi_mode
            Configure ROI mode. Set to the string 'disabled' to disable ROI mode.
            The allowed values depend on the detector.

            For example, for ARINA, to bin to frames of 96x96,
            set `roi_mode` to 'merge2x2'.

            For QUADRO, to select a subset of lines as active, set `roi_mode` to
            'lines'. Then, additionally set `roi_y_size` to one of the supported values.
        roi_bit_depth
            For QUADRO, this can be either 8 or 16. Setting to 8 bit is required
            to reach the highest frame rates.
        roi_y_size
            Select a subset of lines. For QUADRO, this has to be 64, 128, or 256.
            Note that the image size is then two times this value, plus two pixels,
            for example if you select 64 lines, it will result in images with 130
            pixels height and the full width.
        name_pattern
            If given, file writing is enabled and the name pattern is set to the
            given string. Please see the DECTRIS documentation for details!
        '''
        return DectrisActiveController(
            # these two don't need to be repeated:
            api_host=self._api_host,
            api_port=self._api_port,

            trigger_mode=trigger_mode,
            count_time=count_time,
            frame_time=frame_time,
            roi_mode=roi_mode,
            roi_y_size=roi_y_size,
            roi_bit_depth=roi_bit_depth,
            enable_file_writing=enable_file_writing,
            compression=compression,
            name_pattern=name_pattern,
            nimages_per_file=nimages_per_file,
        )

    @property
    def bytes_per_frame(self) -> int:
        return self._bytes_per_frame

    def get_api_client(self):
        ec = DEigerClient(self._api_host, port=self._api_port)
        return ec

    def _ensure_basic_settings(self):
        ec = self.get_api_client()
        ec.setStreamConfig('mode', 'enabled')
        ec.setStreamConfig('header_detail', 'basic')

    def start_series(self, series: int):
        if self._passive_started:
            raise RuntimeError(
                f"Cannot start acquisition for series {series}, "
                "already in passive mode"
            )
        self._ensure_basic_settings()
        self._conn.start(series)

    def get_conn_impl(self):
        return self._conn

    @classmethod
    def _make_socket_path(cls):
        temp_path = tempfile.mkdtemp()
        return os.path.join(temp_path, 'dectris-shm-socket')

    def stop_series(self):
        pass  # TODO: what to do?

    def close(self):
        self._conn.close()
        self._conn = None

    def reconnect(self):
        if self._conn is not None:
            self.close()
        self._conn = self._connect()

    def log_stats(self):
        self._conn.log_shm_stats()


class DectrisAcquisition(AcquisitionMixin, DataSet):
    '''
    Acquisition from a DECTRIS detector

    Parameters
    ----------
    conn
        An existing `DectrisDetectorConnection` instance
    nav_shape
        The number of scan positions as a 2-tuple :code:`(height, width)`
    trigger : function
        See :meth:`~libertem_live.api.LiveContext.prepare_acquisition`
        and :ref:`trigger` for details!
    frames_per_partition
        A tunable for configuring the feedback rate - more frames per partition
        means slower feedback, but less computational overhead. Might need to be tuned
        to adapt to the dwell time.
    enable_corrections
        Automatically correct defect pixels, downloading the pixel mask from the
        detector configuration.
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
    '''
    def __init__(
        self,

        # this replaces the {api,data}_{host,port} parameters:
        conn: DectrisDetectorConnection,

        nav_shape: Tuple[int, ...],
        trigger=lambda aq: None,
        frames_per_partition: int = 128,
        enable_corrections: bool = False,

        # in passive mode, we get this:
        pending_aq: Optional[DectrisPendingAcquisition] = None,

        # in passive mode, we don't pass the controller.
        # (in active mode, you _can_ pass it, but if you don't, a default
        # controller will be created):
        controller: Optional[DectrisActiveController] = None,
    ):
        super().__init__(trigger=trigger)
        self._nav_shape = nav_shape
        self._sig_shape: Tuple[int, ...] = ()
        self._acq_state: Optional[AcquisitionParams] = None
        self._frames_per_partition = min(frames_per_partition, prod(nav_shape))
        self._enable_corrections = enable_corrections

        if controller is None and pending_aq is None:
            # default to active mode, with no settings to change:
            controller = conn.get_active_controller()

        self._conn = conn
        self._controller = controller

        if pending_aq is not None:
            if controller is not None:
                raise ValueError(
                    "Got both controller and pending acquisition, please only provide one"
                )
            self._detector_config = pending_aq.detector_config
            self._series = pending_aq.series
        else:
            self._detector_config = None
            self._series = None

    def get_api_client(self):
        return self._conn.get_api_client()

    def get_detector_config(self) -> DetectorConfig:
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

    def get_correction_data(self):
        if not self._enable_corrections:
            return CorrectionSet()
        ec = self._conn.get_api_client()
        mask = ec.detectorConfig("pixel_mask")
        mask_arr = get_darray(mask)
        excluded_pixels = mask_arr > 0
        return CorrectionSet(excluded_pixels=excluded_pixels)

    @contextmanager
    def acquire(self):
        with tracer.start_as_current_span('acquire'):
            if self._controller is not None:
                self._controller.apply_file_writing()
                self._controller.apply_scan_settings(self._nav_shape)
                self._controller.apply_misc_settings()
                sequence_id = self._controller.arm()
                if self._series is None:
                    self._series = sequence_id
                nimages = prod(self.shape.nav)
                self._acq_state = AcquisitionParams(
                    sequence_id=sequence_id,
                    nimages=nimages,
                )
                with tracer.start_as_current_span("DectrisAcquisition.trigger"):
                    self.trigger()
            else:
                # in passive mode, we should have the series available:
                assert self._series is not None
                nimages = prod(self.shape.nav)
                self._acq_state = AcquisitionParams(
                    sequence_id=self._series,
                    nimages=nimages,
                )

            yield

    def check_valid(self):
        pass

    def need_decode(self, read_dtype, roi, corrections):
        return True  # FIXME: we just do this to get a large tile size

    def adjust_tileshape(self, tileshape, roi):
        depth = 12  # FIXME: hardcoded, hmm...
        return (depth, *self.meta.shape.sig)
        # return Shape((self._end_idx - self._start_idx, 256, 256), sig_dims=2)

    def get_max_io_size(self):
        # return 12*256*256*8
        # FIXME magic numbers?
        return 12*np.prod(self.meta.shape.sig)*8

    def get_base_shape(self, roi):
        return (1, *self.meta.shape.sig)

    @property
    def acquisition_state(self):
        return self._acq_state

    def get_partitions(self):
        # FIXME: only works for inline executor or similar, as we are using a zeromq socket
        # which is not safe to be passed to other threads
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
        assert array_backend in (None, NUMPY, CUDA)
        assert len(tiling_scheme) == 1, "only supports full frames tiling scheme for now"
        logger.debug("reading up to frame idx %d for this partition", self._end_idx)
        to_read = self._end_idx - self._start_idx
        depth = tiling_scheme.depth
        # over-allocate buffer by a bit so we can handle larger incoming raw tiles
        buf = np.zeros((2 * depth,) + tuple(tiling_scheme[0].shape), dtype=dest_dtype)
        tile_start = self._start_idx
        frames = get_frames(self._worker_context.get_worker_queue())
        frames_in_tile = 0
        while to_read > 0:
            # 1) put frame into tile buffer (including dtype conversion if needed)
            try:
                raw_tile = next(frames)
                frames_in_tile = raw_tile.shape[0]
                if frames_in_tile > buf.shape[0]:
                    buf = np.zeros(
                        (frames_in_tile,) + tuple(tiling_scheme[0].shape),
                        dtype=dest_dtype
                    )
                # FIXME: make copy optional if dtype already matches
                buf[:frames_in_tile] = raw_tile
                to_read -= frames_in_tile
            except StopIteration:
                assert to_read == 0, f"we were still expecting to read {to_read} frames more!"

            tile_buf = buf[:frames_in_tile]
            if tile_buf.shape[0] == 0:
                assert to_read == 0
                continue  # we are done and the buffer is empty

            tile_shape = Shape(
                (frames_in_tile,) + tuple(tiling_scheme[0].shape),
                sig_dims=2
            )
            tile_slice = Slice(
                origin=(tile_start,) + (0, 0),
                shape=tile_shape,
            )
            # print(f"yielding tile for {tile_slice}")
            self._preprocess(tile_buf, tile_slice)
            yield DataTile(
                tile_buf,
                tile_slice=tile_slice,
                scheme_idx=0,
            )
            tile_start += frames_in_tile
        logger.debug("LivePartition.get_tiles: end of method")

    def get_tiles(self, tiling_scheme, dest_dtype="float32", roi=None,
            array_backend: Optional["ArrayBackend"] = None):
        yield from self._get_tiles_fullframe(
            tiling_scheme, dest_dtype, roi,
            array_backend=array_backend
        )

    def __repr__(self):
        return f"<DectrisLivePartition {self._start_idx}:{self._end_idx}>"
