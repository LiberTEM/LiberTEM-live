import os
import math
import tempfile
from typing import Optional, TYPE_CHECKING, Type

from libertem_live.detectors.base.connection import (
    DetectorConnection,
)
import libertem_dectris
from .common import DectrisPendingAcquisition, TriggerMode
from .controller import DectrisActiveController
from .DEigerClient import DEigerClient


if TYPE_CHECKING:
    from .acquisition import DectrisAcquisition


class DectrisDetectorConnection(DetectorConnection):
    '''
    Connection to a DECTRIS DCU, both for detector configuration and for accessing
    the data stream.

    Please see :class:`libertem_live.detectors.dectris.DectrisConnectionBuilder`
    for a description of the parameters, and use
    :meth:`libertem_live.api.LiveContext.make_connection` to create a connection.


    Examples
    --------
    >>> from libertem_live.api import LiveContext
    >>> with LiveContext() as ctx, ctx.make_connection('dectris').open(
    ...     api_host='127.0.0.1',
    ...     api_port=DCU_API_PORT,
    ...     data_host='127.0.0.1',
    ...     data_port=DCU_DATA_PORT,
    ... ) as conn:
    ...     print("connected!")
    connected!
    '''
    def __init__(
        self,
        api_host: str,
        api_port: int,
        data_host: str,
        data_port: int,
        buffer_size: int = 2048,
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

    def __enter__(self):
        # in most cases, we are already connected, but for
        # re-using this object in multiple `with`-statements,
        # we need to handle the re-connection case here:
        if self._conn is None:
            self._conn = self._connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def wait_for_acquisition(
        self, timeout: Optional[float] = None
    ) -> Optional[DectrisPendingAcquisition]:
        """
        Wait for at most `timeout` seconds for an acquisition to start. This
        does not perform any triggering itself and expects something external
        to arm and trigger the acquisition.

        Once the detector is armed, this function returns a `PendingAcquisition`,
        which can be converted to a full `Acquisition` object using
        :meth:`libertem_live.api.LiveContext.make_acquisition`.

        The function returns `None` on timeout.

        Parameters
        ----------
        timeout
            Timeout in seconds. If `None`, wait indefinitely.

        Examples
        --------
        >>> with ctx.make_connection('dectris').open(
        ...     api_host='127.0.0.1',
        ...     api_port=DCU_API_PORT,
        ...     data_host='127.0.0.1',
        ...     data_port=DCU_DATA_PORT,
        ... ) as conn:
        ...     # NOTE: this is the part that is usually done by an external software,
        ...     # but we include it here to have a running example:
        ...     ec = conn.get_api_client()
        ...     arm_response = ec.sendDetectorCommand('arm')
        ...
        ...     pending_aq = conn.wait_for_acquisition(timeout=1)
        ...     assert pending_aq is not None, "timeout"
        ...
        ...     aq = ctx.make_acquisition(
        ...         conn=conn,
        ...         nav_shape=(32, 32),
        ...         pending_aq=pending_aq,
        ...     )
        ...     ctx.run_udf(dataset=aq, udf=SumUDF())
        {'intensity': ...}
        """
        # FIXME: it would be better to ask `self._conn` if it is currently
        # in the passive mode, otherwise we have to carefully track the
        # current state from the outside, too.
        if not self._passive_started:
            self._conn.start_passive()
            self._passive_started = True
        self._ensure_basic_settings()
        if timeout is None:
            timeout = 1.0
            while True:
                config_series = self._conn.wait_for_arm(timeout)
                if config_series is not None:
                    break
        else:
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
        enable_corrections: bool = False,
        mask_to_zero: Optional[bool] = None,
    ):
        '''
        Create a controller object that knows about the detector settings
        to apply when the acquisition starts.

        Any settings left out or set to None will be left unchanged.

        In general, please consult your detector manual for details!

        Parameters
        ----------
        trigger_mode
            One of 'exte', 'inte', 'exts', 'ints', as defined in the manual.
        count_time
            Exposure time per image in seconds
        frame_time
            The interval between start of image acquisitions in seconds. Must be
            greater than or equal to `count_time`.
        roi_mode
            Configure ROI mode. Set to the string 'disabled' to disable ROI mode.
            The allowed values depend on the detector.

            For example, for ARINA, to bin to frames of 96x96,
            set `roi_mode` to 'merge2x2'.

            For QUADRO and other supported detectors, to select a subset of
            lines as active, set `roi_mode` to 'lines'. Then, additionally set
            `roi_y_size` to one of the supported values.
        roi_bit_depth
            For QUADRO, this can be either 8 or 16. Setting to 8 bit is required
            to reach the highest frame rates.
        roi_y_size
            Select a subset of lines. For QUADRO, this has to be between 2 and
            256, inclusive. Note that the image size is then two times this
            value, plus two pixels, for example if you select 64 lines, it will
            result in images with 130 pixels height and the full width.

            Starting with DECTRIS software version "release-2022.1", this is the
            total height instead, so any even number between 4 and 512,
            inclusive.
        enable_file_writing
            Enable file writing on the DCU. If set to `False`, explicitly
            disable file writing.
        compression
            The type of compression to enable. Can be, for example,
            'bslz4', or 'lz4'.
        name_pattern
            If given, the name pattern is set to the given string. Only set if
            file writing is enabled.
        nimages_per_file
            If file writing is enabled, split the written files such that at
            maximum this number of images are saved per file. The default is not
            to split into multiple files.
        enable_corrections
            Let LiberTEM automatically correct defect pixels, downloading the
            pixel mask from the detector configuration.
        mask_to_zero
            When set to `True`, the bad pixels that are configured in the
            detector software are set to zero instead of MAX_INT. This is
            using the detector-internal correction facilities, not those
            of LiberTEM.

            Needs a current version of the DECTRIS software to work correctly
            (at least "release-2022.1.2").
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
            enable_corrections=enable_corrections,
            mask_to_zero=mask_to_zero,
        )

    @property
    def bytes_per_frame(self) -> int:
        return self._bytes_per_frame

    def get_api_client(self) -> DEigerClient:
        ec = DEigerClient(self._api_host, port=self._api_port)
        return ec

    def _ensure_basic_settings(self):
        ec = self.get_api_client()
        ec.setStreamConfig('mode', 'enabled')
        ec.setStreamConfig('header_detail', 'basic')

    def prepare_for_active(self):
        if self._passive_started:
            # get the background thread into the correct state:
            self.reconnect()

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
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        self._passive_started = False

    def reconnect(self):
        if self._conn is not None:
            self.close()
        self._conn = self._connect()

    def log_stats(self):
        self._conn.log_shm_stats()

    def get_acquisition_cls(self) -> Type["DectrisAcquisition"]:
        from .acquisition import DectrisAcquisition
        return DectrisAcquisition


class DectrisConnectionBuilder:
    """
    Builder class that can construct :class:`DectrisDetectorConnection` instances.

    Use the :meth:`open` method to create a connection.
    """
    def open(
        self,
        api_host: str,
        api_port: int,
        data_host: str,
        data_port: int,
        buffer_size: int = 2048,
        bytes_per_frame: Optional[int] = None,
        frame_stack_size: int = 24,
        huge_pages: bool = False,
    ) -> DectrisDetectorConnection:
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
        return DectrisDetectorConnection(
            api_host=api_host,
            api_port=api_port,
            data_host=data_host,
            data_port=data_port,
            buffer_size=buffer_size,
            bytes_per_frame=bytes_per_frame,
            frame_stack_size=frame_stack_size,
            huge_pages=huge_pages,
        )
