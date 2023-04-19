from typing import Optional, Tuple, TYPE_CHECKING
import logging

from libertem.common.math import prod
from .common import TriggerMode
from .DEigerClient import DEigerClient
from ..base.controller import AcquisitionController

if TYPE_CHECKING:
    from .acquisition import DectrisDetectorConnection

logger = logging.getLogger(__name__)


class DectrisActiveController(AcquisitionController):
    """
    Detector settings. If you leave them out or set them to None, they
    will not be changed.

    Users should call :meth:`DectrisDetectorConnection.get_active_controller`
    instead of manually constructing an instance of this class.
    """
    # NOTE: when adding a parameter here, also update
    # `DectrisDetectorConnection.get_active_controller`
    def __init__(
        self,
        api_host: str,
        api_port: int,
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
        self._api_host = api_host
        self._api_port = api_port
        self._trigger_mode = trigger_mode
        self._enable_file_writing = enable_file_writing
        self._name_pattern = name_pattern
        self._nimages_per_file = nimages_per_file
        self._count_time = count_time
        self._frame_time = frame_time
        self._compression = compression
        self._roi_mode = roi_mode
        self._roi_y_size = roi_y_size
        self._roi_bit_depth = roi_bit_depth
        self._enable_corrections = enable_corrections
        self._mask_to_zero = mask_to_zero

    def get_api_client(self) -> DEigerClient:
        ec = DEigerClient(self._api_host, port=self._api_port)
        return ec

    def apply_scan_settings(self, nav_shape: Tuple[int, ...]):
        ec = self.get_api_client()
        nimages = prod(nav_shape)
        if self._trigger_mode is not None:
            ec.setDetectorConfig('ntrigger', 1)
            ec.setDetectorConfig('nimages', 1)
            ec.setDetectorConfig('trigger_mode', self._trigger_mode)
        if self._trigger_mode in ('exte', 'exts'):
            ec.setDetectorConfig('ntrigger', nimages)
        elif self._trigger_mode in ('ints',):
            ec.setDetectorConfig('nimages', nimages)

    def apply_misc_settings(self):
        ec = self.get_api_client()

        # Set ROI settings first, as they will reset other settings:
        if self._roi_mode is not None:
            ec.setDetectorConfig('roi_mode', self._roi_mode)
        if self._roi_y_size is not None:
            ec.setDetectorConfig('roi_y_size', self._roi_y_size)
        if self._roi_bit_depth is not None:
            ec.setDetectorConfig('roi_bit_depth', self._roi_bit_depth)

        # `frame_time`` must be longer than `count_time`. From the manual:
        # "To acquire images with a certain frame rate and best possible duty
        # cycle, a simple procedure is to first set `count_time` to the inverse
        # of the frame rate and subsequently `frame_time` to the inverse of the
        # frame rate."
        if self._count_time is not None:
            ec.setDetectorConfig('count_time', self._count_time)
        if self._frame_time is not None:
            ec.setDetectorConfig('frame_time', self._frame_time)
        if self._compression is not None:
            ec.setDetectorConfig('compression', self._compression)

        # mask to zero instead of MAX_INT:
        if self._mask_to_zero is not None:
            ec.setDetectorConfig('mask_to_zero', self._mask_to_zero)

    def apply_file_writing(self):
        """
        Enable/disable file writing etc.
        """
        ec = self.get_api_client()
        if self._enable_file_writing:
            ec.setFileWriterConfig("mode", "enabled")
            if self._name_pattern is not None:
                ec.setFileWriterConfig("name_pattern", self._name_pattern)
            ec.setFileWriterConfig("nimages_per_file", self._nimages_per_file)
        elif self._enable_file_writing is False:
            ec.setFileWriterConfig("mode", "disabled")

    def arm(self) -> int:
        logger.info("arming detector")
        ec = self.get_api_client()
        result = ec.sendDetectorCommand('arm')
        logger.info("detector armed successfully")
        return result['sequence id']

    def handle_start(self, conn: "DectrisDetectorConnection", series: int):
        """
        This is called from the `TaskCommHandler` before the first task
        """
        conn.start_series(series)

    def handle_stop(self, conn: "DectrisDetectorConnection"):
        """
        This is called from the `TaskCommHandler` after the last task result has
        arrived, or an error has occured.
        """
        # conn.stop_series()
        pass  # FIXME: do we have to do anything here?

    @property
    def enable_corrections(self) -> bool:
        return self._enable_corrections
