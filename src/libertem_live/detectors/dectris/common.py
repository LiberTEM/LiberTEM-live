from typing import NamedTuple, Union, TYPE_CHECKING
from typing_extensions import Literal
from libertem_live.detectors.base.connection import (
    PendingAcquisition,
)

if TYPE_CHECKING:
    import libertem_dectris


TriggerMode = Union[
    Literal['exte'],  # acquire one image for each trigger, `ntrigger` times
    Literal['exts'],  # acquire series of `nimages` with a single trigger
    Literal['ints'],  # internal software triggering
    Literal['inte'],  # internal software enable -> one image for each soft-trigger
]


class AcquisitionParams(NamedTuple):
    sequence_id: int
    nimages: int


class DetectorConfig(NamedTuple):
    x_pixels_in_detector: int
    y_pixels_in_detector: int
    bit_depth: int


class DectrisPendingAcquisition(PendingAcquisition):
    def __init__(self, detector_config: "libertem_dectris.DetectorConfig", series: int):
        self._detector_config = detector_config
        self._series = series

    @property
    def detector_config(self):
        return self._detector_config

    @property
    def series(self):
        return self._series

    @property
    def nimages(self) -> int:
        return self.detector_config.get_num_frames()

    def __repr__(self):
        return f"<DectrisPendingAcquisition series={self.series} config={self.detector_config}>"
