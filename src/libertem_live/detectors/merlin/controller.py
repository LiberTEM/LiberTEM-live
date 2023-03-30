from typing import Tuple
from libertem_live.detectors.base.controller import AcquisitionController


class MerlinActiveController(AcquisitionController):
    def determine_nav_shape(self, nimages: int) -> Tuple[int, ...]:
        raise NotImplementedError()
