from libertem_live.detectors.base.controller import AcquisitionController


class MerlinActiveController(AcquisitionController):
    def determine_nav_shape(self, nimages: int) -> tuple[int, ...]:
        raise NotImplementedError()
