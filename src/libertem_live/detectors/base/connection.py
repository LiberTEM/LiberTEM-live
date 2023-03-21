from typing import Optional


class PendingAcquisition:
    def get_acquisition(self, *args, trigger=None, **kwargs):
        raise NotImplementedError()


class DetectorConnection:
    def wait_for_acquisition(self, timeout: Optional[float] = None) -> Optional[PendingAcquisition]:
        """
        Wait for at most `timeout` seconds for an acquisition to start. This
        does not perform any triggering itself and expects something external
        to arm and trigger the acquisition.

        Once the detector is armed, this function returns a `PendingAcquisition`,
        which can be converted to a full `Acquisition` object.

        The function returns `None` on timeout.

        Parameters
        ----------
        timeout : Optional[float]
            Timeout in seconds. If `None`, wait indefinitely.
        """
        raise NotImplementedError()
