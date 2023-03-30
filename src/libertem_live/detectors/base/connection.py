from typing import Optional, Type

from .acquisition import AcquisitionMixin


class PendingAcquisition:
    """
    Token object that is obtained from
    `DetectorConnection.wait_for_acquisition`.

    Currently doesn't carry any user-accessible information.
    """
    pass


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

    def close(self):
        raise NotImplementedError()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def get_acquisition_cls(self) -> Type[AcquisitionMixin]:
        """
        Returns the matching `Acquisition` class
        """
        raise NotImplementedError()
