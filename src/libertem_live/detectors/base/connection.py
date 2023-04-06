from typing import Optional, Type

from .acquisition import AcquisitionProtocol


class PendingAcquisition:
    """
    A token object that can be obtained from
    :meth:`libertem_live.detectors.base.connection.DetectorConnection.wait_for_acquisition`.

    Currently doesn't carry any user-accessible information.
    Pass this object into :meth:`libertem_live.api.LiveContext.make_acquisition`
    to start processing the incoming data stream.
    """
    pass


class DetectorConnection:
    """
    Base class for detector connections.
    """
    def wait_for_acquisition(self, timeout: Optional[float] = None) -> Optional[PendingAcquisition]:
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
        """
        raise NotImplementedError()

    def close(self):
        """
        Close the connection. It's important to call this function once
        you don't need the connection anymore, as an open connection
        might interfere with other software using the detector.

        If possible, use this object as a context manager instead,
        using a :code:`with`-statement.
        """
        raise NotImplementedError()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def get_acquisition_cls(self) -> Type[AcquisitionProtocol]:
        """
        Returns the matching `Acquisition` class.

        :meta private:
        """
        raise NotImplementedError()
