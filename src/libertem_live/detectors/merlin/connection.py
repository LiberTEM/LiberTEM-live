import os
import logging
from typing import Optional
from collections.abc import Generator
from contextlib import contextmanager

from libertem_live.detectors.base.connection import DetectorConnection, PendingAcquisition
from libertem_live.detectors.base.acquisition import AcquisitionProtocol

from .control import MerlinControl

from opentelemetry import trace
import libertem_qd_mpx

tracer = trace.get_tracer(__name__)
logger = logging.getLogger(__name__)


class MerlinPendingAcquisition(PendingAcquisition):
    def __init__(self, header):
        self._header = header

    @property
    def header(self):
        return self._header

    @property
    def nimages(self) -> int:
        return self.header.frames_in_acquisition()


class MerlinDetectorConnection(DetectorConnection):
    """
    This class holds a permanent data connection to the merlin software.

    Control connections are also possible to obtain from this class,
    but are created on demand and not kept open.

    You can use the convenience function
    :meth:`libertem_live.api.LiveContext.make_connection` to create an instance,
    instead of calling this constructor directly.

    Parameters
    ----------
    api_host
        Hostname of the Merlin control server, default '127.0.0.1'
        Should in most cases be the same as `data_host`.
    api_port
        Port of the Merlin control server, default 6341
    data_host
        Hostname of the Merlin data server, default '127.0.0.1'
    data_port
        Data port of the Merlin data server, default 6342
    drain
        Drain the socket before triggering. Enable this when
        using old versions of the Merlin software, but not when
        using an internal trigger.

    Examples
    --------
    Usually, this class is instantiated using
    :meth:`libertem_live.api.LiveContext.make_connection`:

    >>> with ctx.make_connection('merlin').open(
    ...     api_host='127.0.0.1',
    ...     api_port=MERLIN_API_PORT,
    ...     data_host='127.0.0.1',
    ...     data_port=MERLIN_DATA_PORT,
    ... ) as conn:
    ...     aq = ctx.make_acquisition(conn=conn, nav_shape=(32, 32))
    ...     ctx.run_udf(dataset=aq, udf=SumUDF())
    {'intensity': ...}
    """

    def __init__(
        self,
        *,
        api_host: str = '127.0.0.1',
        api_port: int = 6341,
        data_host: str = '127.0.0.1',
        data_port: int = 6342,
        drain: bool = False,
    ):
        self._api_host = api_host
        self._api_port = api_port
        self._data_host = data_host
        self._data_port = data_port
        self._data_socket: Optional[libertem_qd_mpx.QdConnection] = None
        self._drain = drain
        self.connect()

    def connect(self):
        if self._data_socket is not None:
            return self._data_socket  # already connected
        self._data_socket = libertem_qd_mpx.QdConnection(
            data_host=self._data_host,
            data_port=self._data_port,
            frame_stack_size=16,  # FIXME! make configurable or determine automatically
            shm_handle_path=self._make_socket_path(),
            drain=self._drain,
        )
        self._data_socket.start_passive()
        return self._data_socket

    @classmethod
    def _make_socket_path(cls):
        import tempfile
        temp_path = tempfile.mkdtemp()
        return os.path.join(temp_path, 'qd-shm-socket')

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

        Examples
        --------
        >>> with ctx.make_connection('merlin').open(
        ...     api_host='127.0.0.1',
        ...     api_port=MERLIN_API_PORT,
        ...     data_host='127.0.0.1',
        ...     data_port=MERLIN_DATA_PORT,
        ... ) as conn:
        ...     pending_aq = conn.wait_for_acquisition(timeout=1)
        ...     # at this point, something else is arming and triggering the
        ...     # detector:
        ...     aq = ctx.make_acquisition(
        ...         conn=conn,
        ...         nav_shape=(32, 32),
        ...         pending_aq=pending_aq,
        ...     )
        ...     ctx.run_udf(dataset=aq, udf=SumUDF())
        {'intensity': ...}
        """
        if self._data_socket is None:
            self.connect()
        assert self._data_socket is not None
        acq_header = self._data_socket.wait_for_arm(timeout=timeout)
        if acq_header is None:
            return None
        return MerlinPendingAcquisition(header=acq_header)

    def get_acquisition_cls(self) -> type[AcquisitionProtocol]:
        from .acquisition import MerlinAcquisition
        return MerlinAcquisition

    def __enter__(self):
        if self._data_socket is None:
            self.connect()
        return self

    def close(self):
        # implementing "close" and __enter__ above gives us contextmanager API
        # for free:
        if self._data_socket is not None:
            self._data_socket.close()
            self._data_socket = None

    def get_conn_impl(self):
        return self._data_socket  # maybe remove `get_data_socket` function?

    def get_data_socket(self) -> libertem_qd_mpx.QdConnection:
        assert self._data_socket is not None, "need to be connected to call this"
        return self._data_socket

    @contextmanager
    def control(self) -> Generator[MerlinControl, None, None]:
        with MerlinControl(host=self._api_host, port=self._api_port) as c:
            yield c

    def read_sig_shape(self) -> tuple[int, int]:
        with self.control() as c:
            width = int(c.get('IMAGEX'))
            height = int(c.get('IMAGEY'))
            return (height, width)

    def read_bitdepth(self) -> int:
        with self.control() as c:
            return int(c.get('COUNTERDEPTH'))

    def get_active_controller(self):
        from .controller import MerlinActiveController
        return MerlinActiveController()


class MerlinConnectionBuilder:
    """
    Builder class that can construct :class:`MerlinDetectorConnection` instances.

    Use the :meth:`open` method to create a connection.
    """
    def open(
        self,
        *,
        api_host: str = '127.0.0.1',
        api_port: int = 6341,
        data_host: str = '127.0.0.1',
        data_port: int = 6342,
        drain: bool = False,
    ) -> MerlinDetectorConnection:
        """
        Connect to a Merlin Medipix detector system.

        Parameters
        ----------
        api_host
            Hostname of the Merlin control server, default '127.0.0.1'
            Should in most cases be the same as `data_host`.
        api_port
            Port of the Merlin control server, default 6341
        data_host
            Hostname of the Merlin data server, default '127.0.0.1'
        data_port
            Data port of the Merlin data server, default 6342
        drain
            Drain the socket before triggering. Enable this when
            using old versions of the Merlin software, but not when
            using an internal trigger.

        Examples
        --------
        Usually, this method is directly used together with
        :meth:`libertem_live.api.LiveContext.make_connection`:

        >>> with ctx.make_connection('merlin').open(
        ...     api_host='127.0.0.1',
        ...     api_port=MERLIN_API_PORT,
        ...     data_host='127.0.0.1',
        ...     data_port=MERLIN_DATA_PORT,
        ... ) as conn:
        ...     aq = ctx.make_acquisition(conn=conn, nav_shape=(32, 32))
        ...     ctx.run_udf(dataset=aq, udf=SumUDF())
        {'intensity': ...}
        """
        return MerlinDetectorConnection(
            api_host=api_host,
            api_port=api_port,
            data_host=data_host,
            data_port=data_port,
            drain=drain,
        )
