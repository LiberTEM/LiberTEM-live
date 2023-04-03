import logging
from typing import Optional, Type, Generator, Tuple
from contextlib import contextmanager

from libertem_live.detectors.base.connection import DetectorConnection, PendingAcquisition
from libertem_live.detectors.base.acquisition import AcquisitionMixin

from .control import MerlinControl
from .data import MerlinRawSocket

from opentelemetry import trace

tracer = trace.get_tracer(__name__)
logger = logging.getLogger(__name__)


class MerlinPendingAcquisition(PendingAcquisition):
    """
    This object carries the acquisition header and can provide the
    parsed values.
    """
    def __init__(self):
        pass


class MerlinDetectorConnection(DetectorConnection):
    """
    This class holds a permanent data connection to the merlin software.

    Control connections are also possible to obtain from this class,
    but are created on demand and not kept open.

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
        self._drain = drain
        self._connect()

    def _connect(self):
        self._data_socket = MerlinRawSocket(
            host=self._data_host,
            port=self._data_port,
        ).connect()
        return self._data_socket

    def wait_for_acquisition(self, timeout: Optional[float] = None) -> Optional[PendingAcquisition]:
        if self._data_socket is None:
            self._connect()
        # Hmm, with Merlin, we have a "situation":
        # - we don't have all the info we need in the acquisition header
        # - what is missing?
        #    - for `read_multi_frames`: `header_size_bytes` and `image_size_bytes`
        #    - same for `get_input_buffer`
        #    - -> we don't need these until we actually receive data!
        #    - `MerlinRawSocket` should maybe not include these methods,
        raise NotImplementedError()

    def get_acquisition_cls(self) -> Type[AcquisitionMixin]:
        from .acquisition import MerlinAcquisition
        return MerlinAcquisition

    def __enter__(self):
        if self._data_socket is None:
            self._connect()
        return self

    def close(self):
        # implementing "close" gives us contextmanager API for free.
        if self._data_socket is not None:
            self._data_socket.close()
            self._data_socket = None

    def maybe_drain(self, timeout: float = 1.0):
        assert self._data_socket is not None
        if self._drain:
            with tracer.start_as_current_span("drain") as span:
                drained_bytes = self._data_socket.drain()
                span.set_attributes({
                    "libertem_live.drained_bytes": drained_bytes,
                })
            if drained_bytes > 0:
                logger.info(f"drained {drained_bytes} bytes of garbage")

    def get_data_socket(self) -> MerlinRawSocket:
        assert self._data_socket is not None, "need to be connected to call this"
        return self._data_socket

    @contextmanager
    def control(self) -> Generator[MerlinControl, None, None]:
        with MerlinControl(host=self._api_host, port=self._api_port) as c:
            yield c

    def read_sig_shape(self) -> Tuple[int, int]:
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
    def open(
        self,
        *,
        api_host: str = '127.0.0.1',
        api_port: int = 6341,
        data_host: str = '127.0.0.1',
        data_port: int = 6342,
        drain: bool = False,
    ):
        """
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
        """
        return MerlinDetectorConnection(
            api_host=api_host,
            api_port=api_port,
            data_host=data_host,
            data_port=data_port,
            drain=drain,
        )
