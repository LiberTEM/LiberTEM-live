import os
from typing import Optional, Type, Tuple
import tempfile
import logging

from libertem_live.detectors.base.connection import (
    PendingAcquisition, DetectorConnection,
    AcquisitionProtocol,
)

import libertem_asi_mpx3
from libertem_asi_mpx3 import ServalConnection, ServalAPIClient

logger = logging.getLogger(__name__)


class AsiMpx3PendingAcquisition(PendingAcquisition):
    def __init__(self, pending_config):
        self._pending_config = pending_config

    @property
    def detector_config(self):
        return self._pending_config.get_detector_config()

    @property
    def sig_shape(self) -> Tuple[int, int]:
        return (
            self._pending_config.get_frame_height(),
            self._pending_config.get_frame_width(),
        )

    @property
    def nimages(self) -> int:
        return self.detector_config.get_n_triggers()

    def __repr__(self):
        return f"<AsiPendingAcquisition header={self._pending_config}>"


class AsiMpx3DetectorConnection(DetectorConnection):
    """
    Connection to the ASI MPX3 software.

    Please see :class:`libertem_live.detectors.asi_mpx3.AsiMpx3ConnectionBuilder`
    for a description of the parameters, and use
    :meth:`libertem_live.api.LiveContext.make_connection` to create a connection.
    """
    def __init__(
        self,
        data_host: str,
        data_port: int,
        api_host: str,
        api_port: int,
        buffer_size: int = 2048,
        frame_stack_size: int = 24,
        huge_pages: bool = False,
    ):
        self._passive_started = False
        self._data_host = data_host
        self._data_port = data_port
        self._api_host = api_host
        self._api_port = api_port

        # assumption: most frames aren't using the full 32bit
        # (the bytes per pixel value is dynamic per frame, at least
        # for now)
        bpp = 2
        pix_count = self._get_detector_info().get_pix_count()
        bytes_per_frame = bpp * pix_count

        # approx:
        slot_size = bytes_per_frame * frame_stack_size
        self._num_slots = 1024 * 1024 * buffer_size // slot_size

        self._bytes_per_frame = bytes_per_frame
        self._huge_pages = huge_pages
        self._frame_stack_size = frame_stack_size

        self._conn = self._connect()

    @property
    def api_uri(self):
        return f"http://{self._api_host}:{self._api_port}"

    @property
    def api_client(self):
        return ServalAPIClient(self.api_uri)

    def _get_detector_info(self) -> "libertem_asi_mpx3.DetectorInfo":
        return self.api_client.get_detector_info()

    def _connect(self) -> ServalConnection:
        handle_path = self._make_handle_path()
        data_uri = f"{self._data_host}:{self._data_port}"
        logger.info(f"connecting to {data_uri} with shared memory handle {handle_path}")
        return ServalConnection(
            data_uri=data_uri,
            api_uri=self.api_uri,
            frame_stack_size=self._frame_stack_size,
            num_slots=self._num_slots,
            bytes_per_frame=self._bytes_per_frame,
            huge=self._huge_pages,
            handle_path=handle_path,
        )

    def wait_for_acquisition(
        self,
        timeout: Optional[float] = None,
    ) -> Optional[PendingAcquisition]:
        if not self._passive_started:
            self._conn.start_passive()
            self._passive_started = True
        detector_config = self._conn.wait_for_arm(timeout)
        if detector_config is None:
            return None
        return AsiMpx3PendingAcquisition(
            pending_config=detector_config,
        )

    def get_conn_impl(self):
        return self._conn

    @classmethod
    def _make_handle_path(cls):
        temp_path = tempfile.mkdtemp()
        return os.path.join(temp_path, 'asi-shm-socket')

    def stop_series(self):
        pass  # TODO: what to do?

    def close(self):
        if self._conn is not None:
            logger.info(f"closing connection to {self._data_host}:{self._data_port}")
            self._conn.close()
            self._passive_started = False
            self._conn = None

    def __enter__(self):
        if self._conn is None:
            self._conn = self._connect()
        return self

    def reconnect(self):
        if self._conn is not None:
            self.close()
        self._conn = self._connect()

    def log_stats(self):
        self._conn.log_shm_stats()

    def get_acquisition_cls(self) -> Type[AcquisitionProtocol]:
        from .acquisition import AsiMpx3Acquisition
        return AsiMpx3Acquisition


class AsiMpx3ConnectionBuilder:
    """
    Builder class that can construct :class:`AsiMpx3DetectorConnection` instances.

    Use the :meth:`open` method to create a connection.
    """
    def open(
        self,
        data_host: str = "127.0.0.1",
        data_port: int = 8283,
        api_host: str = "127.0.0.1",
        api_port: int = 8080,
        buffer_size: int = 2048,
        frame_stack_size: int = 24,
        huge_pages: bool = False,
    ) -> AsiMpx3DetectorConnection:
        """
        Connect to the ASI MPX3 detector software (Accos).

        Parameters
        ----------
        data_host
            The hostname or IP address of the computer running Accos
        data_port
            The TCP port to connect to
        buffer_size
            The total receive buffer in MiB that is used to stream data to worker
            processes.
        frame_stack_size
            How many frames should we assemble to a chunk stack?
        huge_pages
            Set to True to allocate shared memory in huge pages. This can improve performance
            by reducing the page fault cost. Currently only available on Linux. Enabling this
            requires reserving huge pages, either at system start, or before connecting.

            For example, to reserve 10000 huge pages, you can run:

            :code:`echo 10000 | sudo tee /proc/sys/vm/nr_hugepages`

            See also the :code:`hugeadm` utility, especially :code:`hugeadm --explain`
            can be useful to check your configuration.
        """
        return AsiMpx3DetectorConnection(
            data_host=data_host,
            data_port=data_port,
            api_host=api_host,
            api_port=api_port,
            buffer_size=buffer_size,
            frame_stack_size=frame_stack_size,
            huge_pages=huge_pages,
        )
