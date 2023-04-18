import os
from typing import Optional, Type, Tuple
import tempfile
import logging

from libertem.common.math import prod

from libertem_live.detectors.base.connection import (
    PendingAcquisition, DetectorConnection,
    AcquisitionProtocol,
)

from libertem_asi_tpx3 import ASITpx3Connection

logger = logging.getLogger(__name__)


class AsiTpx3PendingAcquisition(PendingAcquisition):
    def __init__(self, acquisition_header):
        self._acquisition_header = acquisition_header

    @property
    def header(self):
        return self._acquisition_header

    @property
    def nimages(self) -> int:
        return prod(self.nav_shape)

    @property
    def nav_shape(self) -> Tuple[int, ...]:
        return self._acquisition_header.get_nav_shape()

    def __repr__(self):
        return f"<AsiPendingAcquisition header={self._acquisition_header}>"


class AsiTpx3DetectorConnection(DetectorConnection):
    """
    Connection to the ASI TPX3 software.

    Please see :class:`libertem_live.detectors.asi_tpx3.AsiTpx3ConnectionBuilder`
    for a description of the parameters, and use
    :meth:`libertem_live.api.LiveContext.make_connection` to create a connection.
    """
    def __init__(
        self,
        data_host: str,
        data_port: int,
        buffer_size: int = 2048,
        bytes_per_chunk: Optional[int] = None,
        chunks_per_stack: int = 24,
        huge_pages: bool = False,
    ):
        self._passive_started = False
        self._data_host = data_host
        self._data_port = data_port

        if bytes_per_chunk is None:
            bytes_per_chunk = 16*1024

        # approx:
        slot_size = bytes_per_chunk * chunks_per_stack
        self._num_slots = 1024 * 1024 * buffer_size // slot_size

        self._bytes_per_chunk = bytes_per_chunk
        self._huge_pages = huge_pages
        self._chunks_per_stack = chunks_per_stack

        self._conn = self._connect()

    def _connect(self):
        handle_path = self._make_handle_path()
        uri = f"{self._data_host}:{self._data_port}"
        logger.info(f"connecting to {uri} with shared memory handle {handle_path}")
        return ASITpx3Connection(
            uri=uri,
            chunks_per_stack=self._chunks_per_stack,
            num_slots=self._num_slots,
            bytes_per_chunk=self._bytes_per_chunk,
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
        header = self._conn.wait_for_arm(timeout)
        if header is None:
            return None
        return AsiTpx3PendingAcquisition(
            acquisition_header=header,
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
        from .acquisition import AsiTpx3Acquisition
        return AsiTpx3Acquisition


class AsiTpx3ConnectionBuilder:
    """
    Builder class that can construct :class:`AsiTpx3DetectorConnection` instances.

    Use the :meth:`open` method to create a connection.
    """
    def open(
        self,
        data_host: str = "127.0.0.1",
        data_port: int = 8283,
        buffer_size: int = 2048,
        bytes_per_chunk: Optional[int] = None,
        chunks_per_stack: int = 24,
        huge_pages: bool = False,
    ) -> AsiTpx3DetectorConnection:
        """
        Connect to the ASI TPX3 detector software (Accos).

        Parameters
        ----------
        data_host
            The hostname or IP address of the computer running Accos
        data_port
            The TCP port to connect to
        buffer_size
            The total receive buffer in MiB that is used to stream data to worker
            processes.
        bytes_per_chunk
            How large is each chunk, in bytes. Approximate value, as
            this can change depending on events per scan position
        chunks_per_stack
            How many chunks should we assemble to a chunk stack?
        huge_pages
            Set to True to allocate shared memory in huge pages. This can improve performance
            by reducing the page fault cost. Currently only available on Linux. Enabling this
            requires reserving huge pages, either at system start, or before connecting.

            For example, to reserve 10000 huge pages, you can run:

            :code:`echo 10000 | sudo tee /proc/sys/vm/nr_hugepages`

            See also the :code:`hugeadm` utility, especially :code:`hugeadm --explain`
            can be useful to check your configuration.
        """
        return AsiTpx3DetectorConnection(
            data_host=data_host,
            data_port=data_port,
            buffer_size=buffer_size,
            bytes_per_chunk=bytes_per_chunk,
            chunks_per_stack=chunks_per_stack,
            huge_pages=huge_pages,
        )
