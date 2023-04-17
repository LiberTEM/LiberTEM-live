import os
from typing import Optional, Type, Tuple
import tempfile

from libertem.common.math import prod

from libertem_live.detectors.base.connection import (
    PendingAcquisition, DetectorConnection,
    AcquisitionProtocol,
)

from libertem_asi_tpx3 import ASITpx3Connection


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
    def __init__(
        self,
        uri: str,
        num_slots: int,
        bytes_per_chunk: int,
        chunks_per_stack: int = 24,
        huge_pages: bool = False,
    ):
        self._passive_started = False
        self._uri = uri

        self._num_slots = num_slots
        self._bytes_per_chunk = bytes_per_chunk
        self._huge_pages = huge_pages
        self._chunks_per_stack = chunks_per_stack

        self._conn = self._connect()

    def _connect(self):
        handle_path = self._make_handle_path()
        return ASITpx3Connection(
            uri=self._uri,
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
            self._conn.close()
            self._conn = None

    def __enter__(self):
        if self._conn is None:
            self._connect()
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
        uri: str,
        num_slots: int,
        bytes_per_chunk: int,
        chunks_per_stack: int = 24,
        huge_pages: bool = False,
    ) -> AsiTpx3DetectorConnection:
        """
        Connect to the ASI TPX3 detector software.

        Parameters
        ----------
        uri
            host and port to connect to (example: "localhost:1234")
        num_slots
            Number of shm slots to allocate
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
        # FIXME: tweak parameters a bit
        # - uri -> data_host + data_port for consistency with other detectors
        # - num_slots -> buffer_size: total size in megabytes (see dectris)
        return AsiTpx3DetectorConnection(
            uri=uri,
            num_slots=num_slots,
            bytes_per_chunk=bytes_per_chunk,
            chunks_per_stack=chunks_per_stack,
            huge_pages=huge_pages,
        )
