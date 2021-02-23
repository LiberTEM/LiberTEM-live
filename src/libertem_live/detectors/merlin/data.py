import socket
import logging

logger = logging.getLogger(__name__)


class EOFError(Exception):
    pass


class TCPBackend:
    def __init__(self, host='127.0.0.1', port=6342, timeout=1.0):
        self._host = host
        self._port = port
        self._timeout = timeout
        self._socket = None
        self._acquisition_header = None
        self._is_connected = False

    def connect(self):
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.connect((self._host, self._port))
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.settimeout(self._timeout)
        self._is_connected = True

    def is_connected(self):
        return self._is_connected

    def read(self, length):
        """
        read exactly length bytes from the socket

        note: efficiency to be evaluated
        """
        if not self.is_connected():
            raise RuntimeError("can't read without connection")
        buf = b''
        bytes_read = 0
        while bytes_read < length:
            try:
                data = self._socket.recv(length - bytes_read)
            except socket.timeout:
                continue
            if len(data) == 0:
                raise EOFError("EOF")
            bytes_read += len(data)
            buf += data
        return buf

    def read_mpx_length(self):
        # structure: MPX,<ten digits>,<header>
        hdr = self.read(15)
        # logger.debug("MPX prefix: %r", hdr)
        assert hdr.startswith(b'MPX,'), "Should start with MPX, first bytes are %r" % hdr[:16]
        parts = hdr.split(b',')
        length = int(parts[1])
        # we already consumed the comma, which seems to be part of the
        # length calculation, that's why we substract 1 here:
        return length - 1

    def read_acquisition_header(self):
        # assumption: when we connect, the connection is idle
        # so the first thing we will get is the acquisition header.
        # we read it in an inefficient way, but the header is small,
        # so this should be ok:
        length = self.read_mpx_length()
        header = self.read(length)
        header = self._parse_acq_header(header)
        self._acquisition_header = header
        return header

    def get_acquisition_header(self):
        return self._acquisition_header

    def drain(self):
        """
        read data from the data socket until we hit the timeout; returns
        the number of bytes drained
        """
        bytes_read = 0
        # read from the socket until we hit the timeout:
        while True:
            try:
                data = self._socket.recv(4096)
                bytes_read += len(data)
            except socket.timeout:
                return bytes_read

    def _parse_acq_header(self, header):
        result = {}
        for line in header.decode("latin1").split('\n'):
            try:
                if line.startswith("HDR") or line.startswith("End\t"):
                    continue
                k, v = line.split("\t", 1)
                k = k.rstrip(':')
                v = v.rstrip("\r")
                v = v.rstrip("\n")
            except ValueError:
                logger.debug("error while parsing line %r", line)
                raise
            result[k] = v
        return result

    def close(self):
        self._socket.close()
        self._is_connected = False

    def __enter__(self):
        self.connect()

    def __exit__(self, type, value, traceback):
        self.close()


class FakeLocalBackend(TCPBackend):
    def __init__(self, path):
        self._path = path
        self._fh = open(path, "rb")
        super().__init__()

    def read(self, length):
        return self._fh.read(length)
