import socket
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class MerlinControl:
    '''
    This class can be used to control a merlin medipix detector.

    Parameters
    ----------
    host : str
        The hostname to connect to

    port : int
        The port to connect to

    timeout : float
        The timeout, in seconds, after which a response is expected
    '''
    def __init__(self, host='127.0.0.1', port=6341, timeout=1.0):
        self._host = host
        self._port = port
        self._timeout = timeout
        self._socket = None

        self._header = 'MPX'
        self._num_digits = 10
        self._buffer_size = 4096
        self._protected = False

    def connect(self):
        """
        Connect to the merlin control socket. Usually, you would instead use this
        class as a context manager.
        """
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.connect((self._host, self._port))
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.settimeout(self._timeout)

    def __enter__(self):
        self._protected = True
        self.connect()
        return self

    def __exit__(self, type, value, traceback):
        if self._protected:
            self.close()
            self._protected = False

    def _send(self, message):
        assert self._socket is not None, "must be connected to send"
        self._socket.sendall(message.encode("ascii"))
        done = False
        resp = None
        while not done:
            try:
                resp = self._socket.recv(self._buffer_size)
                done = True
            except socket.timeout:
                pass
        logger.debug(f'Response: {resp}')
        return resp

    def _parse_response(self, resp) -> Optional[bytes]:
        parts = resp.split(b',')
        msgs = ['Command OK', 'System Busy', 'Unrecognised Command',
                'Param out of range']

        rc = int(parts[5]) if parts[2] == b'GET' else int(parts[4])
        if rc > 0:
            raise Exception(msgs[rc])

        if parts[2] == b'GET':
            return parts[4]
        else:
            # no response expected:
            return None

    def _create_cmd(self, typ, cmd, value=None):
        string = ''
        if value is not None:
            string = f'{typ},{cmd},{value}'
        else:
            string = f'{typ},{cmd},0'

        msg = self._create_cmd_raw(string)

        logger.debug(f'Command: {msg}')
        return msg

    def _create_cmd_raw(self, raw_cmd):
        msg = '{hdr},{len},{st}'.format(
            hdr=self._header,
            len=str(len(raw_cmd) + 1).zfill(self._num_digits),
            st=raw_cmd,
        )
        return msg

    def set(self, param, value):
        """
        Send a SET command, and return the response
        """
        return self._parse_response(
            self._send(self._create_cmd('SET', param, value))
        )

    def get(self, param) -> bytes:
        """
        Send a GET command, and return the response
        """
        parsed = self._parse_response(
            self._send(self._create_cmd('GET', param))
        )
        assert parsed is not None, "for GET commands, the repsonse must not be None"
        return parsed

    def send_command_file(self, filename):
        """
        Send the contents of :code:`filename`, which should
        contain complete merlin command lines.
        """
        with open(filename) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                self._parse_response(
                    self._send(self._create_cmd_raw(line))
                )

    def cmd(self, cmd):
        """
        Send a CMD command, and return the response
        """
        return self._parse_response(self._send(self._create_cmd('CMD', cmd)))

    def close(self):
        """
        Close the socket connection. Usually, instead of calling this function,
        you should use this class as a context manager.
        """
        if self._socket is not None:
            self._socket.close()
            self._socket = None
