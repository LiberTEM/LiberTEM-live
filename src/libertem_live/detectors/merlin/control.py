import socket
import logging

logger = logging.getLogger(__name__)


class SimpleMerlinControl:
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
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.connect((self._host, self._port))
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.settimeout(self._timeout)

    def __enter__(self):
        if self._socket is not None:
            self._protected = True
            self.connect()

    def __exit__(self, type, value, traceback):
        if self._protected:
            self.close()
            self._protected = False

    def _send(self, message):
        self._socket.sendall(message.encode("ascii"))
        done = False
        while not done:
            try:
                resp = self._socket.recv(self._buffer_size)
                done = True
            except socket.timeout:
                pass
        logger.debug('Response: {resp}'.format(resp=resp))
        return resp

    def _parse_response(self, resp):
        parts = resp.split(b',')
        msgs = ['Command OK', 'System Busy', 'Unrecognised Command',
                'Param out of range']

        rc = int(parts[5]) if parts[2] == b'GET' else int(parts[4])
        if rc > 0:
            raise Exception(msgs[rc])

        if parts[2] == b'GET':
            return parts[4]

    def _create_cmd(self, typ, cmd, value=None):
        string = ''
        if value is not None:
            string = '{ty},{cmd},{val}'.format(ty=typ, cmd=cmd, val=value)
        else:
            string = '{ty},{cmd},0'.format(ty=typ, cmd=cmd)

        msg = self._create_cmd_raw(string)

        logger.debug('Command: {cmd}'.format(cmd=msg))
        return msg

    def _create_cmd_raw(self, raw_cmd):
        msg = '{hdr},{len},{st}'.format(
            hdr=self._header,
            len=str(len(raw_cmd) + 1).zfill(self._num_digits),
            st=raw_cmd,
        )
        return msg

    def set(self, param, value):
        return self._parse_response(
            self._send(self._create_cmd('SET', param, value))
        )

    def get(self, param):
        return self._parse_response(
            self._send(self._create_cmd('GET', param))
        )

    def send_command_file(self, filename):
        with open(filename, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                self._parse_response(
                    self._send(self._create_cmd_raw(line))
                )

    def cmd(self, cmd):
        return self._parse_response(self._send(self._create_cmd('CMD', cmd)))

    def close(self):
        self._socket.close()
        self._socket = None
