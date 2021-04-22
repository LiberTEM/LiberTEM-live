import threading

from ..common import StoppableThreadMixin
from ..utils.net import bind_mcast_socket


class K2ISDataSocket:
    GROUP = '225.1.1.1'

    def __init__(self, port, local_addr, iface):
        self._socket = None
        self._port = port
        self._local_addr = local_addr
        self._iface = iface

    def connect(self):
        self._socket = bind_mcast_socket(
            port=self._port,
            group=self.GROUP,
            local_addr=self._local_addr,
            iface=self._iface,
        )

    def close(self):
        self._socket.close()
        self._socket = None

    def __enter__(self):
        self.connect()

    def __exit__(self, type, value, traceback):
        self.close()


class K2ISReaderThread(StoppableThreadMixin, threading.Thread):
    """
    Thread that reads and decodes data from the K2IS camera.

    Each thread takes care of data from one socket, which receives data from
    a single sector of the camera. There can be more than one thread per
    sector; in that case, reading and decoding can overlap.

    TODO: synchronization w/ STEMx, or at least the primitives we can
    offer. For example, keeping a small ring buffer, and enabling processing
    by frame_id, or similar (in case we already received the data)
    """
    def __init__(self, socket, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def read_loop(self):
        pass

    def run(self):
        try:
            self.read_loop()
        finally:
            self.stop()
