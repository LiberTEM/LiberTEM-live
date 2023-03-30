import time
import socket
import threading
from contextlib import contextmanager
import functools

import pytest

from libertem_live.detectors.merlin.sim import (
        StopException, ServerThreadMixin,
)


class BadServer(ServerThreadMixin, threading.Thread):
    def __init__(self, exception, *args, **kwargs):
        self.exception = exception
        super().__init__(*args, **kwargs)

    def handle_conn(self, connection):
        raise self.exception


class OtherError(Exception):
    pass


def serve(cls, host='127.0.0.1', port=0):
    server = cls(host=host, port=port)
    server.start()
    server.wait_for_listen()
    yield server
    print("cleaning up server thread")
    server.maybe_raise()
    print("stopping server thread")
    server.stop()
    timeout = 2
    start = time.time()
    while True:
        print("are we there yet?")
        server.maybe_raise()
        if not server.is_alive():
            print("server is dead, we are there")
            break
        if (time.time() - start) >= timeout:
            raise RuntimeError("Server didn't stop gracefully")
        time.sleep(0.1)


@pytest.mark.parametrize(
    'exception_cls', (RuntimeError, ValueError, OtherError)
)
def test_server_throws(exception_cls):
    server = contextmanager(serve)
    exception = exception_cls("Testing...")
    cls = functools.partial(BadServer, exception=exception, name="BadServer")
    with pytest.raises(exception_cls, match="Testing..."):
        with server(cls) as serv:
            host, port = serv.sockname
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((host, port))
                time.sleep(1)
                print("second try...")
                # Making sure the server is stopped
                with pytest.raises(ConnectionRefusedError):
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s2:
                        s2.connect((host, port))


def test_server_stop():
    server = contextmanager(serve)
    exception = StopException("Testing...")
    cls = functools.partial(BadServer, exception=exception, name="BadServer")
    with server(cls) as serv:
        host, port = serv.sockname
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
            time.sleep(1)
            # The above exception should have led to an immediate graceful stop of the server
            with pytest.raises(ConnectionRefusedError):
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s2:
                    s2.connect((host, port))
                    print(s2.getsockname())
