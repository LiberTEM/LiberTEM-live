import threading
import logging
from typing import Optional
import socket
import time

try:
    import prctl
except ImportError:
    prctl = None

logger = logging.getLogger(__name__)


class UndeadException(Exception):
    pass


class StopException(Exception):
    pass


class StoppableThreadMixin:
    def __init__(
        self,
        stop_event: Optional[threading.Event] = None,
        *args, **kwargs
    ):
        if stop_event is None:
            stop_event = threading.Event()
        self._stop_event = stop_event
        super().__init__(*args, **kwargs)

    def stop(self):
        self._stop_event.set()

    def is_stopped(self):
        return self._stop_event.is_set()


class ErrThreadMixin(StoppableThreadMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._error: Optional[Exception] = None

    def get_error(self):
        return self._error

    def error(self, exc):
        logger.error("got exception %r, shutting down thread", exc)
        self._error = exc
        self.stop()

    def maybe_raise(self):
        if self._error is not None:
            raise self._error


class ServerThreadMixin(ErrThreadMixin):
    def __init__(self, host, port, name, *args, listen_event=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._host = host
        self._port = port
        self._name = name
        if listen_event is None:
            listen_event = threading.Event()
        self.listen_event = listen_event

    def wait_for_listen(self, timeout=30):
        """
        To be called from the main thread
        """
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.listen_event.wait(timeout=0.1):
                return
            self.maybe_raise()
        if not self.listen_event.is_set():
            raise RuntimeError("failed to start in %f seconds" % timeout)

    @property
    def sockname(self):
        return self._socket.getsockname()

    @property
    def port(self) -> int:
        return self.sockname[1]

    def handle_conn(self, connection: socket.socket):
        raise NotImplementedError

    def run(self):
        try:
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._socket.bind((self._host, self._port))
            self._socket.settimeout(0.1)
            self._socket.listen(1)
            set_thread_name(f'server:{self._name}')
            logger.info(f"{self._name} listening {self.sockname}")
            self.listen_event.set()
            while not self.is_stopped():
                try:
                    connection, client_addr = self._socket.accept()
                    with connection:
                        logger.info(f"{self._name}: accepted from %s" % (client_addr,))
                        self.handle_conn(connection)
                        logger.info(f"{self._name}: handling done for %s" % (client_addr,))
                except socket.timeout:
                    continue
                except BrokenPipeError:
                    if "client_addr" in locals():
                        logger.info(f"BrokenPipeError: {locals()['client_addr']}")
                    else:
                        logger.info("BrokenPipeError")

                    continue  # the other end died, but that doesn't mean we have to die
                except ConnectionResetError:
                    if "client_addr" in locals():
                        logger.info(f"{self._name}: client {locals()['client_addr']} disconnected")
                    else:
                        logger.info(f"{self._name}: client disconnected")
                # except BrokenPipeError:
                #     # catch broken pipe error - allow the server to continue
                #     # running, even when a client prematurely disconnects
                #     try:
                #         connection.close()
                #     except Exception:
                #         pass
                #     print(f"{self._name} disconnected")
                except RuntimeError as e:
                    print(f"{self._name} exception %s -> stopping" % e)
                    self.error(e)
                    break
                except StopException:
                    break
                except Exception as e:
                    print(f"{self._name} exception? %s" % e)
                    self.error(e)
        except Exception as e:
            return self.error(e)
        finally:
            logger.info(f"{self._name} exiting")
            self._socket.close()


def set_thread_name(name: str):
    """
    Set a thread name; mostly useful for using system tools for profiling

    Parameters
    ----------
    name : str
        The thread name
    """
    if prctl is None:
        return
    prctl.set_name(name)
