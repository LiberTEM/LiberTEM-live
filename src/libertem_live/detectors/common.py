import contextlib
import threading
import logging
import queue
from typing import Optional
import multiprocessing as mp

import cloudpickle
import zmq

logger = logging.getLogger(__name__)


class RecvTimeout(Exception):
    pass


def send_serialized(socket, msg):
    s = cloudpickle.dumps(msg, protocol=5)
    return socket.send(s)


def recv_serialized(socket, timeout=None):
    if timeout is not None:
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        events = dict(poller.poll(timeout))
        if socket not in events:
            raise RecvTimeout("timeout receiving response")
    msg = socket.recv()
    s = cloudpickle.loads(msg)
    return s


class SerializedQueue:
    def __init__(self):
        self.q = mp.Queue()

    def put(self, msg, **kwargs):
        serialized = cloudpickle.dumps(msg, protocol=5)
        return self.q.put(serialized, **kwargs)

    def get(self, **kwargs):
        return self.q.get(**kwargs)

    def close(self):
        return self.q.close()


class StoppableThreadMixin:
    def __init__(self, *args, **kwargs):
        self._stop_event = threading.Event()
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


class ReaderPoolImpl:
    def __init__(self, backend, pool_size, cls, reader_kwargs):
        self._pool_size = pool_size
        self._backend = backend
        self._out_queue = queue.Queue()  # TODO: possibly limit size?
        self._threads = None
        self._reader_kwargs = reader_kwargs
        self._cls = cls

    def __enter__(self):
        self.start_threads()
        return self

    def __exit__(self, *args, **kwargs):
        logger.debug("ReaderPoolImpl.__exit__: stopping threads")
        for t in self._threads:  # TODO: handle errors on stopping/joining? re-throw exceptions?
            t.stop()
            logger.debug("ReaderPoolImpl: stop signal set")
            t.join()
            logger.debug("ReaderPoolImpl: thread joined")
        logger.debug("ReaderPoolImpl.__exit__: threads stopped")

    def start_threads(self):
        self._threads = []
        for i in range(self._pool_size):
            t = self.cls(
                backend=self._backend,
                out_queue=self._out_queue,
                **self._reader_kwargs,
            )
            t.start()
            self._threads.append(t)

    @contextlib.contextmanager
    def get_result(self):
        while True:
            try:
                res = self._out_queue.get(timeout=0.2)
                try:
                    yield res
                finally:
                    res.release()
                return
            except queue.Empty:
                if self.should_stop():
                    yield None
                    return

    def should_stop(self):
        return any(
            t.is_stopped()
            for t in self._threads
        )


class ReaderPool:
    def __init__(self, pool_size, reader_cls, impl_cls=ReaderPoolImpl):
        self._pool_size = pool_size
        self._reader_cls = reader_cls
        self._impl_cls = impl_cls

    def get_impl(self, reader_kwargs=None):
        """
        Returns a new `ReaderPoolImpl`, passing `reader_kwargs`
        to the `ReaderThread` constructor.
        """
        return self._impl_cls(
            pool_size=self._pool_size,
            cls=self._reader_cls,
            reader_kwargs=reader_kwargs or {},
        )
