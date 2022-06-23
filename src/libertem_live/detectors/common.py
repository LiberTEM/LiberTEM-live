import threading
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class StoppableThreadMixin:
    def __init__(self, stop_event: threading.Event = None, *args, **kwargs):
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
