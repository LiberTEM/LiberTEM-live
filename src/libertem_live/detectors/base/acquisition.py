from contextlib import contextmanager
import logging


logger = logging.getLogger(__name__)


class AcquisitionTimeout(Exception):
    pass


class AcquisitionMixin:
    def __init__(self, trigger, *args, **kwargs):
        self._trigger = trigger
        super().__init__(*args, **kwargs)

    @contextmanager
    def acquire(self):
        raise NotImplementedError

    def trigger(self):
        if self._trigger is not None:
            self._trigger(self)
