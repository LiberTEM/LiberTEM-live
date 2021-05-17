from contextlib import contextmanager
import logging

from libertem.io.dataset.memory import MemoryDataSet

from libertem_live.detectors.base.dataset import LiveDataSetMixin

logger = logging.getLogger(__name__)


class MemoryLiveDataSet(LiveDataSetMixin, MemoryDataSet):
    '''
    A live dataset based on memory

    Currently it just splices the additional functionality from
    :class:`~libertem_live.detectors.base.dataset.LiveDataSetMixin` into the
    :class:`~libertem.io.dataset.memory.MemoryDataSet` using multiple
    inheritance and implements dummies for the :meth:`start_control` and
    :meth:`start_acquisition` context managers.
    '''
    def __init__(self, setup, *args, **kwargs):
        # All parameters except setup will be passed on by the mixin to the
        # MemoryDataSet
        LiveDataSetMixin.__init__(self, *args, setup=setup, **kwargs)

    @contextmanager
    def start_control(self):
        yield

    @contextmanager
    def start_acquisition(self):
        yield
