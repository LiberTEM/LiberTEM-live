from contextlib import contextmanager
import logging

from libertem.io.dataset.memory import MemoryDataSet

from libertem_live.detectors.base.dataset import LiveDataSet

logger = logging.getLogger(__name__)


class MemoryLiveDataSet(LiveDataSet, MemoryDataSet):
    '''
    A live dataset based on memory

    Currently it just splices the additional functionality from
    :class:`~libertem_live.detectors.base.dataset.LiveDataSet` into the
    :class:`~libertem.io.dataset.memory.MemoryDataSet` using multiple
    inheritance and implements dummies for the :meth:`start_control` and
    :meth:`start_acquisition` context managers.

    Note that this creates a diamond dependency graph since both
    :class:`~libertem_live.detectors.base.dataset.LiveDataSet` and
    :class:`~libertem.io.dataset.memory.MemoryDataSet` are subclasses of
    :class:`~libertem.io.dataset.base.DataSet`.
    '''
    def __init__(self, setup, *args, **kwargs):
        LiveDataSet.__init__(self, setup=setup)
        MemoryDataSet.__init__(self, *args, **kwargs)

    @contextmanager
    def start_control(self):
        yield

    @contextmanager
    def start_acquisition(self):
        yield
