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
    inheritance and implements a dummy for the :meth:`acquire` context manager.

    Parameters
    ----------

    on_enter, on_exit : function(LiveMeta)
        See :meth:`~libertem_live.api.LiveContext.prepare_acquisition`
        and :ref:`enter exit` for details!

    Notes
    -----

    Other parameters are inherited from and passed
    to :class:`~libertem.io.dataset.memory.MemoryDataSet`.
    '''
    def __init__(self, on_enter, on_exit, *args, **kwargs):
        # All parameters except setup will be passed on by the mixin to the
        # MemoryDataSet
        LiveDataSetMixin.__init__(self, *args, on_enter=on_enter, on_exit=on_exit, **kwargs)

    @contextmanager
    def acquire(self):
        yield
