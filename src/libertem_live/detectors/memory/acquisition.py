from contextlib import contextmanager
import logging

from libertem.io.dataset.memory import MemoryDataSet

from libertem_live.detectors.base.acquisition import AcquisitionMixin

logger = logging.getLogger(__name__)


class MemoryAcquisition(AcquisitionMixin, MemoryDataSet):
    '''
    An acquisition based on memory

    Currently it just splices the additional functionality from
    :class:`~libertem_live.detectors.base.dataset.AcquisitionMixin` into the
    :class:`~libertem.io.dataset.memory.MemoryDataSet` using multiple
    inheritance and implements a dummy for the :meth:`acquire` context manager.

    Parameters
    ----------

    trigger : function()
        See :meth:`~libertem_live.api.LiveContext.prepare_acquisition`
        and :ref:`trigger` for details!

    Notes
    -----

    Other parameters are inherited from and passed
    to :class:`~libertem.io.dataset.memory.MemoryDataSet`.
    '''

    @contextmanager
    def acquire(self):
        self.trigger()
        yield