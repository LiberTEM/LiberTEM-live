import logging

import numpy as np

from libertem.io.dataset.memory import MemoryDataSet

from libertem_live.hooks import Hooks, ReadyForDataEnv
from libertem_live.detectors.base.acquisition import AcquisitionMixin
from libertem_live.detectors.base.connection import DetectorConnection, PendingAcquisition
# from libertem_live.detectors.base.controller import AcquisitionController

logger = logging.getLogger(__name__)


class MemoryConnection(DetectorConnection):
    def __init__(self, data: np.ndarray, extra_kwargs: dict | None = None):
        self._data = data
        if extra_kwargs is None:
            extra_kwargs = {}
        self._extra_kwargs = extra_kwargs

    def wait_for_acquisition(self, timeout: float | None = None) -> PendingAcquisition | None:
        return PendingMemAq()

    def get_acquisition_cls(self) -> type[AcquisitionMixin]:
        return MemoryAcquisition

    def close(self):
        pass


class MemoryConnectionBuilder:
    def open(self, data: np.ndarray, extra_kwargs: dict | None = None):
        return MemoryConnection(
            data=data,
            extra_kwargs=extra_kwargs,
        )


class PendingMemAq(PendingAcquisition):
    pass


class MemoryAcquisition(AcquisitionMixin, MemoryDataSet):
    '''
    An acquisition based on memory

    Currently it just splices the additional functionality from
    :class:`~libertem_live.detectors.base.dataset.AcquisitionMixin` into the
    :class:`~libertem.io.dataset.memory.MemoryDataSet` using multiple
    inheritance and implements a dummy for the :meth:`acquire` context manager.

    Examples
    --------

    >>> import numpy as np
    >>> from libertem_live.api import Hooks
    ...
    >>> data = np.random.random((23, 42, 51, 67))
    ...
    >>> class MyHooks(Hooks):
    ...     def on_ready_for_data(self, env):
    ...         print(f"Triggering! {env.aq.shape.nav}")
    ...
    >>> conn = ctx.make_connection('memory').open(
    ...    data=data
    ... )
    ...
    >>> aq = ctx.make_acquisition(
    ...     conn=conn,
    ...     hooks=MyHooks(),
    ... )
    ...
    >>> udf = SumUDF()
    >>> ctx.run_udf(dataset=aq, udf=udf, plots=True)
    Triggering! (23, 42)
    {'intensity': <BufferWrapper kind=sig dtype=float64 extra_shape=()>}
    '''

    def __init__(
        self,
        conn: "MemoryConnection",
        nav_shape: tuple[int, ...],
        frames_per_partition: int = 128,
        # in passive mode, we get this:
        pending_aq: PendingMemAq | None = None,
        # controller is unused as of now, you can only pass in `None`:
        controller: None = None,
        hooks: Hooks | None = None,
    ):
        # XXX copy/pasta from AcquisitionMixin as we do need to
        # pass extra kwargs to the memory data set underneath:
        self._conn = conn
        self._nav_shape = nav_shape
        self._frames_per_partition = frames_per_partition  # FIXME: ignored!
        self._controller = controller
        self._pending_aq = pending_aq
        if hooks is None:
            hooks = Hooks()
        self._hooks = hooks

        MemoryDataSet.__init__(
            self,
            data=conn._data,
            nav_shape=nav_shape,
            **conn._extra_kwargs,
        )

    def start_acquisition(self):
        if self._pending_aq is None:
            self._hooks.on_ready_for_data(
                ReadyForDataEnv(aq=self),
            )

    def end_acquisition(self):
        pass
