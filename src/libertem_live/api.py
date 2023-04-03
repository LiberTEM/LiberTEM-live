import contextlib
from typing import TYPE_CHECKING, overload, Union, Tuple, Optional
from typing_extensions import Literal

from opentelemetry import trace

from libertem.executor.pipelined import PipelinedExecutor
# Avoid having Context in this module to make sure
# it is not imported by accident instead of LiveContext
from libertem.api import Context as LiberTEM_Context


from .detectors.base.connection import DetectorConnection, PendingAcquisition
from .detectors.base.controller import AcquisitionController
from .hooks import Hooks

if TYPE_CHECKING:
    from libertem_live.detectors.dectris import DectrisConnectionBuilder
    from libertem_live.detectors.merlin import MerlinConnectionBuilder
    from libertem_live.detectors.memory import MemoryConnectionBuilder

tracer = trace.get_tracer(__name__)


class LiveContext(LiberTEM_Context):
    '''
    A :class:`LiveContext` behaves like a :class:`~libertem.api.Context` in most
    circumstances. Notable differences are that it currently starts an
    :class:`~libertem.executor.pipelined.PipelinedExecutor` instead of a
    :class:`~libertem.executor.dask.DaskJobExecutor` if no executor is passed in
    the constructor, and that it can prepare and run acquisitions on top of
    loading offline datasets.

    The docstrings for most functions are inherited from the base class. Most
    methods, in particular :meth:`run_udf` and :meth:`run_udf_iter`, now accept
    both an acquisition object and a dataset as the :code:`dataset` parameter.
    '''
    def _create_local_executor(self):
        '''
        Live acquisition currently requires a suitable executor, for
        example :class:`~libertem.executor.pipelined.PipelinedExecutor`.
        '''
        return PipelinedExecutor()

    @contextlib.contextmanager
    def _do_acquisition(self, acquisition, udf):
        with tracer.start_as_current_span("LiveContext._do_acquisition"):
            if hasattr(acquisition, 'acquire'):
                with acquisition.acquire():
                    yield
            else:
                yield

    @overload
    def make_connection(
        self,
        detector_type: Literal['dectris'],
    ) -> "DectrisConnectionBuilder":
        ...

    @overload
    def make_connection(
        self,
        detector_type: Literal['merlin'],
    ) -> "MerlinConnectionBuilder":
        ...

    @overload
    def make_connection(
        self,
        detector_type: Literal['memory'],
    ) -> "MemoryConnectionBuilder":
        ...

    def make_connection(
        self,
        detector_type: Union[
            Literal['dectris'],
            Literal['merlin'],
            Literal['memory']
        ],
    ):
        """
        Connect to a detector system.
        """
        # FIXME: type hint for cls?
        if detector_type == 'dectris':
            from libertem_live.detectors.dectris import DectrisConnectionBuilder
            cls = DectrisConnectionBuilder
        elif detector_type == 'merlin':
            from libertem_live.detectors.merlin import MerlinConnectionBuilder
            cls = MerlinConnectionBuilder
        elif detector_type == 'memory':
            from libertem_live.detectors.memory import MemoryConnectionBuilder
            cls = MemoryConnectionBuilder
        else:
            raise NotImplementedError(
                f"detector type {detector_type} doesn't support this API"
            )
        return cls()

    def make_acquisition(
        self,
        *,
        conn: DetectorConnection,
        nav_shape: Optional[Tuple[int, ...]] = None,
        frames_per_partition: Optional[int] = None,
        controller: Optional[AcquisitionController] = None,
        pending_aq: Optional[PendingAcquisition] = None,
        hooks: Optional[Hooks] = None,
    ):
        """
        Create an acquisition object.

        Examples
        --------

        >>> data = np.random.random((23, 42, 51, 67))
        >>> with LiveContext() as ctx,\
        ...        ctx.make_connection('memory').open(data=data) as conn:
        ...     aq = ctx.make_acquisition(conn=conn)
        """
        cls = conn.get_acquisition_cls()
        instance = cls(
            conn=conn,
            nav_shape=nav_shape,
            frames_per_partition=frames_per_partition,
            controller=controller,
            pending_aq=pending_aq,
            hooks=hooks,
        )
        return instance.initialize(self.executor)

    def prepare_acquisition(self, detector_type, *args, trigger=None, **kwargs):
        # FIXME implement similar to LiberTEM datasets once
        # we have more detector types to support
        '''
        This method has been removed, please use `make_connection` and `make_acquisition`.
        '''
        raise RuntimeError(
            "`LiveContext.prepare_acquisition` has been replaced with "  # FIXME: text, docstring
            "`LiveContext.make_connection` and `LiveContext.make_acquisition`."
        )

    def _run_sync(self, dataset, udf, iterate=False, *args, **kwargs):
        def _run_sync_iterate():
            with self._do_acquisition(dataset, udf):
                res = super(LiveContext, self)._run_sync(
                    dataset=dataset, udf=udf, iterate=iterate, *args, **kwargs
                )
                yield from res

        if iterate:
            return _run_sync_iterate()
        else:
            with self._do_acquisition(dataset, udf):
                return super()._run_sync(dataset=dataset, udf=udf, iterate=iterate, *args, **kwargs)
