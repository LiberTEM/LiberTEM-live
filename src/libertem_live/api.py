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
    from libertem_live.detectors.dectris import DectrisConnectionBuilder, DectrisAcquisition
    from libertem_live.detectors.dectris.connection import DectrisDetectorConnection
    from libertem_live.detectors.merlin import MerlinConnectionBuilder, MerlinAcquisition
    from libertem_live.detectors.merlin.connection import MerlinDetectorConnection
    from libertem_live.detectors.memory import MemoryConnectionBuilder, MemoryAcquisition
    from libertem_live.detectors.memory.acquisition import MemoryConnection
    from libertem_live.detectors.asi_tpx3 import (
        AsiTpx3DetectorConnection, AsiTpx3ConnectionBuilder, AsiTpx3Acquisition,
    )

tracer = trace.get_tracer(__name__)


class LiveContext(LiberTEM_Context):
    '''
    :class:`LiveContext` handles the computational resources needed to run
    UDFs on live data streams. It is the entry point to most interactions
    with the LiberTEM-live API.

    A :class:`LiveContext` behaves like a :class:`libertem.api.Context` in most
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
        detector_type: Literal['asi_tpx3'],
    ) -> "AsiTpx3ConnectionBuilder":
        ...

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
            Literal['asi_tpx3'],
            Literal['dectris'],
            Literal['merlin'],
            Literal['memory']
        ],
    ) -> Union[
        "AsiTpx3ConnectionBuilder",
        "DectrisConnectionBuilder",
        "MerlinConnectionBuilder",
        "MemoryConnectionBuilder",
    ]:
        """
        Connect to a detector system.

        Parameters
        ----------
        detector_type
            The detector type as a string. Further connection parameters
            are passed to the :code:`open` method of the returned builder object.

        Examples
        --------

        >>> data = np.random.random((23, 42, 51, 67))
        >>> ctx = LiveContext()  # doctest: +SKIP
        >>> with ctx.make_connection('memory').open(data=data) as conn:
        ...     print("connected!")
        connected!
        """
        if detector_type == 'dectris':
            from libertem_live.detectors.dectris import DectrisConnectionBuilder
            return DectrisConnectionBuilder()
        if detector_type == 'asi_tpx3':
            from libertem_live.detectors.asi_tpx3 import AsiTpx3ConnectionBuilder
            return AsiTpx3ConnectionBuilder()
        elif detector_type == 'merlin':
            from libertem_live.detectors.merlin import MerlinConnectionBuilder
            return MerlinConnectionBuilder()
        elif detector_type == 'memory':
            from libertem_live.detectors.memory import MemoryConnectionBuilder
            return MemoryConnectionBuilder()
        else:
            raise NotImplementedError(
                f"Detector type {detector_type} doesn't support this API"
            )

    @overload
    def make_acquisition(
        self,
        *,
        conn: "AsiTpx3DetectorConnection",
        nav_shape: Optional[Tuple[int, ...]] = None,
        frames_per_partition: Optional[int] = None,
        controller: Optional[AcquisitionController] = None,
        pending_aq: Optional[PendingAcquisition] = None,
        hooks: Optional[Hooks] = None,
    ) -> "AsiTpx3Acquisition":
        ...

    @overload
    def make_acquisition(
        self,
        *,
        conn: "DectrisDetectorConnection",
        nav_shape: Optional[Tuple[int, ...]] = None,
        frames_per_partition: Optional[int] = None,
        controller: Optional[AcquisitionController] = None,
        pending_aq: Optional[PendingAcquisition] = None,
        hooks: Optional[Hooks] = None,
    ) -> "DectrisAcquisition":
        ...

    @overload
    def make_acquisition(
        self,
        *,
        conn: "MerlinDetectorConnection",
        nav_shape: Optional[Tuple[int, ...]] = None,
        frames_per_partition: Optional[int] = None,
        controller: Optional[AcquisitionController] = None,
        pending_aq: Optional[PendingAcquisition] = None,
        hooks: Optional[Hooks] = None,
    ) -> "MerlinAcquisition":
        ...

    @overload
    def make_acquisition(
        self,
        *,
        conn: "MemoryConnection",
        nav_shape: Optional[Tuple[int, ...]] = None,
        frames_per_partition: Optional[int] = None,
        controller: Optional[AcquisitionController] = None,
        pending_aq: Optional[PendingAcquisition] = None,
        hooks: Optional[Hooks] = None,
    ) -> "MemoryAcquisition":
        ...

    def make_acquisition(
        self,
        *,
        conn: DetectorConnection,
        nav_shape: Optional[Tuple[int, ...]] = None,
        frames_per_partition: Optional[int] = None,
        controller: Optional[AcquisitionController] = None,
        pending_aq: Optional[PendingAcquisition] = None,
        hooks: Optional[Hooks] = None,
    ) -> Union[
        "AsiTpx3Acquisition",
        "DectrisAcquisition",
        "MerlinAcquisition",
        "MemoryAcquisition",
    ]:
        """
        Create an acquisition object.

        Examples
        --------

        >>> data = np.random.random((23, 42, 51, 67))
        >>> ctx = LiveContext()  # doctest: +SKIP
        >>> with ctx.make_connection('memory').open(data=data) as conn:
        ...     aq = ctx.make_acquisition(conn=conn)
        ...     ctx.run_udf(dataset=aq, udf=SumUDF())
        {'intensity': ...}
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
