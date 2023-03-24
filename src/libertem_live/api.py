import warnings
import contextlib
from typing import TYPE_CHECKING, overload, Literal, Union

from libertem.executor.pipelined import PipelinedExecutor
# Avoid having Context in this module to make sure
# it is not imported by accident instead of LiveContext
from libertem.api import Context as LiberTEM_Context

from opentelemetry import trace

if TYPE_CHECKING:
    from libertem_live.detectors.dectris import DectrisConnectionBuilder, DectrisAcquisitionBuilder
    from libertem_live.detectors.merlin import MerlinConnectionBuilder, MerlinAcquisitionBuilder
    from libertem_live.detectors.memory import MemoryConnectionBuilder, MemoryAcquisitionBuilder

tracer = trace.get_tracer(__name__)


def _noop(aq):
    return None


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
        if detector_type == 'dectris':
            from libertem_live.detectors.dectris import DectrisConnectionBuilder as CLS
        elif detector_type == 'merlin':
            from libertem_live.detectors.merlin import MerlinConnectionBuilder as CLS
        elif detector_type == 'memory':
            from libertem_live.detectors.memory import MemoryConnectionBuilder as CLS
        else:
            raise NotImplementedError(
                f"detector type {detector_type} doesn't support this API"
            )
        return CLS()

    @overload
    def make_acquisition(
        self,
        detector_type: Literal['dectris'],
    ) -> "DectrisAcquisitionBuilder":
        ...

    @overload
    def make_acquisition(
        self,
        detector_type: Literal['merlin'],
    ) -> "MerlinAcquisitionBuilder":
        ...

    @overload
    def make_acquisition(
        self,
        detector_type: Literal['memory'],
    ) -> "MemoryAcquisitionBuilder":
        ...

    def make_acquisition(
        self,
        detector_type: Union[
            Literal['merlin'], Literal['dectris'], Literal['memory'],
        ],
    ):
        """
        Create an acquisition object.

        Examples
        --------

        with LiveContext() as ctx:
            aq = ctx.make_acquisition('memory').open(
                data=np.random.random((23, 42, 51, 67))
            )
        """
        if detector_type == 'dectris':
            from libertem_live.detectors.dectris import DectrisAcquisitionBuilder as CLS
        elif detector_type == 'merlin':
            from libertem_live.detectors.merlin import MerlinAcquisitionBuilder as CLS
        elif detector_type == 'memory':
            from libertem_live.detectors.memory import MemoryAcquisitionBuilder as CLS
        else:
            raise NotImplementedError(
                f"detector type {detector_type} doesn't support this API"
            )
        return CLS(executor=self.executor)

    def prepare_acquisition(self, detector_type, *args, trigger=None, **kwargs):
        # FIXME implement similar to LiberTEM datasets once
        # we have more detector types to support
        '''
        Create an acquisition object.

        Parameters
        ----------

        detector_type : str
            - :code:`'merlin'`: Quantum Detectors Merlin camera.
            - :code:`'dectris'`: DECTRIS camera supporting the SIMPLON API.
            - :code:`'memory'`: Memory-based live data stream.
        trigger : function
            Keyword-only parameter, callback function to trigger an acquisition.
            See :ref:`trigger` for more information.
        *args, **kwargs
            Additional parameters for the acquisition. See :ref:`detector reference` for details
            of individual acquisition types!

        Examples
        --------

        See :ref:`usage` in the documentation!

        '''
        warnings.warn(
            "`LiveContext.prepare_acquisition` is deprecated, please use "
            "`LiveContext.make_acquisition` instead.",
            DeprecationWarning,
        )
        if trigger is None:
            trigger = _noop
        detector_type = str(detector_type).lower()
        if detector_type == 'merlin':
            from libertem_live.detectors.merlin import MerlinAcquisition as CLS
        elif detector_type == 'dectris':
            from libertem_live.detectors.dectris import DectrisAcquisition as CLS
        elif detector_type == 'memory':
            from libertem_live.detectors.memory import MemoryAcquisition as CLS
        else:
            raise ValueError(
                f"Unknown detector type '{detector_type}', supported is 'merlin' or 'dectris'"
            )
        return CLS(*args, trigger=trigger, **kwargs).initialize(self.executor)

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
