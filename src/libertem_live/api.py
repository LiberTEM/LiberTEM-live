import contextlib

from libertem.executor.pipelined import PipelinedExecutor
# Avoid having Context in this module to make sure
# it is not imported by accident instead of LiveContext
from libertem.api import Context as LiberTEM_Context


class LiveContext(LiberTEM_Context):
    '''
    A :class:`LiveContext` behaves like a :class:`~libertem.api.Context` in most
    circumstances. Notable differences are that it currently starts an
    :class:`~libertem.executor.inline.InlineJobExecutor` instead of a
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
        if hasattr(acquisition, 'acquire'):
            with acquisition.acquire():
                yield
        else:
            yield

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
        detector_type = str(detector_type).lower()
        if detector_type == 'merlin':
            from libertem_live.detectors.merlin import MerlinAcquisition
            cls = MerlinAcquisition
        elif detector_type == 'dectris':
            from libertem_live.detectors.dectris import DectrisAcquisition
            cls = DectrisAcquisition
        elif detector_type == 'memory':
            from libertem_live.detectors.memory import MemoryAcquisition
            cls = MemoryAcquisition
        else:
            raise ValueError(f"Unknown detector type '{detector_type}', supported is 'merlin'")
        return cls(*args, trigger=trigger, **kwargs).initialize(self.executor)

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
