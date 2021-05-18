import contextlib

from libertem.executor.inline import InlineJobExecutor
from libertem.api import Context

from libertem_live.detectors.base.meta import LiveMeta


class LiveContext(Context):
    def _create_local_executor(self):
        '''
        Live acquisition currently requires a suitable executor, for
        example :class:`~libertem.executor.inline.InlineJobExecutor`.
        '''
        return InlineJobExecutor()

    @contextlib.contextmanager
    def _do_acquisition(self, dataset, udf):
        if hasattr(dataset, 'run_acquisition'):
            if not isinstance(udf, (list, tuple)):
                udf = [udf]
            meta = LiveMeta(
                dataset_shape=dataset.shape,
                dataset_dtype=dataset.dtype,
                udfs=udf
            )
            with dataset.run_acquisition(meta=meta):
                yield
        else:
            yield

    def prepare_acquisition(self, detector_type, *args, on_enter=None, on_exit=None, **kwargs):
        # FIXME implement similar to LiberTEM datasets once
        # we have more detector types to support
        '''
        Prepare a live data set for acquisition

        Parameters
        ----------

        detector_type : str
            - :code:`'merlin'`: Quantum Detectors Merlin camera.
            - :code:`'memory'`: Memory-based live data stream.
        on_enter, on_exit : function(LiveMeta)
            Keyword-only parameter, function to set up and initiate an acquisition resp.
            clean up afterwards. This can be used to send the commands to the camera and
            other parts of the setup to start a scan.
            :code:`on_enter(meta)` will be called before an acquisition is started.
            :code:`on_exit(meta)` will be called after an acquisition is finished, including
            after an error.
            The function is called with a :class:`libertem_live.detectors.base.meta.LiveMeta`
            object as a parameter that contains information about the current acquisition.
        *args, **kwargs
            Additional parameters for the acquisition. See :ref:`detector reference` for details
            of individual acquisition types!

        Examples
        --------

        >>> import numpy as np
        >>>
        >>> from libertem_live.udf.monitor import SignalMonitorUDF
        >>> from libertem_live import api as ltl
        >>>
        >>> ctx = ltl.LiveContext()
        >>> def on_enter(meta):
        ...     print("Calling on_enter")
        ...     print("Dataset shape:", meta.dataset_shape)
        >>>
        >>> def on_exit(meta):
        ...     print("Calling on_exit")
        >>>
        >>> # We use a memory-based acquisition for demonstration
        >>> # This allows to run this example without a real detector
        >>> data = np.random.random((23, 42, 51, 67))
        >>>
        >>> ds = ctx.prepare_acquisition(
        ...     'memory',
        ...     on_enter=on_enter,
        ...     on_exit=on_exit,
        ...     data=data,
        ... )
        >>>
        >>>
        >>> udf = SignalMonitorUDF()
        >>>
        >>> res = ctx.run_udf(dataset=ds, udf=udf, plots=True)
        Calling on_enter
        Dataset shape: (23, 42, 51, 67)
        Calling on_exit
        '''
        detector_type = str(detector_type).lower()
        if detector_type == 'merlin':
            from libertem_live.detectors.merlin import MerlinLiveDataSet
            cls = MerlinLiveDataSet
        elif detector_type == 'memory':
            from libertem_live.detectors.memory import MemoryLiveDataSet
            cls = MemoryLiveDataSet
        else:
            raise ValueError(f"Unknown detector type '{detector_type}', supported is 'merlin'")
        return cls(*args, on_enter=on_enter, on_exit=on_exit, **kwargs).initialize(self.executor)

    def run_udf(self, dataset, udf, *args, **kwargs):
        with self._do_acquisition(dataset, udf):
            return super().run_udf(dataset=dataset, udf=udf, *args, **kwargs)

    def run_udf_iter(self, dataset, udf, *args, **kwargs):
        with self._do_acquisition(dataset, udf):
            return super().run_udf_iter(dataset=dataset, udf=udf, *args, **kwargs)
