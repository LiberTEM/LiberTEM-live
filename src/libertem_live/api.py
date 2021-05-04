import contextlib

from libertem.executor.inline import InlineJobExecutor
from libertem.api import Context


class LiveContext(Context):
    def _create_local_executor(self):
        '''
        Live acquisition currently requires a suitable executor, for
        example :class:`~libertem.executor.inline.InlineJobExecutor`.
        '''
        return InlineJobExecutor()

    @contextlib.contextmanager
    def _do_setup(self, dataset, udf):
        if hasattr(dataset, 'start_control') and hasattr(dataset, 'run_setup'):
            with dataset.start_control():
                # We simplify the setup interface by always supplying
                # a list of UDFs, not a single one
                if not isinstance(udf, (list, tuple)):
                    udf = [udf]
                with dataset.run_setup(udfs=udf):
                    yield
        else:
            yield

    def prepare_acquisition(self, detector_type, setup, *args, **kwargs):
        # FIXME implement similar to LiberTEM datasets once
        # we have more detector types to support
        '''
        Prepare a live data set for acquisition

        Parameters
        ----------

        detector_type : str
            - :code:`'merlin'`: Quantum Detectors Merlin camera.
            - :code:`'memory'`: Memory-based live data stream.
        setup : contextmanager
            Context manager to initialize the camera and tear it down.
            This can be used to send the commands to the camera and
            other parts of the setup to start a scan and clean up afterwards.
            It will be entered when an acquisition is started.
            Parameters supplied to the context manager are the live dataset and a list of UDFs.
            It is expected to yield after entering the dataset's
            :meth:`libertem-live.dataset.base.LiveDataSet.start_acquisition`
            context manager and perform any cleanup after that yield.
        *args, **kwargs
            Additional parameters for the acquisition, see below

        Merlin Parameters
        -----------------

        scan_size : tuple(int)
        host : str
            Hostname of the Merlin data server, default '127.0.0.1'
        port : int
            Data port of the Merlin data server, default 6342
        control_port : int
            Control port of the Merlin data server, default 6341.
            Set to :code:`None` to deactivate control.
        control_timeout : float
            Timeout for control port of the Merlin data server
            in seconds. Default 1.0.
        frames_per_partition : int
        pool_size`: int
            Number of decoding threads. Defaults to 2

        Memory Parameters
        -----------------

        *args, **kwargs
            See :class:`libertem.io.dataset.memory.MemoryDataSet`!

        Examples
        --------

        >>> @contextmanager
        ... def medipix_setup(dataset, udfs):
        ...     print("priming camera for acquisition")
        ...     # TODO: medipix control socket commands go here
        ...     # TODO interface to be tested, not supported in simulator yet

        ...     # dataset.control.set('numframes', np.prod(SCAN_SIZE, dtype=np.int64))
        ...     # dataset.control.set(...)

        ...     # microscope.configure_scan()
        ...     # microscope.start_scanning()
        ...     print("running acquisition")
        ...     with dataset.start_acquisition():
        ...         yield
        ...     print("camera teardown")
        ...     # teardown routines go here

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
        return cls(setup, *args, **kwargs).initialize(self.executor)

    def run_udf(self, dataset, udf, *args, **kwargs):
        with self._do_setup(dataset, udf):
            return super().run_udf(dataset=dataset, udf=udf, *args, **kwargs)

    def run_udf_iter(self, dataset, udf, *args, **kwargs):
        with self._do_setup(dataset, udf):
            return super().run_udf_iter(dataset=dataset, udf=udf, *args, **kwargs)
