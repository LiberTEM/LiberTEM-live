import contextlib

from libertem.executor.inline import InlineJobExecutor
from libertem.api import Context


class LiveContext(Context):
    def __init__(self, executor=None):
        if executor is None:
            executor = InlineJobExecutor()
        super().__init__(executor=executor)

    @contextlib.contextmanager
    def _do_camera_setup(self, dataset, udf):
        if hasattr(dataset, 'camera_setup'):
            with dataset.camera_setup(dataset, udf):
                yield
        else:
            yield

    def prepare_acquisition(self, detector_type, camera_setup, *args, **kwargs):
        # FIXME implement similar to LiberTEM datasets once
        # we have more detector types to support
        '''
        Prepare a live data set for acquisition

        Parameters
        ----------
        detector_type : str
            - :code:`'merlin'`: Quantum Detectors Merlin camera. Additional parameters:
        camera_setup : contextmanager
            Context manager to initialize the camera and tear it down.
            This can be used to send the commands to the camera and
            other parts of the setup to start a scan and clean up afterwards.
            It will be entered when an acquisition is started.
            Parameters supplied to the context manager are the live dataset and a list of UDFs.
        * args, **kwargs
            Additional parameters for the acquisition, see below

        Merlin Parameters
        -----------------

        * :code:`scan_size`: tuple(int)
        * :code:`host`: str, hostname of the Merlin data server
        * :code:`port`: int, port of the Merlin data server
        * :code:`frames_per_partition`: int
        * :code:`pool_size`: int, number of decoding threads. Defaults to 2
        '''
        detector_type = str(detector_type).lower()
        if detector_type == 'merlin':
            from libertem_live.detectors.merlin.dataset import MerlinLiveDataSet
            cls = MerlinLiveDataSet
        else:
            raise ValueError(f"Unknown detector type '{detector_type}', supported is 'merlin'")
        return cls(camera_setup, *args, **kwargs).initialize(self.executor)

    def run_udf(self, dataset, udf, *args, **kwargs):
        if not isinstance(udf, (list, tuple)):
            udf = [udf]
        with self._do_camera_setup(dataset, udf):
            return super().run_udf(dataset=dataset, udf=udf, *args, **kwargs)

    def run_udf_iter(self, dataset, udf, *args, **kwargs):
        if not isinstance(udf, (list, tuple)):
            udf = [udf]
        with self._do_camera_setup(dataset, udf):
            return super().run_udf_iter(dataset=dataset, udf=udf, *args, **kwargs)
