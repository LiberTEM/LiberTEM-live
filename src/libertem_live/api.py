import contextlib

from libertem.executor.inline import InlineJobExecutor
from libertem.api import Context


class LiveContext(Context):
    def __init__(self, camera_setup, executor=None):
        if executor is None:
            executor = InlineJobExecutor()
        super().__init__(executor=executor)
        self._camera_setup = camera_setup

    @contextlib.contextmanager
    def _do_camera_setup(self, dataset, udf):
        with self._camera_setup(dataset, udf):
            yield

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
