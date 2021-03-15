from libertem.executor.inline import InlineJobExecutor
from libertem.executor.base import AsyncAdapter
from libertem.udf.base import UDFRunner


class LiveContext:
    def __init__(self, camera_setup):
        self._executor = InlineJobExecutor()
        self._async_executor = AsyncAdapter(self._executor)
        self._camera_setup = camera_setup

    def run(self, dataset, udfs, plots=None, sync=False):
        if plots is None:
            plots = []
        if sync:
            return self._run_sync(dataset, udfs, plots)
        else:
            return self._run_async(dataset, udfs, plots)

    async def _run_async(self, dataset, udfs, plots):
        with self._camera_setup(dataset, udfs):
            udfres_iter = UDFRunner(udfs).run_for_dataset_async(
                dataset=dataset,
                executor=self._async_executor,
                cancel_id="none"
            )

            async for udf_results in udfres_iter:
                for plot in plots:
                    plot.new_data(udf_results)
            for plot in plots:
                plot.new_data(udf_results, force=True)

    def _run_sync(self, dataset, udfs, plots):
        with self._camera_setup(dataset, udfs):
            udf_results = UDFRunner(udfs).run_for_dataset(
                dataset=dataset,
                executor=self._executor,
            )
            for plot in plots:
                plot.new_data(udf_results)
            return udf_results
