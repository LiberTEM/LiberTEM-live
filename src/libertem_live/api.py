from libertem.executor.inline import InlineJobExecutor
from libertem.executor.base import AsyncAdapter
from libertem.udf.base import UDFRunner


class LiveContext:
    def __init__(self, camera_setup):
        self._executor = InlineJobExecutor()
        self._async_executor = AsyncAdapter(self._executor)
        self._camera_setup = camera_setup

    def run(self, dataset, udfs, plots=None, sync=False):
        """
        """
        if plots is None:
            plots = []
        if sync:
            return self._run_sync(dataset, udfs, plots)
        else:
            return self._run_async(dataset, udfs, plots)

    async def run_udfs(self, dataset, udfs, plots=None):
        """
        """
        from libertem_live.viz.mpl import MPLLivePlot
        from libertem.udf.base import UDFMeta

        runner = UDFRunner(udfs)

        meta = UDFMeta(
            partition_shape=None,
            dataset_shape=dataset.shape,
            roi=None,
            dataset_dtype=dataset.dtype,
            input_dtype=runner._get_dtype(
                dataset.dtype,
                corrections=None,
            ),
            corrections=None,
        )
        for udf in udfs:
            udf.set_meta(meta)

        # prepare plots:
        plot_instances = []
        if plots is None:
            # default: plot all sig/nav results without extra_shape
            plots = [
                [
                    k
                    for k, buf in udf.get_result_buffers().items()
                    if buf.kind in ('sig', 'nav') and buf.extra_shape == ()
                ]
                for udf in udfs
            ]
        for idx, (udf, channels) in enumerate(zip(udfs, plots)):
            buffers = udf.get_result_buffers()
            for channel in channels:
                buf = buffers[channel]
                p0 = MPLLivePlot(
                    dataset,
                    kind=buf.kind,
                    channel=channel,
                    udf_index=idx,
                    min_delta=0.3
                )
                plot_instances.append(p0)

        # actually run stuff:
        await self._run_async(
            dataset, udfs, plot_instances,
        )

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
