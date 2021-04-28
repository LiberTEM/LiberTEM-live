import uuid

import numpy as np
import cloudpickle

from libertem.udf.base import (
    UDFMeta, UDFResults, UDFTask
)
from libertem.executor.inline import InlineJobExecutor
from libertem.io.dataset.base.tiling import Negotiator
from libertem.io.dataset.base import Partition, DataSet
from libertem.common import Slice, Shape
from libertem.common.buffers import BufferWrapper
from libertem.common.backend import (
    get_device_class, get_use_cuda
)
from libertem.utils.async_utils import async_generator_eager


class UDFRunner:
    def __init__(self, udfs, debug=False):
        self._udfs = udfs
        self._debug = debug

    @classmethod
    def inspect_udf(cls, udf, dataset, roi=None):
        """
        Return result buffer declarations for a given UDF/DataSet/roi combination
        """
        runner = UDFRunner([udf])
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

        udf = udf.copy()
        udf.set_meta(meta)
        buffers = udf.get_result_buffers()
        for buf in buffers.values():
            buf.set_shape_ds(dataset.shape, roi)
        return buffers

    @classmethod
    def dry_run(cls, udfs, dataset, roi=None):
        """
        Return result buffers for a given UDF/DataSet/roi combination
        exactly as running the UDFs would, just skipping execution and
        merging of the processing tasks.

        This can be used to create an empty result to initialize live plots
        before running an UDF.
        """
        runner = UDFRunner(udfs)
        executor = InlineJobExecutor()
        res = runner.run_for_dataset(
            dataset=dataset,
            executor=executor,
            roi=roi,
            dry=True
        )
        return res

    def _get_dtype(self, dtype, corrections):
        if corrections is not None and corrections.have_corrections():
            tmp_dtype = np.result_type(np.float32, dtype)
        else:
            tmp_dtype = dtype
        for udf in self._udfs:
            tmp_dtype = np.result_type(
                udf.get_preferred_input_dtype(),
                tmp_dtype
            )
        return tmp_dtype

    def _init_udfs(self, numpy_udfs, cupy_udfs, partition, roi, corrections, device_class, env):
        dtype = self._get_dtype(partition.dtype, corrections)
        meta = UDFMeta(
            partition_shape=partition.slice.adjust_for_roi(roi).shape,
            dataset_shape=partition.meta.shape,
            roi=roi,
            dataset_dtype=partition.dtype,
            input_dtype=dtype,
            tiling_scheme=None,
            corrections=corrections,
            device_class=device_class,
            threads_per_worker=env.threads_per_worker,
        )
        for udf in numpy_udfs:
            if device_class == 'cuda':
                udf.set_backend('cuda')
            else:
                udf.set_backend('numpy')
        if device_class == 'cpu':
            assert not cupy_udfs
        for udf in cupy_udfs:
            udf.set_backend('cupy')
        udfs = numpy_udfs + cupy_udfs
        for udf in udfs:
            udf.set_meta(meta)
            udf.init_result_buffers()
            udf.allocate_for_part(partition, roi)
            udf.init_task_data()
            # TODO: preprocess doesn't have access to the tiling scheme - is this ok?
            if hasattr(udf, 'preprocess'):
                udf.clear_views()
                udf.preprocess()
        neg = Negotiator()
        # FIXME take compute backend into consideration as well
        # Other boundary conditions when moving input data to device
        tiling_scheme = neg.get_scheme(
            udfs=udfs,
            partition=partition,
            read_dtype=dtype,
            roi=roi,
            corrections=corrections,
        )

        # print(tiling_scheme)

        # FIXME: don't fully re-create?
        meta = UDFMeta(
            partition_shape=partition.slice.adjust_for_roi(roi).shape,
            dataset_shape=partition.meta.shape,
            roi=roi,
            dataset_dtype=partition.dtype,
            input_dtype=dtype,
            tiling_scheme=tiling_scheme,
            corrections=corrections,
            device_class=device_class,
            threads_per_worker=env.threads_per_worker,
        )
        for udf in udfs:
            udf.set_meta(meta)
        return (meta, tiling_scheme, dtype)

    def _run_tile(self, udfs, partition, tile, device_tile):
        for udf in udfs:
            method = udf.get_method()
            if method == 'tile':
                udf.set_contiguous_views_for_tile(partition, tile)
                udf.set_slice(tile.tile_slice)
                udf.process_tile(device_tile)
            elif method == 'frame':
                tile_slice = tile.tile_slice
                for frame_idx, frame in enumerate(device_tile):
                    frame_slice = Slice(
                        origin=(tile_slice.origin[0] + frame_idx,) + tile_slice.origin[1:],
                        shape=Shape((1,) + tuple(tile_slice.shape)[1:],
                                    sig_dims=tile_slice.shape.sig.dims),
                    )
                    udf.set_slice(frame_slice)
                    udf.set_views_for_frame(partition, tile, frame_idx)
                    udf.process_frame(frame)
            elif method == 'partition':
                udf.set_views_for_tile(partition, tile)
                udf.set_slice(partition.slice)
                udf.process_partition(device_tile)

    def _run_udfs(self, numpy_udfs, cupy_udfs, partition, tiling_scheme, roi, dtype):
        # FIXME pass information on target location (numpy or cupy)
        # to dataset so that is can already move it there.
        # In the future, it might even decode data on the device instead of CPU
        tiles = partition.get_tiles(
            tiling_scheme=tiling_scheme,
            roi=roi, dest_dtype=dtype,
        )

        if cupy_udfs:
            xp = cupy_udfs[0].xp

        for tile in tiles:
            self._run_tile(numpy_udfs, partition, tile, tile)
            if cupy_udfs:
                # Work-around, should come from dataset later
                device_tile = xp.asanyarray(tile)
                self._run_tile(cupy_udfs, partition, tile, device_tile)

    def _wrapup_udfs(self, numpy_udfs, cupy_udfs, partition):
        udfs = numpy_udfs + cupy_udfs
        for udf in udfs:
            udf.flush(self._debug)
            if hasattr(udf, 'postprocess'):
                udf.clear_views()
                udf.postprocess()

            udf.cleanup()
            udf.clear_views()
            udf.export_results()

        if self._debug:
            try:
                cloudpickle.loads(cloudpickle.dumps(partition))
            except TypeError:
                raise TypeError("could not pickle partition")
            try:
                cloudpickle.loads(cloudpickle.dumps(
                    [u._do_get_results() for u in udfs]
                ))
            except TypeError:
                raise TypeError("could not pickle results")

    def _udf_lists(self, device_class):
        numpy_udfs = []
        cupy_udfs = []
        if device_class == 'cuda':
            for udf in self._udfs:
                backends = udf.get_backends()
                if 'cuda' in backends:
                    numpy_udfs.append(udf)
                elif 'cupy' in backends:
                    cupy_udfs.append(udf)
                else:
                    raise ValueError(f"UDF backends are {backends}, supported on CUDA are "
                            "'cuda' and 'cupy'")
        elif device_class == 'cpu':
            assert all('numpy' in udf.get_backends() for udf in self._udfs)
            numpy_udfs = self._udfs
        else:
            raise ValueError(f"Unknown device class {device_class}, "
                "supported are 'cpu' and 'cuda'")
        return (numpy_udfs, cupy_udfs)

    def run_for_partition(self, partition: Partition, roi, corrections, env):
        with env.enter():
            try:
                previous_id = None
                device_class = get_device_class()
                # numpy_udfs and cupy_udfs contain references to the objects in
                # self._udfs
                numpy_udfs, cupy_udfs = self._udf_lists(device_class)
                # Will only be populated if actually on CUDA worker
                # and any UDF supports 'cupy' (and not 'cuda')
                if cupy_udfs:
                    # Avoid importing if not used
                    import cupy
                    device = get_use_cuda()
                    previous_id = cupy.cuda.Device().id
                    cupy.cuda.Device(device).use()
                (meta, tiling_scheme, dtype) = self._init_udfs(
                    numpy_udfs, cupy_udfs, partition, roi, corrections, device_class, env,
                )
                # print("UDF TilingScheme: %r" % tiling_scheme.shape)
                partition.set_corrections(corrections)
                self._run_udfs(numpy_udfs, cupy_udfs, partition, tiling_scheme, roi, dtype)
                self._wrapup_udfs(numpy_udfs, cupy_udfs, partition)
            finally:
                if previous_id is not None:
                    cupy.cuda.Device(previous_id).use()
            # Make sure results are in the same order as the UDFs
            return tuple(udf.results for udf in self._udfs)

    def _debug_task_pickling(self, tasks):
        if self._debug:
            cloudpickle.loads(cloudpickle.dumps(tasks))

    def _check_preconditions(self, dataset: DataSet, roi):
        if roi is not None and np.product(roi.shape) != np.product(dataset.shape.nav):
            raise ValueError(
                "roi: incompatible shapes: %s (roi) vs %s (dataset)" % (
                    roi.shape, dataset.shape.nav
                )
            )

    def _prepare_run_for_dataset(
        self, dataset: DataSet, executor, roi, corrections, backends, dry
    ):
        self._check_preconditions(dataset, roi)
        meta = UDFMeta(
            partition_shape=None,
            dataset_shape=dataset.shape,
            roi=roi,
            dataset_dtype=dataset.dtype,
            input_dtype=self._get_dtype(dataset.dtype, corrections),
            corrections=corrections,
        )
        for udf in self._udfs:
            udf.set_meta(meta)
            udf.init_result_buffers()
            udf.allocate_for_full(dataset, roi)

            if hasattr(udf, 'preprocess'):
                udf.set_views_for_dataset(dataset)
                udf.preprocess()
        if dry:
            tasks = []
        else:
            tasks = list(self._make_udf_tasks(dataset, roi, corrections, backends))
        return tasks

    def run_for_dataset(self, dataset: DataSet, executor,
                        roi=None, progress=False, corrections=None, backends=None, dry=False):
        for res in self.run_for_dataset_sync(
            dataset=dataset,
            executor=executor.ensure_sync(),
            roi=roi,
            progress=progress,
            corrections=corrections,
            backends=backends,
            dry=dry
        ):
            pass
        return res

    def run_for_dataset_sync(self, dataset: DataSet, executor,
                        roi=None, progress=False, corrections=None, backends=None, dry=False):
        tasks = self._prepare_run_for_dataset(
            dataset, executor, roi, corrections, backends, dry
        )
        cancel_id = str(uuid.uuid4())
        self._debug_task_pickling(tasks)

        if progress:
            from tqdm import tqdm
            t = tqdm(total=len(tasks))

        executor = executor.ensure_sync()

        damage = BufferWrapper(kind='nav', dtype=bool)
        damage.set_shape_ds(dataset.shape, roi)
        damage.allocate()
        if tasks:
            for part_results, task in executor.run_tasks(tasks, cancel_id):
                if progress:
                    t.update(1)
                for results, udf in zip(part_results, self._udfs):
                    udf.set_views_for_partition(task.partition)
                    udf.merge(
                        dest=udf.results.get_proxy(),
                        src=results.get_proxy()
                    )
                    udf.clear_views()
                v = damage.get_view_for_partition(task.partition)
                v[:] = True
                yield UDFResults(
                    buffers=tuple(
                        udf._do_get_results()
                        for udf in self._udfs
                    ),
                    damage=damage
                )
        else:
            # yield at least one result (which should be empty):
            for udf in self._udfs:
                udf.clear_views()
            yield UDFResults(
                buffers=tuple(
                    udf._do_get_results()
                    for udf in self._udfs
                ),
                damage=damage
            )

        if progress:
            t.close()

    async def run_for_dataset_async(
        self, dataset: DataSet, executor, cancel_id, roi=None, corrections=None, backends=None,
        progress=False, dry=False
    ):
        gen = self.run_for_dataset_sync(
            dataset=dataset,
            executor=executor.ensure_sync(),
            roi=roi,
            progress=progress,
            corrections=corrections,
            backends=backends,
            dry=dry
        )

        async for res in async_generator_eager(gen):
            yield res

    def _roi_for_partition(self, roi, partition: Partition):
        return roi.reshape(-1)[partition.slice.get(nav_only=True)]

    def _make_udf_tasks(self, dataset: DataSet, roi, corrections, backends):
        for idx, partition in enumerate(dataset.get_partitions()):
            if roi is not None:
                roi_for_part = self._roi_for_partition(roi, partition)
                if np.count_nonzero(roi_for_part) == 0:
                    # roi is empty for this partition, ignore
                    continue
            udfs = [
                udf.copy_for_partition(partition, roi)
                for udf in self._udfs
            ]
            yield UDFTask(
                partition=partition, idx=idx, udfs=udfs, roi=roi, corrections=corrections,
                backends=backends,
            )
