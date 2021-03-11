import logging
import numpy as np
from libertem.common import Shape, Slice
from libertem.io.dataset.base import (
    DataSet, DataTile, DataSetMeta, BasePartition, Partition,
)

from .data import MerlinDataSocket, ReaderPool

logger = logging.getLogger(__name__)


class LiveDataSet(DataSet):
    def __init__(
        self, scan_size,
        data_socket: MerlinDataSocket,
        pool: ReaderPool,
        frames_per_partition=256,
    ):
        self._data_socket = data_socket
        self._pool = pool
        self._scan_size = scan_size
        self._frames_per_partition = frames_per_partition

    def initialize(self, executor):
        # FIXME: possibly need to have an "acquisition plan" object
        # so we know all relevant parameters beforehand
        dtype = np.uint8  # FIXME: don't know the dtype yet
        self._meta = DataSetMeta(
            shape=Shape(self._scan_size + (256, 256), sig_dims=2),
            raw_dtype=dtype,
            dtype=dtype,
        )
        return self

    @property
    def dtype(self):
        return self._meta.dtype

    @property
    def raw_dtype(self):
        return self._meta.raw_dtype

    @property
    def shape(self):
        return self._meta.shape

    @property
    def meta(self):
        return self._meta

    def check_valid(self):
        pass

    def get_msg_converter(self):
        raise NotImplementedError()

    def get_cache_key(self):
        raise NotImplementedError()

    def wait_for_acquisition(self):
        logger.info("waiting for acquisition header")
        header = self._data_socket.get_acquisition_header()

        bit_depth = int(header['Counter Depth (number)'])
        if bit_depth in (1, 6):
            dtype = np.uint8
        elif bit_depth in (12,):
            dtype = np.uint16
        else:  # 24 bit?
            dtype = np.uint32
        self._meta = DataSetMeta(
            shape=Shape(self._scan_size + (256, 256), sig_dims=2),
            raw_dtype=dtype,
            dtype=dtype,
        )
        return header

    def get_partitions(self):
        # FIXME: only works for inline executor or similar, as we pass a
        # TCP connection to each partition, which cannot be serialized
        num_frames = np.prod(self._scan_size, dtype=np.uint64)
        num_partitions = int(num_frames // self._frames_per_partition)

        header = self._data_socket.get_acquisition_header()

        slices = BasePartition.make_slices(self.shape, num_partitions)
        for part_slice, start, stop in slices:
            yield LivePartition(
                start_idx=start,
                end_idx=stop,
                meta=self._meta,
                partition_slice=part_slice,
                data_socket=self._data_socket,
                pool=self._pool,
                acq_header=header,
            )


class LivePartition(Partition):
    def __init__(
        self, start_idx, end_idx, partition_slice,
        data_socket, meta, acq_header, pool,
    ):
        super().__init__(meta=meta, partition_slice=partition_slice, io_backend=None)
        self._start_idx = start_idx
        self._end_idx = end_idx
        self._data_socket = data_socket
        self._pool = pool
        self._acq_header = acq_header

    def shape_for_roi(self, roi):
        return self.slice.adjust_for_roi(roi).shape

    @property
    def shape(self):
        return self.slice.shape

    @property
    def dtype(self):
        return self.meta.raw_dtype

    def set_corrections(self, corrections):
        self._corrections = corrections

    def need_decode(self, read_dtype, roi, corrections):
        return True  # FIXME: we just do this to get a large tile size

    def adjust_tileshape(self, tileshape, roi):
        return (11, 256, 256)
        # return Shape((self._end_idx - self._start_idx, 256, 256), sig_dims=2)

    def get_max_io_size(self):
        return 12*256*256*8

    def get_base_shape(self, roi):
        return (1, 1, 256)

    def _get_tiles_fullframe(self, tiling_scheme, dest_dtype="float32", roi=None):
        # assert len(tiling_scheme) == 1
        pool = self._pool.get_impl(
            read_upto_frame=self._end_idx,
            chunk_size=11,
            # chunk_size=tiling_scheme.depth,
        )
        to_read = self._end_idx - self._start_idx
        with pool:
            while to_read > 0:
                with pool.get_result() as res_wrapped:
                    frames_in_tile = res_wrapped.stop - res_wrapped.start
                    tile_shape = Shape(
                        (frames_in_tile,) + tuple(tiling_scheme[0].shape),
                        sig_dims=2
                    )
                    tile_slice = Slice(
                        origin=(res_wrapped.start,) + (0, 0),
                        shape=tile_shape,
                    )
                    yield DataTile(
                        res_wrapped.buf,
                        tile_slice=tile_slice,
                        scheme_idx=0,
                    )
                    to_read -= frames_in_tile

    def get_tiles(self, tiling_scheme, dest_dtype="float32", roi=None):
        yield from self._get_tiles_fullframe(tiling_scheme, dest_dtype, roi)
        return
        # assert len(tiling_scheme) == 1
        print(tiling_scheme)
        pool = self._pool.get_impl(
            read_upto_frame=self._end_idx,
            chunk_size=tiling_scheme.depth,
        )
        to_read = int(self._end_idx - self._start_idx)
        nav_slices_raw = [
            (...,) + slice_.get(sig_only=True)
            for idx, slice_ in tiling_scheme.slices
        ]
        with pool:
            while to_read > 0:
                with pool.get_result() as res_wrapped:
                    frames_in_tile = res_wrapped.stop - res_wrapped.start
                    for (idx, slice_), nav_slice_raw in zip(tiling_scheme.slices, nav_slices_raw):
                        tile_shape = Shape(
                            (frames_in_tile,) + tuple(slice_.shape),
                            sig_dims=2
                        )
                        tile_slice = Slice(
                            origin=(res_wrapped.start,) + tuple(slice_.origin),
                            shape=tile_shape,
                        )
                        sliced_res = res_wrapped.buf[nav_slice_raw]
                        yield DataTile(sliced_res, tile_slice=tile_slice, scheme_idx=idx)
                    to_read -= frames_in_tile
