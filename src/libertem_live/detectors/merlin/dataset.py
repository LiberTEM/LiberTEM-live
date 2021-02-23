import logging
import numpy as np
from libertem.common import Shape, Slice
from libertem.io.dataset.base import (
    DataSet, DataTile, DataSetMeta, BasePartition, Partition,
)

from .data import TCPBackend
from .parser import MIBParser

logger = logging.getLogger(__name__)


class LiveDataSet(DataSet):
    def __init__(self, scan_size, backend: TCPBackend, parser: MIBParser, frames_per_partition=256):
        self._backend = backend
        self._parser = parser
        self._scan_size = scan_size
        self._frames_per_partition = frames_per_partition

    def initialize(self, executor):
        if not self._backend.is_connected():
            self._backend.connect()

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

    def check_valid(self):
        pass

    def get_msg_converter(self):
        raise NotImplementedError()

    def get_cache_key(self):
        raise NotImplementedError()

    def wait_for_acquisition(self):
        # FIXME: warmup for the right dtype?
        self._parser.warmup()

        logger.info("waiting for acquisition header")

        header = self._backend.read_acquisition_header()
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
        num_frames = np.prod(self._scan_size)
        num_partitions = num_frames // self._frames_per_partition

        header = self._backend.get_acquisition_header()

        slices = BasePartition.make_slices(self.shape, num_partitions)
        for part_slice, start, stop in slices:
            yield LivePartition(
                start_idx=start,
                end_idx=stop,
                meta=self._meta,
                partition_slice=part_slice,
                backend=self._backend,
                parser=self._parser,
                acq_header=header,
            )


class LivePartition(Partition):
    def __init__(
        self, start_idx, end_idx, partition_slice,
        backend, meta, acq_header, parser,
    ):
        super().__init__(meta=meta, io_backend=None)
        self.slice = partition_slice
        self._start_idx = start_idx
        self._end_idx = end_idx
        self._backend = backend
        self._parser = parser
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
        return False  # FIXME

    def adjust_tileshape(self, tileshape):
        # FIXME
        # return Shape((64, 256, 256), sig_dims=2)
        return tileshape

    def get_base_shape(self):
        return Shape((1, 256, 256), sig_dims=2)

    def get_tiles(self, tiling_scheme, dest_dtype="float32", roi=None):
        assert len(tiling_scheme) == 1
        assert np.dtype(dest_dtype) == np.dtype(self._parser._read_dtype)
        if tiling_scheme.depth == 1:
            # FIXME:
            tile_shape = Shape(
                (1,) + tuple(tiling_scheme[0].shape), sig_dims=2)
            parser = self._parser

            for i in range(self.shape[0]):
                tile_slice = Slice(
                    origin=(self._start_idx + i,) + (0, 0),
                    shape=tile_shape,
                )
                frame = parser.get_frame()
                yield DataTile(
                    frame.reshape(tile_shape),
                    tile_slice=tile_slice,
                    scheme_idx=0,
                )
        else:
            parser = self._parser
            to_read = self._end_idx - self._start_idx
            for i in range(0, self.shape[0], tiling_scheme.depth):
                frames_in_tile = min(to_read, tiling_scheme.depth)
                tile_shape = Shape(
                    (frames_in_tile,) + tuple(tiling_scheme[0].shape),
                    sig_dims=2
                )
                buf = np.empty(tile_shape, dtype=dest_dtype)
                tile_slice = Slice(
                    origin=(self._start_idx + i,) + (0, 0),
                    shape=tile_shape,
                )
                # print("reading %d frames from parser" % frames_in_tile)
                parser.get_many(num_frames=frames_in_tile, out=buf)
                yield DataTile(
                    buf,
                    tile_slice=tile_slice,
                    scheme_idx=0,
                )
                to_read -= frames_in_tile
