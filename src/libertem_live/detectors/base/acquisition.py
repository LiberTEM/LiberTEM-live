from contextlib import contextmanager
import logging

import numpy as np

from libertem.common import Shape, Slice
from libertem.common.executor import WorkerContext
from libertem.io.dataset.base import DataTile, Partition, TilingScheme


logger = logging.getLogger(__name__)


class AcquisitionTimeout(Exception):
    pass


class AcquisitionMixin:
    def __init__(self, trigger, *args, **kwargs):
        self._trigger = trigger
        super().__init__(*args, **kwargs)

    @contextmanager
    def acquire(self):
        raise NotImplementedError

    def trigger(self):
        if self._trigger is not None:
            self._trigger(self)


class FullframeLivePartition(Partition):
    def __init__(
        self, start_idx, end_idx, get_frames, partition_slice,
        meta,
    ):
        super().__init__(meta=meta, partition_slice=partition_slice, io_backend=None, decoder=None)
        self._start_idx = start_idx
        self._end_idx = end_idx
        self._get_frames = get_frames

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

    def set_worker_context(self, worker_context: WorkerContext):
        self._worker_context = worker_context

    def _preprocess(self, tile_data, tile_slice):
        if self._corrections is None:
            return
        self._corrections.apply(tile_data, tile_slice)

    def _get_tiles_fullframe(self, tiling_scheme: TilingScheme, dest_dtype="float32", roi=None):
        assert len(tiling_scheme) == 1
        logger.debug("reading up to frame idx %d for this partition", self._end_idx)
        to_read = self._end_idx - self._start_idx
        depth = tiling_scheme.depth
        buf = np.zeros((depth,) + tiling_scheme[0].shape, dtype=dest_dtype)
        buf_idx = 0
        tile_start = self._start_idx
        frames = self._get_frames(self._worker_context.get_worker_queue())
        while to_read > 0:
            # 1) put frame into tile buffer (including dtype conversion if needed)
            assert buf_idx < depth,\
                    f"buf_idx should be in bounds of buf! ({buf_idx} < ({depth} == {buf.shape[0]}))"
            try:
                frame = next(frames)
                buf[buf_idx] = frame
                buf_idx += 1
                to_read -= 1

                # if buf is full, or the partition is done, yield the tile
                tile_done = buf_idx == depth
                partition_done = to_read == 0
            except StopIteration:
                assert to_read == 0, f"we were still expecting to read {to_read} frames more!"
                tile_done = True
                partition_done = True

            if tile_done or partition_done:
                frames_in_tile = buf_idx
                tile_buf = buf[:frames_in_tile]
                if tile_buf.shape[0] == 0:
                    assert to_read == 0
                    continue  # we are done and the buffer is empty

                tile_shape = Shape(
                    (frames_in_tile,) + tuple(tiling_scheme[0].shape),
                    sig_dims=2
                )
                tile_slice = Slice(
                    origin=(tile_start,) + (0, 0),
                    shape=tile_shape,
                )
                # print(f"yielding tile for {tile_slice}")
                self._preprocess(tile_buf, tile_slice)
                yield DataTile(
                    tile_buf,
                    tile_slice=tile_slice,
                    scheme_idx=0,
                )
                tile_start += frames_in_tile
                buf_idx = 0
        logger.debug("LivePartition.get_tiles: end of method")

    def get_tiles(self, tiling_scheme, dest_dtype="float32", roi=None):
        yield from self._get_tiles_fullframe(tiling_scheme, dest_dtype, roi)

    def __repr__(self):
        return f"<FullframeLivePartition {self._start_idx}:{self._end_idx}>"
