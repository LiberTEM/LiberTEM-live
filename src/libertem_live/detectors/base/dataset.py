from contextlib import contextmanager
import logging

from libertem.common import Shape
from libertem.io.dataset.base import TilingScheme

logger = logging.getLogger(__name__)


class LiveDataSetMixin:
    def __init__(self, on_enter, on_exit, *args, **kwargs):
        self._on_enter = on_enter
        self._on_exit = on_exit
        super().__init__(*args, **kwargs)

    @contextmanager
    def run_acquisition(self, meta):
        if self._on_enter is not None:
            self._on_enter(meta)
        with self.acquire():
            yield
        if self._on_exit is not None:
            self._on_exit(meta)

    @contextmanager
    def acquire(self):
        raise NotImplementedError


def bench_noop(ds, data_source):
    ts = TilingScheme.make_for_shape(
        tileshape=Shape((10, 256, 256), sig_dims=2),
        dataset_shape=ds.shape
    )
    with data_source:
        for p in ds.get_partitions():
            for t in p.get_tiles(tiling_scheme=ts):
                pass
