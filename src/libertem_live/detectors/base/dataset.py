from contextlib import contextmanager
import logging

from libertem.common import Shape
from libertem.io.dataset.base import TilingScheme

logger = logging.getLogger(__name__)


class LiveDataSetMixin:
    def __init__(self, setup, *args, **kwargs):
        self._setup = setup
        super().__init__(*args, **kwargs)

    @contextmanager
    def run_setup(self, udfs):
        if self._setup is not None:
            with self._setup(self, udfs):
                yield
        else:
            yield

    @contextmanager
    def start_control(self):
        raise NotImplementedError

    @contextmanager
    def start_acquisition(self):
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
