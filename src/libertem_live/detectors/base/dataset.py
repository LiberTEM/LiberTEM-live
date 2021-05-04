from contextlib import contextmanager
import logging

from libertem.common import Shape
from libertem.io.dataset.base import DataSet, TilingScheme

logger = logging.getLogger(__name__)


class LiveDataSet(DataSet):
    def __init__(self, setup):
        self._setup = setup

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
