from contextlib import contextmanager
import logging

from libertem.common import Shape
from libertem.io.dataset.base import TilingScheme

logger = logging.getLogger(__name__)


class AcquisitionMixin:
    def __init__(self, trigger, *args, **kwargs):
        self._trigger = trigger
        super().__init__(*args, **kwargs)

    @contextmanager
    def acquire(self):
        raise NotImplementedError

    def trigger(self):
        if self._trigger is not None:
            self._trigger()


def bench_noop(ds, data_source):
    ts = TilingScheme.make_for_shape(
        tileshape=Shape((10, 256, 256), sig_dims=2),
        dataset_shape=ds.shape
    )
    with data_source:
        for p in ds.get_partitions():
            for t in p.get_tiles(tiling_scheme=ts):
                pass
