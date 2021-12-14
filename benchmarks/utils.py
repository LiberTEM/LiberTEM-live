from libertem.common import Shape
from libertem.io.dataset.base import TilingScheme


def bench_noop(ds, data_source):
    ts = TilingScheme.make_for_shape(
        tileshape=Shape((10, 256, 256), sig_dims=2),
        dataset_shape=ds.shape
    )
    with data_source:
        for p in ds.get_partitions():
            for t in p.get_tiles(tiling_scheme=ts):
                pass
