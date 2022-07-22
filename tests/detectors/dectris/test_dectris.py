import os
from contextlib import contextmanager
import numpy as np

from libertem.udf.sumsigudf import SumSigUDF
from libertem.udf.sum import SumUDF
from libertem.common import Shape
from libertem.io.dataset.base import TilingScheme
import pytest
from libertem_live.detectors.dectris.acquisition import (
    AcquisitionParams, DectrisAcquisition, DetectorConfig, Receiver
)
from libertem_live.detectors.dectris.sim import DectrisSim

from utils import get_testdata_path, run_camera_sim


DECTRIS_TESTDATA_PATH = os.path.join(
    get_testdata_path(),
    'dectris', 'zmqdump.dat.128x128-id34-exte-bslz4'
)
HAVE_DECTRIS_TESTDATA = os.path.exists(DECTRIS_TESTDATA_PATH)


def run_dectris_sim(*args, path=DECTRIS_TESTDATA_PATH, **kwargs):
    return run_camera_sim(cls=DectrisSim, path=path, port=0, zmqport=0)


@pytest.fixture(scope='module')
def dectris_runner():
    yield from run_dectris_sim()


@pytest.fixture(scope='module')
def dectris_sim(dectris_runner):
    '''
    port, zmqport tuple of the simulator
    '''
    return (dectris_runner.port, dectris_runner.zmqport)


class OfflineReceiver(Receiver):
    """
    Mock Receiver that reads from a numpy array
    """
    def __init__(self, data):
        self.data = data
        assert len(data.shape) == 3
        self._idx = 0

    def __next__(self) -> np.ndarray:
        if self._idx == self.data.shape[0]:
            raise StopIteration
        data = self.data[self._idx]
        self._idx += 1
        return data


class OfflineAcquisition(DectrisAcquisition):
    def __init__(self, mock_data, *args, **kwargs):
        self.data = mock_data
        super().__init__(*args, **kwargs)

    def connect(self):
        pass  # NOOP

    def get_receiver(self):
        return OfflineReceiver(data=self.data)

    def get_api_client(self):
        return None  # should not make API calls in testing!

    def get_detector_config(self) -> DetectorConfig:
        shape_y = 512
        shape_x = 512
        bit_depth = 8
        return DetectorConfig(
            x_pixels_in_detector=shape_x, y_pixels_in_detector=shape_y, bit_depth=bit_depth
        )

    @contextmanager
    def acquire(self):
        try:
            self._acq_state = AcquisitionParams(
                sequence_id=42,
                nimages=128,
                trigger_mode=self._trigger_mode
            )
            self.trigger()  # <-- this triggers, either via API or via HW trigger
            yield
        finally:
            self._acq_state = None


def test_dry_run(ltl_ctx):
    dataset_shape = Shape((128, 512, 512), sig_dims=2)
    data = np.random.randn(*dataset_shape).astype(np.uint8)
    aq = OfflineAcquisition(
        nav_shape=tuple(dataset_shape.nav),
        mock_data=data,
        frames_per_partition=42,  # chosen not to evenly divide `dataset_shape.nav`
        api_host=None,
        api_port=None,
        data_host=None,
        data_port=None,
        trigger_mode="exte",
    )
    aq.initialize(ltl_ctx.executor)
    tileshape = Shape((7, 512, 512), sig_dims=2)
    ts = TilingScheme.make_for_shape(tileshape, dataset_shape)

    for p in aq.get_partitions():
        print(p)
        for tile in p.get_tiles(tiling_scheme=ts):
            assert np.allclose(
                data[tile.tile_slice.get()],
                tile
            )


def test_udf_sig(ltl_ctx):
    dataset_shape = Shape((128, 512, 512), sig_dims=2)
    data = np.random.randn(*dataset_shape).astype(np.uint8)
    aq = OfflineAcquisition(
        nav_shape=tuple(dataset_shape.nav),
        mock_data=data,
        frames_per_partition=42,  # chosen not to evenly divide `dataset_shape.nav`
        api_host=None,
        api_port=None,
        data_host=None,
        data_port=None,
        trigger_mode="exte",
    )
    aq.initialize(ltl_ctx.executor)

    udf = SumUDF()
    res = ltl_ctx.run_udf(dataset=aq, udf=udf)

    assert np.allclose(
        res['intensity'].data,
        data.sum(axis=0),
    )


def test_udf_nav(ltl_ctx):
    dataset_shape = Shape((128, 512, 512), sig_dims=2)
    data = np.random.randn(*dataset_shape).astype(np.uint8)
    aq = OfflineAcquisition(
        nav_shape=tuple(dataset_shape.nav),
        mock_data=data,
        frames_per_partition=42,  # chosen not to evenly divide `dataset_shape.nav`
        api_host=None,
        api_port=None,
        data_host=None,
        data_port=None,
        trigger_mode="exte",
    )
    aq.initialize(ltl_ctx.executor)

    udf = SumSigUDF()
    res = ltl_ctx.run_udf(dataset=aq, udf=udf)

    assert np.allclose(
        res['intensity'].data,
        data.sum(axis=(1, 2)),
    )


@pytest.mark.data
def test_sum(ltl_ctx, dectris_sim):
    api_port, data_port = dectris_sim
    aq = DectrisAcquisition(
        nav_shape=(128, 128),
        trigger=lambda x: None,
        frames_per_partition=32,
        api_host='127.0.0.1',
        api_port=api_port,
        data_host='127.0.0.1',
        data_port=data_port,
        trigger_mode='exte',
    )
    aq.initialize(ltl_ctx.executor)
    # FIXME verify result
    _ = ltl_ctx.run_udf(dataset=aq, udf=SumUDF())
