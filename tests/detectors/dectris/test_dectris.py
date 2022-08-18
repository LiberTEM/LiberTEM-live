import os
import sys
from typing import NamedTuple, Tuple
import numpy as np

from libertem.udf.sumsigudf import SumSigUDF
from libertem.udf.sum import SumUDF
from libertem.common import Shape
from libertem.executor.pipelined import PipelinedExecutor
import pytest
from libertem_live.api import LiveContext
from libertem_live.detectors.dectris.acquisition import DectrisAcquisition
from libertem_live.detectors.dectris.mock import OfflineAcquisition
import libertem

from utils import get_testdata_path, run_camera_sim


DECTRIS_TESTDATA_PATH = os.path.join(
    get_testdata_path(),
    'dectris', 'zmqdump.dat.128x128-id34-exte-bslz4'
)
HAVE_DECTRIS_TESTDATA = os.path.exists(DECTRIS_TESTDATA_PATH)

pytestmark = pytest.mark.skipif(sys.version_info < (3, 7),
                                reason="DECTRIS support requires Python 3.7")


def run_dectris_sim(*args, path=DECTRIS_TESTDATA_PATH, **kwargs):
    from libertem_live.detectors.dectris.sim import DectrisSim
    return run_camera_sim(cls=DectrisSim, path=path, port=0, zmqport=0, **kwargs)


@pytest.fixture(scope='module')
def dectris_runner():
    yield from run_dectris_sim()


@pytest.fixture(scope='module')
def dectris_sim(dectris_runner):
    '''
    port, zmqport tuple of the simulator
    '''
    return (dectris_runner.port, dectris_runner.zmqport)


class MockRawFrame(NamedTuple):
    shape: Tuple
    dtype: np.dtype
    encoding: str
    data: np.ndarray


_item_counter = 0


@pytest.fixture()
def skipped_dectris_runner():

    item_counter = {0: 0}

    def skip(data):
        result = None
        # print(data[:10])
        if item_counter[0] not in (10, 11, 12, 13):
            result = data
        item_counter[0] += 1
        return result

    yield from run_dectris_sim(data_filter=skip)


@pytest.fixture()
def skipped_dectris_sim(skipped_dectris_runner):
    '''
    port, zmqport tuple of the simulator
    '''
    return (skipped_dectris_runner.port, skipped_dectris_runner.zmqport)


def test_udf_sig_mock(ctx_pipelined):
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
    aq.initialize(ctx_pipelined.executor)

    udf = SumUDF()
    res = ctx_pipelined.run_udf(dataset=aq, udf=udf)

    assert np.allclose(
        res['intensity'].data,
        data.astype(np.float32).sum(axis=0),
    )


def test_udf_nav_mock(ctx_pipelined):
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
    aq.initialize(ctx_pipelined.executor)

    udf = SumSigUDF()
    res = ctx_pipelined.run_udf(dataset=aq, udf=udf)

    assert np.allclose(
        res['intensity'].data,
        data.sum(axis=(1, 2)),
    )


@pytest.mark.skipif(not HAVE_DECTRIS_TESTDATA, reason="need DECTRIS testdata")
@pytest.mark.data
def test_udf_sig(ctx_pipelined, dectris_sim):
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
    aq.initialize(ctx_pipelined.executor)

    udf = SumUDF()
    ctx_pipelined.run_udf(dataset=aq, udf=udf)


@pytest.mark.skipif(not HAVE_DECTRIS_TESTDATA, reason="need DECTRIS testdata")
@pytest.mark.data
def test_udf_nav(ctx_pipelined, dectris_sim):
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
    aq.initialize(ctx_pipelined.executor)
    udf = SumSigUDF()
    ctx_pipelined.run_udf(dataset=aq, udf=udf)


@pytest.mark.skipif(not HAVE_DECTRIS_TESTDATA, reason="need DECTRIS testdata")
@pytest.mark.data
def test_sum(ctx_pipelined, dectris_sim):
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
    aq.initialize(ctx_pipelined.executor)
    # FIXME verify result
    _ = ctx_pipelined.run_udf(dataset=aq, udf=SumUDF())


@pytest.mark.skipif(not HAVE_DECTRIS_TESTDATA, reason="need DECTRIS testdata")
@pytest.mark.data
@pytest.mark.timeout(120)  # May lock up because of executor bug
def test_frame_skip(skipped_dectris_sim, dectris_sim):
    # Only newer 0.10.0 because of a bug in the PipelinedExecutor
    _version_bits = libertem.__version__.split('.')
    version_tuple = tuple(int(b) for b in _version_bits[:3])
    if version_tuple <= (0, 10, 0):
        pytest.skip(reason="LiberTEM version too old")

    # uses its own executor to not potentially bring
    # the `ctx_pipelined` executor into a bad state
    try:
        executor = PipelinedExecutor(
            spec=PipelinedExecutor.make_spec(cpus=range(4), cudas=[]),
            # to prevent issues in already-pinned situations (i.e. containerized
            # environments), don't pin our worker processes in testing:
            pin_workers=False,
            cleanup_timeout=0.5,
        )
        api_port, data_port = skipped_dectris_sim
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
        ctx = LiveContext(executor=executor)
        aq.initialize(ctx.executor)
        # Originally an AssertionError, but may cause downstream issues
        # in the executor, TODO revisit after some time if executor behavior changed
        with pytest.raises(Exception):
            _ = ctx.run_udf(dataset=aq, udf=SumUDF())
        # Ensure the executor is still alive
        api_port, data_port = dectris_sim
        aq2 = DectrisAcquisition(
            nav_shape=(128, 128),
            trigger=lambda x: None,
            frames_per_partition=32,
            api_host='127.0.0.1',
            api_port=api_port,
            data_host='127.0.0.1',
            data_port=data_port,
            trigger_mode='exte',
        )
        aq2.initialize(ctx.executor)
        _ = ctx.run_udf(dataset=aq2, udf=SumUDF())
    finally:
        if executor is not None:
            executor.close()
