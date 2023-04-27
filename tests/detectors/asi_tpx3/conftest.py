import os

import pytest


from utils import get_testdata_path, run_camera_sim


TPX3_TESTDATA_PATH = os.path.join(
    get_testdata_path(),
    'asi-tpx3', 'header_data_with_padding_vals32bits.scr',
)
HAVE_TPX3_TESTDATA = os.path.exists(TPX3_TESTDATA_PATH)


def run_tpx_sim(*args, path=TPX3_TESTDATA_PATH, **kwargs):
    if not HAVE_TPX3_TESTDATA:
        pytest.skip("need ASI TPX3 testdata")
    from libertem_live.detectors.asi_tpx3.sim import TpxCameraSim
    return run_camera_sim(
        cls=TpxCameraSim,
        paths=[path],
        port=0,
        sleep=0.1,
        cached='MEM',
        **kwargs,
    )


@pytest.fixture(scope='module')
def tpx_runner():
    yield from run_tpx_sim()


@pytest.fixture(scope='module')
def tpx_sim(tpx_runner):
    return tpx_runner.server_t.port


@pytest.fixture
def tpx_testdata_path():
    return TPX3_TESTDATA_PATH


@pytest.fixture(scope='module')
def tpx_runner_mock():
    from libertem_live.detectors.asi_tpx3.sim import TpxCameraSim
    mock_sim = run_camera_sim(
        cls=TpxCameraSim,
        mock_nav_shape=(32, 32),
        port=0,
        sleep=0.1,
        cached='MEM',
    )
    yield from mock_sim


@pytest.fixture(scope='module')
def tpx_sim_mock(tpx_runner_mock):
    return tpx_runner_mock.server_t.port
