import os
import platform

import pytest

from utils import get_testdata_path, run_camera_sim
from libertem_live.detectors.merlin.sim import (
    CameraSim
)
from libertem_live.api import LiveContext

MIB_TESTDATA_PATH = os.path.join(get_testdata_path(), 'default.mib')
PTYCHO_TESTDATA_PATH = os.path.join(get_testdata_path(), '20200518 165148', 'default.hdr')


def run_merlin_sim(*args, **kwargs):
    return run_camera_sim(
        *args,
        cls=CameraSim,
        host='127.0.0.1',
        data_port=0,
        control_port=0,
        trigger_port=0,
        **kwargs
    )


@pytest.fixture(scope='module')
def merlin_detector_sim_threads():
    '''
    Untriggered default simulator.
    '''
    yield from run_merlin_sim(
        path=MIB_TESTDATA_PATH,
        nav_shape=(32, 32),
        initial_params={
            'IMAGEX': "256",
            'IMAGEY': "256",
            'COUNTERDEPTH': '12',
        },
    )


@pytest.fixture(scope='module')
def merlin_detector_sim(merlin_detector_sim_threads):
    '''
    Host, port tuple of the untriggered default simulator
    '''
    return merlin_detector_sim_threads.server_t.sockname


@pytest.fixture(scope='module')
def merlin_triggered_garbage_threads():
    '''
    Triggered simulator with garbage.
    '''
    cached = None
    if platform.system() == 'Linux':
        cached = 'MEMFD'
    yield from run_merlin_sim(
        path=MIB_TESTDATA_PATH, nav_shape=(32, 32),
        cached=cached,
        wait_trigger=True, garbage=True,
        initial_params={
            'IMAGEX': "256",
            'IMAGEY': "256",
            'COUNTERDEPTH': '12',
        },
    )


@pytest.fixture(scope='module')
def merlin_control_sim(merlin_triggered_garbage_threads):
    '''
    Host, port tuple of the control port for the triggered simulator
    '''
    return merlin_triggered_garbage_threads.control_t.sockname


@pytest.fixture(scope='module')
def trigger_sim(merlin_triggered_garbage_threads):
    '''
    Host, port tuple of the trigger port for the triggered simulator
    '''
    return merlin_triggered_garbage_threads.trigger_t.sockname


@pytest.fixture(scope='module')
def garbage_sim(merlin_triggered_garbage_threads):
    '''
    Host, port tuple of the data port for the triggered simulator
    '''
    return merlin_triggered_garbage_threads.server_t.sockname


@pytest.fixture
def merlin_ds(ctx_pipelined):
    return ctx_pipelined.load('MIB', path=MIB_TESTDATA_PATH, nav_shape=(32, 32))


@pytest.fixture
def merlin_ds_ptycho_flat(ltl_ctx):
    return ltl_ctx.load(
        'MIB',
        path=PTYCHO_TESTDATA_PATH,
        nav_shape=(128*128,)
    )


@pytest.fixture(scope='module')
def merlin_detector_sim_threads_ptycho():
    '''
    Untriggered default simulator.
    '''
    yield from run_merlin_sim(
        path=PTYCHO_TESTDATA_PATH,
        nav_shape=(128, 128),
        initial_params={
            'IMAGEX': "256",
            'IMAGEY': "256",
            'COUNTERDEPTH': '6',
        },
    )


@pytest.fixture(scope='module')
def merlin_detector_sim_ptycho(merlin_detector_sim_threads_ptycho):
    '''
    Host, port tuple of the untriggered default simulator
    (with alternate data set)
    '''
    return merlin_detector_sim_threads_ptycho.server_t.sockname


@pytest.fixture(scope='module')
def merlin_control_sim_ptycho(merlin_detector_sim_threads_ptycho):
    '''
    Host, port tuple of the untriggered default simulator control channel
    (with alternate data set)
    '''
    return merlin_detector_sim_threads_ptycho.control_t.sockname


@pytest.fixture(scope='function')
def default_conn(
    ctx_pipelined: LiveContext,
    merlin_detector_sim,
    merlin_control_sim,
):
    host, port = merlin_detector_sim
    api_host, api_port = merlin_control_sim
    with ctx_pipelined.make_connection('merlin').open(
        data_host=host,
        data_port=port,
        api_host=api_host,
        api_port=api_port,
        drain=False,
    ) as conn:
        yield conn


@pytest.fixture(scope='module')
def merlin_detector_cached_threads():
    '''
    Untriggered default simulator with memory cache.
    '''
    yield from run_merlin_sim(
        path=MIB_TESTDATA_PATH, nav_shape=(32, 32),
        cached='MEM'
    )


@pytest.fixture(scope='module')
def merlin_detector_cached(merlin_detector_cached_threads):
    '''
    Host, port tuple of the untriggered default simulator with memory cache
    '''
    return merlin_detector_cached_threads.server_t.sockname


@pytest.mark.skipif(platform.system() != 'Linux',
                    reason="MemFD is Linux-only")
@pytest.fixture(scope='module')
def merlin_detector_memfd_threads():
    '''
    Untriggered default simulator with memfd cache.
    '''
    yield from run_merlin_sim(
        path=MIB_TESTDATA_PATH, nav_shape=(32, 32),
        cached='MEMFD'
    )


@pytest.mark.skipif(platform.system() != 'Linux',
                    reason="MemFD is Linux-only")
@pytest.fixture(scope='module')
def merlin_detector_memfd(merlin_detector_memfd_threads):
    '''
    Host, port tuple of the untriggered default simulator with memfd cache
    '''
    return merlin_detector_memfd_threads.server_t.sockname
