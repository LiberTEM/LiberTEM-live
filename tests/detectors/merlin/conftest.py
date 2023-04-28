import os
import platform

import pytest
import pathlib
import numpy as np

from utils import get_testdata_path, run_camera_sim
import libertem.api as lt
from libertem.common.shape import Shape
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
def merlin_triggered_sim_threads():
    '''
    Triggered non-garbage simulator.
    '''
    yield from run_merlin_sim(
        path=MIB_TESTDATA_PATH,
        nav_shape=(32, 32),
        wait_trigger=True,
        initial_params={
            'IMAGEX': "256",
            'IMAGEY': "256",
            'COUNTERDEPTH': '12',
        },
    )


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
    merlin_control_sim,  # XXX not really the matching control, but it kinda works...
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


@pytest.fixture(scope='function')
def conn_triggered(
    ctx_pipelined: LiveContext,
    merlin_triggered_sim_threads,
):
    host, port = merlin_triggered_sim_threads.server_t.sockname
    api_host, api_port = merlin_triggered_sim_threads.control_t.sockname
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


def get_header_string(num_frames, counter_depth):
    return fr"""HDR,
Counter Depth (number):	{counter_depth}
Frames in Acquisition (Number):	{num_frames}
Frames per Trigger (Number):	{num_frames}
End	                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               """  # noqa: W291,E501


def get_frame_header_encoded(sig_size, counter_depth, header_size=None):
    if header_size is None:
        header_size = 0
    if header_size > 10000:
        raise ValueError('Cannot encode header with size more than 5 digits')
    if sig_size > 1000:
        raise ValueError('Cannot encode frame with size more than 4 digits')
    encoded = fr"""MQ1,000001,{header_size:0>5d},01,{sig_size:0>4d},{sig_size:0>4d},U08,   1x1,01,2020-05-18 16:51:49.971626,0.000555,0,0,0,1.200000E+2,5.110000E+2,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,3RX,175,511,000,000,000,000,000,000,125,255,125,125,100,100,082,100,087,030,128,004,255,129,128,176,168,511,511,MQ1A,2020-05-18T14:51:49.971626178Z,555000ns,{counter_depth}
""".encode("ascii")  # noqa
    if header_size == 0:
        return get_frame_header_encoded(sig_size, counter_depth, header_size=len(encoded))
    return encoded


@pytest.fixture(scope='module')
def mock_mib_ds(tmpdir_factory):
    datadir = tmpdir_factory.mktemp('mock_mib_dataset')
    datadir = pathlib.Path(datadir)

    shape = Shape((8, 8, 16, 16), sig_dims=2)
    sig_square = shape.sig.to_tuple()[0]
    counter_depth = 6
    header_string = get_header_string(shape.nav.size, counter_depth)
    frame_header = get_frame_header_encoded(sig_square, counter_depth)

    hdr_path = (datadir / 'default.hdr')
    with hdr_path.open('wb') as fp:
        fp.write(header_string.encode("ascii"))

    with (datadir / 'default.mib').open('wb') as fp:
        for idx in range(shape.nav.size):
            fp.write(frame_header)
            fp.write(np.full(shape.sig, idx, dtype=np.uint8).tobytes())
    return str(hdr_path), shape, counter_depth


@pytest.fixture(scope='module')
def mock_merlin_ds(mock_mib_ds):
    ctx = lt.Context.make_with('inline')
    hdr_path, shape, _ = mock_mib_ds
    return ctx.load(
        'mib',
        path=hdr_path,
        nav_shape=shape.nav,
    )


@pytest.fixture(scope='module')
def mock_merlin_sim_threads(mock_mib_ds):
    '''
    Untriggered non-garbage simulator.
    '''
    hdr_path, shape, counter_depth = mock_mib_ds
    sigy, sigx = shape.sig
    yield from run_merlin_sim(
        path=hdr_path,
        nav_shape=tuple(shape.nav),
        initial_params={
            'IMAGEX': f"{sigx}",
            'IMAGEY': f"{sigy}",
            'COUNTERDEPTH': f'{counter_depth}',
        },
    )


@pytest.fixture(scope='module')
def mock_merlin_detector_sim(mock_merlin_sim_threads):
    '''
    Host, port tuple of the untriggered default simulator
    '''
    return mock_merlin_sim_threads.server_t.sockname


@pytest.fixture(scope='module')
def mock_merlin_control_sim(mock_merlin_sim_threads):
    '''
    Host, port tuple of the control port for the triggered simulator
    '''
    return mock_merlin_sim_threads.control_t.sockname


@pytest.fixture(scope='function')
def mock_ds_conn(
    ctx_pipelined: LiveContext,
    mock_merlin_detector_sim,
    mock_merlin_control_sim,  # XXX not really the matching control, but it kinda works...
):
    host, port = mock_merlin_detector_sim
    api_host, api_port = mock_merlin_control_sim
    with ctx_pipelined.make_connection('merlin').open(
        data_host=host,
        data_port=port,
        api_host=api_host,
        api_port=api_port,
        drain=False,
    ) as conn:
        yield conn


@pytest.fixture(scope='module')
def mock_merlin_triggered_garbage(mock_mib_ds):
    '''
    Triggered simulator with garbage.
    '''
    hdr_path, shape, counter_depth = mock_mib_ds
    sigy, sigx = shape.sig
    cached = None
    try:
        import pymemfd  # noqa
        cached = 'MEMFD'
    except ImportError:
        if platform.system() == 'Linux':
            cached = 'MEM'
    yield from run_merlin_sim(
        path=hdr_path,
        cached=cached,
        wait_trigger=True,
        garbage=True,
        nav_shape=tuple(shape.nav),
        initial_params={
            'IMAGEX': f"{sigx}",
            'IMAGEY': f"{sigy}",
            'COUNTERDEPTH': f'{counter_depth}',
        },
    )


@pytest.fixture(scope='module')
def mock_merlin_detector_sim_garbage(mock_merlin_triggered_garbage):
    '''
    Host, port tuple of the untriggered default simulator
    '''
    return mock_merlin_triggered_garbage.server_t.sockname


@pytest.fixture(scope='module')
def mock_merlin_control_sim_garbage(mock_merlin_triggered_garbage):
    '''
    Host, port tuple of the control port for the triggered simulator
    '''
    return mock_merlin_triggered_garbage.control_t.sockname


@pytest.fixture(scope='module')
def mock_merlin_trigger_sim_garbage(mock_merlin_triggered_garbage):
    '''
    Host, port tuple of the control port for the triggered simulator
    '''
    return mock_merlin_triggered_garbage.trigger_t.sockname
