import functools
import os
import platform
import time
import concurrent.futures
import socket
import threading
from contextlib import contextmanager

import pytest
import numpy as np
from numpy.testing import assert_allclose

from libertem.udf.sum import SumUDF
from libertem.udf.sumsigudf import SumSigUDF
from libertem.contrib.daskadapter import make_dask_array
from libertem.io.dataset.base import TilingScheme
from libertem.common import Shape

from libertem_live.detectors.merlin import MerlinDataSource, MerlinControl
from libertem_live.detectors.merlin.sim import (
        CameraSim, ServerThreadMixin, StopException, TriggerClient, UndeadException
)

from utils import get_testdata_path


MIB_TESTDATA_PATH = os.path.join(get_testdata_path(), 'default.mib')
HAVE_MIB_TESTDATA = os.path.exists(MIB_TESTDATA_PATH)

PTYCHO_TESTDATA_PATH = os.path.join(get_testdata_path(), '20200518 165148', 'default.hdr')
HAVE_PTYCHO_TESTDATA = os.path.exists(PTYCHO_TESTDATA_PATH)

pytestmark = [
    pytest.mark.skipif(not HAVE_MIB_TESTDATA, reason="need .mib testdata"),
    pytest.mark.data,
]


def run_camera_sim(*args, **kwargs):
    server = CameraSim(
        *args, host='127.0.0.1', data_port=0, control_port=0, trigger_port=0, **kwargs
    )
    server.start()
    server.wait_for_listen()
    yield server
    print("cleaning up server thread")
    server.maybe_raise()
    print("stopping server thread")
    try:
        server.stop()
    except UndeadException:
        raise RuntimeError("Server didn't stop gracefully")


@pytest.fixture(scope='module')
def merlin_detector_sim_threads():
    '''
    Untriggered default simulator.
    '''
    yield from run_camera_sim(
        path=MIB_TESTDATA_PATH, nav_shape=(32, 32),
    )


@pytest.fixture(scope='module')
def merlin_detector_sim(merlin_detector_sim_threads):
    '''
    Host, port tuple of the untriggered default simulator
    '''
    return merlin_detector_sim_threads.server_t.sockname


@pytest.fixture(scope='module')
def merlin_detector_cached_threads():
    '''
    Untriggered default simulator with memory cache.
    '''
    yield from run_camera_sim(
        path=MIB_TESTDATA_PATH, nav_shape=(32, 32),
        cached='MEM'
    )


@pytest.fixture(scope='module')
def merlin_detector_cached(merlin_detector_cached_threads):
    '''
    Host, port tuple of the untriggered default simulator with memory cache
    '''
    return merlin_detector_cached_threads.server_t.sockname


@pytest.fixture(scope='module')
def merlin_detector_memfd_threads():
    '''
    Untriggered default simulator with memfd cache.
    '''
    yield from run_camera_sim(
        path=MIB_TESTDATA_PATH, nav_shape=(32, 32),
        cached='MEMFD'
    )


@pytest.fixture(scope='module')
def merlin_detector_memfd(merlin_detector_memfd_threads):
    '''
    Host, port tuple of the untriggered default simulator with memfd cache
    '''
    return merlin_detector_memfd_threads.server_t.sockname


@pytest.fixture(scope='module')
def merlin_triggered_garbage_threads():
    '''
    Triggered simulator with garbage.
    '''
    yield from run_camera_sim(
        path=MIB_TESTDATA_PATH, nav_shape=(32, 32),
        cached='MEMFD',
        wait_trigger=True, garbage=True,
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
def merlin_ds(ltl_ctx):
    return ltl_ctx.load('MIB', path=MIB_TESTDATA_PATH, nav_shape=(32, 32))


@pytest.fixture
def merlin_ds_ptycho(ltl_ctx):
    return ltl_ctx.load(
        'MIB', path=PTYCHO_TESTDATA_PATH, nav_shape=(128, 128)
    )


@pytest.fixture(scope='module')
def merlin_detector_sim_threads_ptycho():
    '''
    Untriggered default simulator.
    '''
    yield from run_camera_sim(
        path=PTYCHO_TESTDATA_PATH, nav_shape=(128, 128),
    )


@pytest.fixture(scope='module')
def merlin_detector_sim_ptycho(merlin_detector_sim_threads_ptycho):
    '''
    Host, port tuple of the untriggered default simulator
    '''
    return merlin_detector_sim_threads_ptycho.server_t.sockname


@pytest.mark.with_numba  # Get coverage for decoders
def test_acquisition(ltl_ctx, merlin_detector_sim, merlin_ds):
    triggered = triggered = np.array((False,))

    def trigger(acquisition):
        triggered[:] = True
        assert acquisition.shape.nav == merlin_ds.shape.nav

    host, port = merlin_detector_sim
    aq = ltl_ctx.prepare_acquisition(
        'merlin',
        trigger=trigger,
        nav_shape=(32, 32),
        host=host,
        port=port,
        drain=False,
        pool_size=16,
    )
    udf = SumUDF()

    res = ltl_ctx.run_udf(dataset=aq, udf=udf)
    ref = ltl_ctx.run_udf(dataset=merlin_ds, udf=udf)

    assert_allclose(res['intensity'], ref['intensity'])


def test_acquisition_dry_run(ltl_ctx, merlin_detector_sim, merlin_ds):
    triggered = triggered = np.array((False,))

    def trigger(acquisition):
        triggered[:] = True
        assert acquisition.shape.nav == merlin_ds.shape.nav

    host, port = merlin_detector_sim
    aq = ltl_ctx.prepare_acquisition(
        'merlin',
        trigger=trigger,
        nav_shape=(32, 32),
        host=host,
        port=port,
        drain=False,
        pool_size=16,
    )
    udf = SumUDF()

    runner_cls = ltl_ctx.executor.get_udf_runner()
    runner_cls.dry_run([udf], aq, None)
    ltl_ctx.run_udf(dataset=aq, udf=udf)


def test_acquisition_iter(ltl_ctx, merlin_detector_sim, merlin_ds):
    triggered = triggered = np.array((False,))

    def trigger(acquisition):
        triggered[:] = True
        assert acquisition.shape.nav == merlin_ds.shape.nav

    host, port = merlin_detector_sim
    aq = ltl_ctx.prepare_acquisition(
        'merlin',
        trigger=trigger,
        nav_shape=(32, 32),
        host=host,
        port=port,
        drain=False,
        pool_size=16,
    )
    udf = SumUDF()

    for res in ltl_ctx.run_udf_iter(dataset=aq, udf=udf, sync=True):
        pass

    ref = ltl_ctx.run_udf(dataset=merlin_ds, udf=udf)

    assert_allclose(res.buffers[0]['intensity'], ref['intensity'])


@pytest.mark.asyncio
async def test_acquisition_async(ltl_ctx, merlin_detector_sim, merlin_ds):
    triggered = triggered = np.array((False,))

    def trigger(acquisition):
        triggered[:] = True
        assert acquisition.shape.nav == merlin_ds.shape.nav

    host, port = merlin_detector_sim
    aq = ltl_ctx.prepare_acquisition(
        'merlin',
        trigger=trigger,
        nav_shape=(32, 32),
        host=host,
        port=port,
        drain=False,
        pool_size=16,
    )
    udf = SumUDF()

    res = await ltl_ctx.run_udf(dataset=aq, udf=udf, sync=False)
    ref = ltl_ctx.run_udf(dataset=merlin_ds, udf=udf)

    assert_allclose(res['intensity'], ref['intensity'])

    async for res in ltl_ctx.run_udf_iter(dataset=aq, udf=udf, sync=False):
        pass

    assert_allclose(res.buffers[0]['intensity'], ref['intensity'])


def test_get_tiles_comparison(ltl_ctx, merlin_detector_sim_ptycho, merlin_ds_ptycho):
    merlin_ds = merlin_ds_ptycho
    da, _ = make_dask_array(merlin_ds)
    host, port = merlin_detector_sim_ptycho
    aq = ltl_ctx.prepare_acquisition(
        'merlin',
        trigger=None,
        nav_shape=merlin_ds.shape.nav,
        host=host,
        port=port,
        drain=False,
        pool_size=1,
    )
    s = TilingScheme.make_for_shape(
        tileshape=Shape((7, 256, 256), sig_dims=2),
        dataset_shape=aq.shape
    )

    with ltl_ctx._do_acquisition(aq, None):
        for p in aq.get_partitions():
            part_data = da.reshape((-1, 256, 256), limit='512MiB')[p.slice.get()].compute()
            print(f"comparing partition {p}")
            for tile in p.get_tiles(s):
                print(f"comparing tile {tile.tile_slice} in partition {p.slice}")
                tile_data = part_data[tile.tile_slice.shift(p.slice).get()]
                assert np.allclose(tile, tile_data)


@pytest.mark.parametrize(
    # Test matching and mismatching shape
    'sig_shape', ((256, 256), (512, 512))
)
def test_acquisition_shape(ltl_ctx, merlin_detector_sim, merlin_ds, sig_shape):
    triggered = triggered = np.array((False,))

    def trigger(acquisition):
        triggered[:] = True
        assert acquisition.shape.nav == merlin_ds.shape.nav

    host, port = merlin_detector_sim
    aq = ltl_ctx.prepare_acquisition(
        'merlin',
        trigger=trigger,
        nav_shape=(32, 32),
        sig_shape=sig_shape,
        host=host,
        port=port,
        drain=False,
        pool_size=16,
    )

    try:
        udf = SumUDF()

        res = ltl_ctx.run_udf(dataset=aq, udf=udf)
        ref = ltl_ctx.run_udf(dataset=merlin_ds, udf=udf)

        assert_allclose(res['intensity'], ref['intensity'])
        assert sig_shape == tuple(merlin_ds.shape.sig)
    except ValueError as e:
        assert sig_shape != tuple(merlin_ds.shape.sig)
        assert 'received "image_size" header' in e.args[0]


def test_acquisition_cached(ltl_ctx, merlin_detector_cached, merlin_ds):
    triggered = triggered = np.array((False,))

    def trigger(acquisition):
        triggered[:] = True
        assert acquisition.shape.nav == merlin_ds.shape.nav

    host, port = merlin_detector_cached
    aq = ltl_ctx.prepare_acquisition(
        'merlin',
        trigger=trigger,
        nav_shape=(32, 32),
        host=host,
        port=port,
        drain=False,
        pool_size=16,
    )
    udf = SumUDF()

    res = ltl_ctx.run_udf(dataset=aq, udf=udf)
    ref = ltl_ctx.run_udf(dataset=merlin_ds, udf=udf)

    assert_allclose(res['intensity'], ref['intensity'])


@pytest.mark.skipif(platform.system() != 'Linux',
                    reason="MemFD is Linux-only")
def test_acquisition_memfd(ltl_ctx, merlin_detector_memfd, merlin_ds):
    triggered = triggered = np.array((False,))

    def trigger(acquisition):
        triggered[:] = True
        assert acquisition.shape.nav == merlin_ds.shape.nav

    host, port = merlin_detector_memfd
    aq = ltl_ctx.prepare_acquisition(
        'merlin',
        trigger=trigger,
        nav_shape=(32, 32),
        host=host,
        port=port,
        drain=False,
        pool_size=16,
    )
    udf = SumUDF()

    res = ltl_ctx.run_udf(dataset=aq, udf=udf)
    ref = ltl_ctx.run_udf(dataset=merlin_ds, udf=udf)

    assert_allclose(res['intensity'], ref['intensity'])


def test_acquisition_triggered_garbage(
        ltl_ctx, merlin_control_sim, trigger_sim, garbage_sim, merlin_ds):
    sim_host, sim_port = garbage_sim

    pool = concurrent.futures.ThreadPoolExecutor(1)

    trig_res = {
        0: None
    }

    def trigger(acquisition):
        control = MerlinControl(*merlin_control_sim)
        with control:
            control.cmd('STARTACQUISITION')
        tr = TriggerClient(*trigger_sim)
        print("Trigger connection:", trigger_sim)
        tr.connect()
        tr.trigger()

        def do_scan():
            '''
            Emulated blocking scan function using the Merlin simulator
            '''
            print("do_scan()")

        fut = pool.submit(do_scan)
        trig_res[0] = fut
        tr.close()

    aq = ltl_ctx.prepare_acquisition(
        'merlin',
        trigger=trigger,
        nav_shape=(32, 32),
        host=sim_host,
        port=sim_port,
        pool_size=16,
    )
    udf = SumUDF()

    res = ltl_ctx.run_udf(dataset=aq, udf=udf)
    assert trig_res[0].result() is None

    ref = ltl_ctx.run_udf(dataset=merlin_ds, udf=udf)

    assert_allclose(res['intensity'], ref['intensity'])


def test_acquisition_triggered_control(ltl_ctx, merlin_control_sim, garbage_sim, merlin_ds):
    sim_host, sim_port = garbage_sim

    pool = concurrent.futures.ThreadPoolExecutor(1)
    trig_res = {
        0: None
    }

    def trigger(acquisition):
        control = MerlinControl(*merlin_control_sim)
        with control:
            control.cmd('STARTACQUISITION')

        def do_scan():
            '''
            Emulated blocking scan function using the Merlin simulator
            '''
            print("do_scan()")
            with control:
                control.cmd('SOFTTRIGGER')

        fut = pool.submit(do_scan)
        trig_res[0] = fut

    aq = ltl_ctx.prepare_acquisition(
        'merlin',
        trigger=trigger,
        nav_shape=(32, 32),
        host=sim_host,
        port=sim_port,
        pool_size=16,
    )
    udf = SumUDF()

    res = ltl_ctx.run_udf(dataset=aq, udf=udf)
    assert trig_res[0].result() is None

    ref = ltl_ctx.run_udf(dataset=merlin_ds, udf=udf)

    assert_allclose(res['intensity'], ref['intensity'])


@pytest.mark.parametrize(
    'inline', (True, False),
)
@pytest.mark.parametrize(
    # auto, correct, wrong
    'sig_shape', (None, (256, 256), (512, 512)),
)
def test_datasource(ltl_ctx, merlin_detector_sim, merlin_ds, inline, sig_shape):
    print("Merlin sim:", merlin_detector_sim)
    source = MerlinDataSource(*merlin_detector_sim, sig_shape=sig_shape, pool_size=16)

    res = np.zeros(merlin_ds.shape.sig)
    try:
        with source:
            if inline:
                for chunk in source.inline_stream():
                    res += chunk.sum(axis=0)
            else:
                for chunk in source.stream(num_frames=32*32):
                    res += chunk.buf.sum(axis=0)
        udf = SumUDF()
        ref = ltl_ctx.run_udf(dataset=merlin_ds, udf=udf)
        assert_allclose(res, ref['intensity'])
        assert (sig_shape is None) or (sig_shape == tuple(merlin_ds.shape.sig))
    except ValueError as e:
        assert sig_shape != tuple(merlin_ds.shape.sig)
        assert 'received "image_size" header' in e.args[0]


def test_datasource_nav(ltl_ctx, merlin_detector_sim, merlin_ds):
    source = MerlinDataSource(*merlin_detector_sim, pool_size=16)

    res = np.zeros(merlin_ds.shape.nav).reshape((-1,))
    with source:
        for chunk in source.stream(num_frames=merlin_ds.shape.nav.size):
            res[chunk.start:chunk.stop] = chunk.buf.sum(axis=(-1, -2))
    udf = SumSigUDF()
    ref = ltl_ctx.run_udf(dataset=merlin_ds, udf=udf)
    assert_allclose(res.reshape(merlin_ds.shape.nav), ref['intensity'])


class BadServer(ServerThreadMixin, threading.Thread):
    def __init__(self, exception, *args, **kwargs):
        self.exception = exception
        super().__init__(*args, **kwargs)

    def handle_conn(self, connection):
        raise self.exception


class OtherError(Exception):
    pass


def serve(cls, host='127.0.0.1', port=0):
    server = cls(host=host, port=port)
    server.start()
    server.wait_for_listen()
    yield server
    print("cleaning up server thread")
    server.maybe_raise()
    print("stopping server thread")
    server.stop()
    timeout = 2
    start = time.time()
    while True:
        print("are we there yet?")
        server.maybe_raise()
        if not server.is_alive():
            print("server is dead, we are there")
            break
        if (time.time() - start) >= timeout:
            raise RuntimeError("Server didn't stop gracefully")
        time.sleep(0.1)


@pytest.mark.parametrize(
    'exception_cls', (RuntimeError, ValueError, OtherError)
)
def test_server_throws(exception_cls):
    server = contextmanager(serve)
    exception = exception_cls("Testing...")
    cls = functools.partial(BadServer, exception=exception, name="BadServer")
    with pytest.raises(exception_cls, match="Testing..."):
        with server(cls) as serv:
            host, port = serv.sockname
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((host, port))
                time.sleep(1)
                print("second try...")
                # Making sure the server is stopped
                with pytest.raises(ConnectionRefusedError):
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s2:
                        s2.connect((host, port))


def test_server_stop():
    server = contextmanager(serve)
    exception = StopException("Testing...")
    cls = functools.partial(BadServer, exception=exception, name="BadServer")
    with server(cls) as serv:
        host, port = serv.sockname
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
            time.sleep(1)
            # The above exception should have led to an immediate graceful stop of the server
            with pytest.raises(ConnectionRefusedError):
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s2:
                    s2.connect((host, port))
                    print(s2.getsockname())


def test_control(merlin_control_sim, tmp_path):
    path = tmp_path / 'cmd.txt'
    with path.open('w') as f:
        f.write("SET,COUNTERDEPTH,12\n")
        f.write("\n")
        f.write("SET,RUNHEADLESS,1\n")
    c = MerlinControl(*merlin_control_sim)
    with c:
        c.set("NUMFRAMESTOACQUIRE", 23)
        c.cmd('STARTACQUISITION')
        assert c.get("NUMFRAMESTOACQUIRE") == b'23'
        c.send_command_file(path)
        assert c.get("COUNTERDEPTH") == b'12'
