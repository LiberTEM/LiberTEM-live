import functools
import os
import time
import concurrent.futures
import socket
import threading
from contextlib import contextmanager

import pytest
import numpy as np

from libertem.udf.sum import SumUDF

from libertem_live.detectors.merlin import MerlinDataSource, MerlinControl
from libertem_live.detectors.merlin.sim import (
        DataSocketServer, ControlSocketServer,
        DataSocketSimulator, CachedDataSocketSim, MemfdSocketSim,
        ServerThreadMixin, StopException, TriggerSocketServer, TriggerClient
)

from utils import get_testdata_path


MIB_TESTDATA_PATH = os.path.join(get_testdata_path(), 'default.mib')
HAVE_MIB_TESTDATA = os.path.exists(MIB_TESTDATA_PATH)

pytestmark = [
    pytest.mark.skipif(not HAVE_MIB_TESTDATA, reason="need .mib testdata"),
    pytest.mark.data,
]


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


@pytest.fixture(scope='module')
def merlin_detector_sim_thread():
    sim = DataSocketSimulator(path=MIB_TESTDATA_PATH, nav_shape=(32, 32))
    cls = functools.partial(DataSocketServer, sim=sim)
    yield from serve(cls)


@pytest.fixture(scope='module')
def merlin_detector_sim(merlin_detector_sim_thread):
    return merlin_detector_sim_thread.sockname


@pytest.fixture(scope='module')
def merlin_detector_cached_thread():
    sim = CachedDataSocketSim(path=MIB_TESTDATA_PATH, nav_shape=(32, 32))
    cls = functools.partial(DataSocketServer, sim=sim)
    yield from serve(cls)


@pytest.fixture(scope='module')
def merlin_detector_cached(merlin_detector_cached_thread):
    return merlin_detector_cached_thread.sockname


@pytest.fixture(scope='module')
def merlin_detector_memfd_thread():
    sim = MemfdSocketSim(path=MIB_TESTDATA_PATH, nav_shape=(32, 32))
    cls = functools.partial(DataSocketServer, sim=sim)
    yield from serve(cls)


@pytest.fixture(scope='module')
def merlin_detector_memfd(merlin_detector_memfd_thread):
    return merlin_detector_memfd_thread.sockname


@pytest.fixture(scope='module')
def merlin_control_sim_thread(merlin_detector_sim_garbage_thread):
    control = functools.partial(
        ControlSocketServer,
        trigger_event=merlin_detector_sim_garbage_thread.trigger_event
    )
    yield from serve(control)


@pytest.fixture(scope='module')
def merlin_control_sim(merlin_control_sim_thread):
    return merlin_control_sim_thread.sockname


@pytest.fixture(scope='module')
def merlin_detector_sim_garbage_thread():
    sim = DataSocketSimulator(path=MIB_TESTDATA_PATH, nav_shape=(32, 32))
    cls = functools.partial(DataSocketServer, sim=sim, garbage=True, wait_trigger=True)
    yield from serve(cls)


@pytest.fixture(scope='module')
def triggered_sim_thread(merlin_detector_sim_garbage_thread):
    sim = merlin_detector_sim_garbage_thread
    cls = functools.partial(
        TriggerSocketServer,
        trigger_event=sim.trigger_event,
        finish_event=sim.finish_event
    )
    yield from serve(cls)


@pytest.fixture(scope='module')
def trigger_sim(triggered_sim_thread):
    return triggered_sim_thread.sockname


@pytest.fixture(scope='module')
def garbage_sim(merlin_detector_sim_garbage_thread):
    return merlin_detector_sim_garbage_thread.sockname


@pytest.fixture
def merlin_ds(ltl_ctx):
    return ltl_ctx.load('MIB', path=MIB_TESTDATA_PATH, nav_shape=(32, 32))


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
        drain=False
    )
    udf = SumUDF()

    res = ltl_ctx.run_udf(dataset=aq, udf=udf)
    ref = ltl_ctx.run_udf(dataset=merlin_ds, udf=udf)

    assert np.allclose(res['intensity'], ref['intensity'])


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
        drain=False
    )
    udf = SumUDF()

    res = ltl_ctx.run_udf(dataset=aq, udf=udf)
    ref = ltl_ctx.run_udf(dataset=merlin_ds, udf=udf)

    assert np.allclose(res['intensity'], ref['intensity'])


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
        drain=False
    )
    udf = SumUDF()

    res = ltl_ctx.run_udf(dataset=aq, udf=udf)
    ref = ltl_ctx.run_udf(dataset=merlin_ds, udf=udf)

    assert np.allclose(res['intensity'], ref['intensity'])


def test_acquisition_triggered_garbage(ltl_ctx, trigger_sim, garbage_sim, merlin_ds):
    sim_host, sim_port = garbage_sim

    pool = concurrent.futures.ThreadPoolExecutor(1)

    trig_res = {
        0: None
    }

    def trigger(acquisition):
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
        port=sim_port
    )
    udf = SumUDF()

    res = ltl_ctx.run_udf(dataset=aq, udf=udf)
    assert trig_res[0].result() is None

    ref = ltl_ctx.run_udf(dataset=merlin_ds, udf=udf)

    assert np.allclose(res['intensity'], ref['intensity'])


def test_acquisition_triggered_control(ltl_ctx, merlin_control_sim, garbage_sim, merlin_ds):
    sim_host, sim_port = garbage_sim

    pool = concurrent.futures.ThreadPoolExecutor(1)
    trig_res = {
        0: None
    }

    def trigger(acquisition):
        control = MerlinControl(*merlin_control_sim)

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
        port=sim_port
    )
    udf = SumUDF()

    res = ltl_ctx.run_udf(dataset=aq, udf=udf)
    assert trig_res[0].result() is None

    ref = ltl_ctx.run_udf(dataset=merlin_ds, udf=udf)

    assert np.allclose(res['intensity'], ref['intensity'])


@pytest.mark.parametrize(
    'inline', (True, False)
)
def test_datasource(ltl_ctx, merlin_detector_sim, merlin_ds, inline):
    print("Merlin sim:", merlin_detector_sim)
    source = MerlinDataSource(*merlin_detector_sim, sig_shape=tuple(merlin_ds.shape.sig))

    res = np.zeros(merlin_ds.shape.sig)

    with source:
        if inline:
            for chunk in source.inline_stream():
                res += chunk.sum(axis=0)
        else:
            for chunk in source.stream(num_frames=32*32):
                res += chunk.buf.sum(axis=0)

    udf = SumUDF()
    ref = ltl_ctx.run_udf(dataset=merlin_ds, udf=udf)
    assert np.allclose(res, ref['intensity'])


class BadServer(ServerThreadMixin, threading.Thread):
    def __init__(self, exception, *args, **kwargs):
        self.exception = exception
        super().__init__(*args, **kwargs)

    def handle_conn(self, connection):
        raise self.exception


class OtherError(Exception):
    pass


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
