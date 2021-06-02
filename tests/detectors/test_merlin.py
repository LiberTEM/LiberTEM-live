import functools
import os
import time
from typing_extensions import runtime
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
        ServerThreadMixin, StopException
)

from utils import get_testdata_path


MIB_TESTDATA_PATH = os.path.join(get_testdata_path(), 'default.mib')
HAVE_MIB_TESTDATA = os.path.exists(MIB_TESTDATA_PATH)

pytestmark = pytest.mark.skipif(not HAVE_MIB_TESTDATA, reason="need .mib testdata")  # NOQA


def serve(cls, host='127.0.0.1', port=0):
    server = cls(host=host, port=port)
    server.start()
    server.listen_event.wait()
    yield server.sockname
    server.maybe_raise()
    server.stop()
    timeout = 2
    start = time.time()
    while True:
        server.maybe_raise()
        if not server.is_alive():
            break
        if (time.time() - start) >= timeout:
            raise RuntimeError("Server didn't stop gracefully")
        time.sleep(0.1)


@pytest.fixture(scope='module')
def merlin_detector_sim():
    sim = DataSocketSimulator(path=MIB_TESTDATA_PATH, nav_shape=(32, 32))
    cls = functools.partial(DataSocketServer, sim=sim)
    yield from serve(cls)

@pytest.fixture(scope='module')
def merlin_detector_cached():
    sim = CachedDataSocketSim(path=MIB_TESTDATA_PATH, nav_shape=(32, 32))
    cls = functools.partial(DataSocketServer, sim=sim)
    yield from serve(cls)

@pytest.fixture(scope='module')
def merlin_detector_memfd():
    sim = MemfdSocketSim(path=MIB_TESTDATA_PATH, nav_shape=(32, 32))
    cls = functools.partial(DataSocketServer, sim=sim)
    yield from serve(cls)


@pytest.fixture(scope='module')
def merlin_control_sim():
    yield from serve(ControlSocketServer)


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
    aq = ltl_ctx.prepare_acquisition('merlin', trigger=trigger, scan_size=(32, 32), host=host, port=port)
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
    aq = ltl_ctx.prepare_acquisition('merlin', trigger=trigger, scan_size=(32, 32), host=host, port=port)
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
    aq = ltl_ctx.prepare_acquisition('merlin', trigger=trigger, scan_size=(32, 32), host=host, port=port)
    udf = SumUDF()

    res = ltl_ctx.run_udf(dataset=aq, udf=udf)
    ref = ltl_ctx.run_udf(dataset=merlin_ds, udf=udf)

    assert np.allclose(res['intensity'], ref['intensity'])

@pytest.mark.parametrize(
    'inline', (False, True)
)
def test_datasource(ltl_ctx, merlin_detector_sim, merlin_ds, inline):
    source = MerlinDataSource(*merlin_detector_sim)

    res = np.zeros(merlin_ds.shape.sig)

    if inline:
        for chunk in source.inline_stream():
            res += chunk.sum(axis=0)
    else:
        with source:
            for chunk in source.stream():
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
        with server(cls) as sockname:
            host, port = sockname
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
    with server(cls) as sockname:
        host, port = sockname
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

