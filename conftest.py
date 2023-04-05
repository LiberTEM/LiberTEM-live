"""
Global test configuration. Use this file to define fixtures to use
in both doctests and regular tests.
"""
import os
import importlib
from contextlib import contextmanager

from libertem.executor.inline import InlineJobExecutor
from libertem.executor.pipelined import PipelinedExecutor

import pytest
import numpy as np

from libertem.viz.base import Dummy2DPlot

from libertem_live.detectors.dectris.sim import DectrisSim
from libertem_live import api as ltl
# A bit of gymnastics to import the test utilities since this
# conftest.py file is shared between the doctests and unit tests
# and this file is outside the package
basedir = os.path.dirname(__file__)
location = os.path.join(basedir, "tests/utils.py")
spec = importlib.util.spec_from_file_location("utils", location)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)


DECTRIS_TESTDATA_PATH = os.path.join(
    utils.get_testdata_path(),
    'dectris', 'zmqdump.dat.128x128-id34-exte-bslz4'
)
HAVE_DECTRIS_TESTDATA = os.path.exists(DECTRIS_TESTDATA_PATH)


@pytest.fixture
def ltl_ctx():
    # Debugging disabled since acquisition objects can't always be serialized
    inline_executor = InlineJobExecutor(debug=False, inline_threads=2)
    return ltl.LiveContext(executor=inline_executor, plot_class=Dummy2DPlot)


@pytest.fixture
def ltl_ctx_fast():
    inline_executor = InlineJobExecutor(debug=False, inline_threads=2)
    return ltl.LiveContext(executor=inline_executor, plot_class=Dummy2DPlot)


@pytest.fixture(scope="session")
def ctx_pipelined():
    executor = None
    try:
        executor = PipelinedExecutor()
        yield ltl.LiveContext(executor=executor, plot_class=Dummy2DPlot)
    finally:
        if executor is not None:
            executor.close()


@pytest.fixture()
def default_aq(ltl_ctx):
    conn = ltl_ctx.make_connection('memory').open(
        data=None, extra_kwargs={'datashape': [16, 16, 32, 32]}
    )
    return ltl_ctx.make_acquisition(conn=conn)


def pytest_collectstart(collector):
    # nbval: ignore some output types
    if collector.fspath and collector.fspath.ext == '.ipynb':
        collector.skip_compare += 'text/html', 'application/javascript', 'stderr',


@pytest.fixture(autouse=True)
def add_np(doctest_namespace):
    doctest_namespace['np'] = np


@pytest.fixture(autouse=True)
def add_helpers(doctest_namespace, ctx_pipelined):
    from libertem.udf.sum import SumUDF
    doctest_namespace['ctx'] = ctx_pipelined
    doctest_namespace['SumUDF'] = SumUDF


@pytest.fixture(autouse=True)
def add_sims(doctest_namespace):
    if not HAVE_DECTRIS_TESTDATA:
        # FIXME: add some kind of proxy object that calls
        # pytest.skip on access? is this possible somehow?
        yield
        return
    path = DECTRIS_TESTDATA_PATH

    sim = contextmanager(utils.run_camera_sim)

    with sim(
        cls=DectrisSim,
        path=path,
        port=0,
        zmqport=0,
    ) as dectris_runner:
        api_port, data_port = dectris_runner.port, dectris_runner.zmqport

        doctest_namespace['DCU_API_PORT'] = api_port
        doctest_namespace['DCU_DATA_PORT'] = data_port

        yield
