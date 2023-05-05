"""
Global test configuration. Use this file to define fixtures to use
in both doctests and regular tests.
"""
import os
import importlib
from contextlib import contextmanager

from libertem.executor.inline import InlineJobExecutor
from libertem.executor.pipelined import PipelinedExecutor

import psutil
import pytest
import numpy as np

from libertem.viz.base import Dummy2DPlot

from libertem_live.detectors.dectris.sim import DectrisSim
from libertem_live.detectors.merlin.sim import CameraSim
from libertem_live.detectors.asi_tpx3.sim import TpxCameraSim
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
MIB_TESTDATA_PATH = os.path.join(utils.get_testdata_path(), 'default.mib')
HAVE_MIB_TESTDATA = os.path.exists(MIB_TESTDATA_PATH)


TPX3_TESTDATA_PATH = os.path.join(
    utils.get_testdata_path(),
    'asi-tpx3', 'header_data_with_padding_vals32bits.scr',
)
HAVE_TPX3_TESTDATA = os.path.exists(TPX3_TESTDATA_PATH)


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
        num_cpus = min(
            psutil.cpu_count(logical=False),
            4
        )
        spec = PipelinedExecutor.make_spec(
            cpus=num_cpus,
            cudas=0,
        )
        executor = PipelinedExecutor(spec=spec)
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


@pytest.fixture(autouse=True, scope='session')
def add_np(doctest_namespace):
    doctest_namespace['np'] = np


@pytest.fixture(autouse=True, scope='session')
def add_helpers(doctest_namespace, ctx_pipelined):
    from libertem.udf.sum import SumUDF
    doctest_namespace['ctx'] = ctx_pipelined
    doctest_namespace['SumUDF'] = SumUDF


@pytest.fixture(autouse=True, scope='session')
def add_sims(doctest_namespace):
    if not HAVE_DECTRIS_TESTDATA or not HAVE_MIB_TESTDATA:
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
        tolerate_timeouts=False,
    ) as dectris_runner, sim(
        cls=CameraSim,
        host='127.0.0.1',
        data_port=0,
        control_port=0,
        trigger_port=0,
        path=MIB_TESTDATA_PATH,
        nav_shape=(32, 32),
        wait_trigger=False,
        initial_params={
            'IMAGEX': "256",
            'IMAGEY': "256",
            'COUNTERDEPTH': '12',
        },
    ) as merlin_runner, sim(
        cls=TpxCameraSim,
        paths=[TPX3_TESTDATA_PATH],
        cached='MEM',
        sleep=0.0,
        port=0,
    ) as tpx_runner:
        dectris_api_port, dectris_data_port = dectris_runner.port, dectris_runner.zmqport
        merlin_api_port, merlin_data_port = (
            merlin_runner.control_t.sockname[1],
            merlin_runner.server_t.sockname[1]
        )

        doctest_namespace['DCU_API_PORT'] = dectris_api_port
        doctest_namespace['DCU_DATA_PORT'] = dectris_data_port

        doctest_namespace['MERLIN_API_PORT'] = merlin_api_port
        doctest_namespace['MERLIN_DATA_PORT'] = merlin_data_port

        doctest_namespace['TPX3_PORT'] = tpx_runner.server_t.port

        yield
