"""
Global test configuration. Use this file to define fixtures to use
in both doctests and regular tests.
"""
from libertem.executor.inline import InlineJobExecutor
from libertem.executor.pipelined import PipelinedExecutor

import pytest
import numpy as np

from libertem.viz.base import Dummy2DPlot

from libertem_live import api as ltl


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
