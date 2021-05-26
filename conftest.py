"""
Global test configuration. Use this file to define fixtures to use
in both doctests and regular tests.
"""
from libertem.executor.inline import InlineJobExecutor
import pytest

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


@pytest.fixture()
def default_aq(ltl_ctx):
    return ltl_ctx.prepare_acquisition('memory', trigger=None, datashape=[16, 16, 32, 32])
