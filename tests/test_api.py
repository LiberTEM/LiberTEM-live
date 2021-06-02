import numpy as np
import pytest

from libertem.udf.base import NoOpUDF

from libertem_live.api import LiveContext
from libertem_live.udf.monitor import SignalMonitorUDF


def test_default_ctx():
    data = np.random.random((13, 17, 19, 23))
    ctx = LiveContext()

    aq = ctx.prepare_acquisition('memory', data=data)

    udf1 = NoOpUDF()

    ctx.run_udf(dataset=aq, udf=udf1)
    for res in ctx.run_udf_iter(dataset=aq, udf=udf1):
        pass


def test_trigger(ltl_ctx):
    data = np.random.random((13, 17, 19, 23))

    triggered = np.array((False,))

    def trigger(acquisition):
        triggered[:] = True
        assert tuple(acquisition.shape.nav) == data.shape[:2]

    aq = ltl_ctx.prepare_acquisition('memory', trigger=trigger, data=data)

    udf1 = SignalMonitorUDF()
    udf2 = NoOpUDF()

    res = ltl_ctx.run_udf(dataset=aq, udf=[udf1, udf2])

    assert np.all(res[0]['intensity'].data == data[-1, -1])
    assert triggered


def test_bad_type(ltl_ctx):
    with pytest.raises(ValueError):
        ltl_ctx.prepare_acquisition('asdf does not exist')


def test_dataset(ltl_ctx):
    data = np.random.random((13, 17, 19, 23))

    ds = ltl_ctx.load('memory', data=data)

    udf = SignalMonitorUDF()

    res = ltl_ctx.run_udf(dataset=ds, udf=udf)

    assert np.all(res['intensity'].data == data[-1, -1])
