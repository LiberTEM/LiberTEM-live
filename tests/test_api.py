import numpy as np
import pytest

from libertem.udf.base import NoOpUDF

from libertem_live.api import LiveContext, Hooks
from libertem_live.udf.monitor import SignalMonitorUDF


def test_default_ctx():
    data = np.random.random((13, 17, 19, 23))
    ctx = LiveContext()
    with ctx.make_connection('memory').open(data=data) as conn:
        aq = ctx.make_acquisition(conn=conn)
        udf1 = NoOpUDF()
        ctx.run_udf(dataset=aq, udf=udf1)
        for _ in ctx.run_udf_iter(dataset=aq, udf=udf1):
            pass


def test_trigger(ltl_ctx):
    data = np.random.random((13, 17, 19, 23))
    triggered = np.array((False,))

    class MyHooks(Hooks):
        def on_ready_for_data(self, env):
            triggered[:] = True
            assert tuple(env.aq.shape.nav) == data.shape[:2]

    with ltl_ctx.make_connection('memory').open(data=data) as conn:
        aq = ltl_ctx.make_acquisition(conn=conn, hooks=MyHooks())

        udf1 = SignalMonitorUDF()
        udf2 = NoOpUDF()
        res = ltl_ctx.run_udf(dataset=aq, udf=[udf1, udf2])

        assert np.all(res[0]['intensity'].data == data[-1, -1])
        assert triggered


def test_dataset(ltl_ctx):
    data = np.random.random((13, 17, 19, 23))

    ds = ltl_ctx.load('memory', data=data)

    udf = SignalMonitorUDF()

    res = ltl_ctx.run_udf(dataset=ds, udf=udf)

    assert np.all(res['intensity'].data == data[-1, -1])


def test_removed_api(ltl_ctx):
    with pytest.raises(RuntimeError):
        ltl_ctx.prepare_acquisition('merlin')
