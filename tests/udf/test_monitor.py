import numpy as np

from libertem_live.udf.monitor import SignalMonitorUDF


def test_monitor(ltl_ctx):
    data = np.random.random((13, 17, 19, 23))
    aq = ltl_ctx.prepare_acquisition('memory', trigger=None, data=data)

    udf = SignalMonitorUDF()

    res = ltl_ctx.run_udf(dataset=aq, udf=udf)

    assert np.all(res['intensity'].data == data[-1, -1])
