import numpy as np

from libertem_live.api import LiveContext
from libertem.udf.sum import SumUDF


def test_active_api(ltl_ctx: LiveContext):
    data = np.random.random((13, 17, 19, 23))
    conn = ltl_ctx.make_connection('memory').open(data=data)
    with conn:
        aq = ltl_ctx.make_acquisition(conn=conn)
        udf = SumUDF()
        res = ltl_ctx.run_udf(dataset=aq, udf=udf)
        s = data.sum(axis=(0, 1))
        print(s.shape)
        print(res['intensity'].data.shape)
        assert np.allclose(res['intensity'].data, s)


def test_passive_api(ltl_ctx: LiveContext):
    data = np.random.random((13, 17, 19, 23))
    conn = ltl_ctx.make_connection('memory').open(data=data)
    with conn:
        pending_aq = conn.wait_for_acquisition(10.0)
        assert pending_aq is not None
        aq = ltl_ctx.make_acquisition(conn=conn, pending_aq=pending_aq)
        udf = SumUDF()
        res = ltl_ctx.run_udf(dataset=aq, udf=udf)
        s = data.sum(axis=(0, 1))
        print(s.shape)
        print(res['intensity'].data.shape)
        assert np.allclose(res['intensity'].data, s)
