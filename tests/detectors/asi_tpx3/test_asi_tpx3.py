from typing import Optional, Tuple

import numpy as np
import pytest

from libertem_live.api import LiveContext, Hooks
from libertem_live.hooks import ReadyForDataEnv, DetermineNavShapeEnv
from libertem.udf.sumsigudf import SumSigUDF

pytestmark = [
    pytest.mark.data,
]


def test_smoke(ctx_pipelined: LiveContext, tpx_sim):
    with ctx_pipelined.make_connection('asi_tpx3').open(
        data_port=tpx_sim,
    ) as conn:
        pending_aq = conn.wait_for_acquisition(10.0)
        assert pending_aq is not None
        aq = ctx_pipelined.make_acquisition(
            conn=conn,
            pending_aq=pending_aq,
        )
        res = ctx_pipelined.run_udf(dataset=aq, udf=SumSigUDF())
        assert not np.allclose(res['intensity'], 0)


def test_hooks(ctx_pipelined: LiveContext, tpx_sim):

    class MyHooks(Hooks):
        def __init__(self):
            self.ready_called = False
            self.det_shape_called = False

        def on_determine_nav_shape(self, env: DetermineNavShapeEnv) -> Optional[Tuple[int, ...]]:
            self.det_shape_called = True
            return None

        def on_ready_for_data(self, env: ReadyForDataEnv):
            self.ready_called = True

    with ctx_pipelined.make_connection('asi_tpx3').open(
        data_port=tpx_sim,
    ) as conn:
        pending_aq = conn.wait_for_acquisition(10.0)
        assert pending_aq is not None
        hooks = MyHooks()
        aq = ctx_pipelined.make_acquisition(
            conn=conn,
            pending_aq=pending_aq,
            hooks=hooks,
        )
        assert not hooks.ready_called
        res = ctx_pipelined.run_udf(dataset=aq, udf=SumSigUDF())
        assert not hooks.ready_called
        assert hooks.det_shape_called
        assert not np.allclose(res['intensity'], 0)


def test_multiple_with_stmt(ctx_pipelined: LiveContext, tpx_sim):
    conn = ctx_pipelined.make_connection('asi_tpx3').open(
        data_port=tpx_sim,
    )

    with conn:
        pending_aq = conn.wait_for_acquisition(1.0)
        assert pending_aq is not None
        aq = ctx_pipelined.make_acquisition(
            conn=conn,
            pending_aq=pending_aq,
        )
        ctx_pipelined.run_udf(dataset=aq, udf=SumSigUDF())

    with conn:
        pending_aq = conn.wait_for_acquisition(1.0)
        assert pending_aq is not None
        aq = ctx_pipelined.make_acquisition(
            conn=conn,
            pending_aq=pending_aq,
        )
        ctx_pipelined.run_udf(dataset=aq, udf=SumSigUDF())


def test_reconnect(ctx_pipelined: LiveContext, tpx_sim):
    conn = ctx_pipelined.make_connection('asi_tpx3').open(
        data_port=tpx_sim,
    )

    try:
        pending_aq = conn.wait_for_acquisition(1.0)
        assert pending_aq is not None
        aq = ctx_pipelined.make_acquisition(
            conn=conn,
            pending_aq=pending_aq,
        )
        ctx_pipelined.run_udf(dataset=aq, udf=SumSigUDF())
        conn.reconnect()

        pending_aq = conn.wait_for_acquisition(2.0)
        assert pending_aq is not None
        aq = ctx_pipelined.make_acquisition(
            conn=conn,
            pending_aq=pending_aq,
        )
        ctx_pipelined.run_udf(dataset=aq, udf=SumSigUDF())
    finally:
        conn.close()
