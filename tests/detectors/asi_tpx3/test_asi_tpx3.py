from libertem_live.api import LiveContext
from libertem.udf.sumsigudf import SumSigUDF


def test_smoke(ctx_pipelined: LiveContext, tpx_sim):
    with ctx_pipelined.make_connection('asi_tpx3').open(
        uri=f"127.0.0.1:{tpx_sim}",
        num_slots=1000,
        bytes_per_chunk=1500000,
        chunks_per_stack=16,
    ) as conn:
        pending_aq = conn.wait_for_acquisition(10.0)
        assert pending_aq is not None
        aq = ctx_pipelined.make_acquisition(
            conn=conn,
            pending_aq=pending_aq,
        )
        ctx_pipelined.run_udf(dataset=aq, udf=SumSigUDF())
