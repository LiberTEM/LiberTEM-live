"""
Extra tests to get coverage for the simulator configurations
"""
import os
from contextlib import contextmanager

import pytest
import numpy as np

from libertem_live.api import LiveContext
from libertem_live.detectors.asi_tpx3.sim import TpxCameraSim
from libertem.udf.sumsigudf import SumSigUDF

from utils import run_camera_sim

pytestmark = [
    pytest.mark.data,
]


def test_memfd_smoke(ctx_pipelined: LiveContext, tpx_testdata_path):
    try:
        import memfd  # noqa
    except ImportError:
        pytest.skip("need pymemfd for this test (Linux only)")

    if not os.path.exists(tpx_testdata_path):
        pytest.skip("need ASI TPX3 testdata")

    sim_ctx_mgr = contextmanager(run_camera_sim)

    with sim_ctx_mgr(
        cls=TpxCameraSim,
        paths=[tpx_testdata_path],
        port=0,
        sleep=0.1,
        cached='MEMFD',
    ) as tpx_sim:
        with ctx_pipelined.make_connection('asi_tpx3').open(
            data_port=tpx_sim.server_t.port,
        ) as conn:
            pending_aq = conn.wait_for_acquisition(10.0)
            assert pending_aq is not None
            aq = ctx_pipelined.make_acquisition(
                conn=conn,
                pending_aq=pending_aq,
            )
            res = ctx_pipelined.run_udf(dataset=aq, udf=SumSigUDF())
            assert not np.allclose(res['intensity'], 0)


def test_memfd_mock_data(ctx_pipelined: LiveContext):
    try:
        import memfd  # noqa
    except ImportError:
        pytest.skip("need pymemfd for this test (Linux only)")

    sim_ctx_mgr = contextmanager(run_camera_sim)

    with sim_ctx_mgr(
        cls=TpxCameraSim,
        mock_nav_shape=(32, 32),
        port=0,
        sleep=0.1,
        cached='MEMFD',
    ) as tpx_sim:
        with ctx_pipelined.make_connection('asi_tpx3').open(
            data_port=tpx_sim.server_t.port,
        ) as conn:
            pending_aq = conn.wait_for_acquisition(10.0)
            assert pending_aq is not None
            aq = ctx_pipelined.make_acquisition(
                conn=conn,
                pending_aq=pending_aq,
            )
            res = ctx_pipelined.run_udf(dataset=aq, udf=SumSigUDF())
            assert not np.allclose(res['intensity'], 0)


def test_mock_data_shape(ctx_pipelined: LiveContext):
    sim_ctx_mgr = contextmanager(run_camera_sim)

    with sim_ctx_mgr(
        cls=TpxCameraSim,
        mock_nav_shape=(64, 64),
        port=0,
        sleep=0.1,
        cached='MEM',
    ) as tpx_sim:
        with ctx_pipelined.make_connection('asi_tpx3').open(
            data_port=tpx_sim.server_t.port,
        ) as conn:
            pending_aq = conn.wait_for_acquisition(10.0)
            assert pending_aq is not None
            aq = ctx_pipelined.make_acquisition(
                conn=conn,
                pending_aq=pending_aq,
            )
            res = ctx_pipelined.run_udf(dataset=aq, udf=SumSigUDF())
            assert not np.allclose(res['intensity'], 0)
