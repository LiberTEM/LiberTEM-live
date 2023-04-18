"""
Tests that are specifically meant to exercise the different parameters of the
simulator, and not the usual LiberTEM-live APIs.
"""

import os
import platform

import pytest
from numpy.testing import assert_allclose

from libertem.udf.sum import SumUDF
from libertem_live.api import LiveContext
from utils import get_testdata_path


MIB_TESTDATA_PATH = os.path.join(get_testdata_path(), 'default.mib')
HAVE_MIB_TESTDATA = os.path.exists(MIB_TESTDATA_PATH)

PTYCHO_TESTDATA_PATH = os.path.join(get_testdata_path(), '20200518 165148', 'default.hdr')
HAVE_PTYCHO_TESTDATA = os.path.exists(PTYCHO_TESTDATA_PATH)

pytestmark = [
    pytest.mark.skipif(not HAVE_MIB_TESTDATA, reason="need .mib testdata"),
    pytest.mark.data,
]


def test_acquisition_cached(
    ctx_pipelined: LiveContext,
    merlin_detector_cached,
    merlin_control_sim,
    merlin_ds
):
    host, port = merlin_detector_cached
    api_host, api_port = merlin_control_sim

    with ctx_pipelined.make_connection('merlin').open(
        data_host=host,
        data_port=port,
        api_host=api_host,
        api_port=api_port,
        drain=False,
    ) as conn:
        aq = ctx_pipelined.make_acquisition(
            conn=conn,
            nav_shape=(32, 32),
        )
        udf = SumUDF()

        res = ctx_pipelined.run_udf(dataset=aq, udf=udf)
        ref = ctx_pipelined.run_udf(dataset=merlin_ds, udf=udf)

        assert_allclose(res['intensity'], ref['intensity'])


@pytest.mark.skipif(platform.system() != 'Linux',
                    reason="MemFD is Linux-only")
def test_acquisition_memfd(
    ctx_pipelined: LiveContext,
    merlin_detector_memfd,
    merlin_control_sim,
    merlin_ds,
):
    host, port = merlin_detector_memfd
    api_host, api_port = merlin_control_sim

    with ctx_pipelined.make_connection('merlin').open(
        data_host=host,
        data_port=port,
        api_host=api_host,
        api_port=api_port,
        drain=False,
    ) as conn:
        aq = ctx_pipelined.make_acquisition(
            conn=conn,
            nav_shape=(32, 32),
        )
        udf = SumUDF()

        res = ctx_pipelined.run_udf(dataset=aq, udf=udf)
        ref = ctx_pipelined.run_udf(dataset=merlin_ds, udf=udf)

        assert_allclose(res['intensity'], ref['intensity'])
