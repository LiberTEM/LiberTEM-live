import pytest
import numpy as np
from numpy.testing import assert_allclose
from sparseconverter import get_device_class

from libertem.udf.sum import SumUDF

from libertem_live.api import LiveContext
from libertem_live.udf.monitor import SignalMonitorUDF, PartitionMonitorUDF

from utils import set_device_class


@pytest.mark.parametrize(
    'backend', tuple(SignalMonitorUDF().get_backends()) + (None, )
)
def test_monitor(ltl_ctx: LiveContext, backend):
    with set_device_class(get_device_class(backend)):
        if backend is None:
            backends = None
        else:
            backends = (backend, )
        data = np.random.random((13, 17, 19, 23))
        conn = ltl_ctx.make_connection('memory').open(
            data=data,
            extra_kwargs=dict(array_backends=backends)
        )
        with conn:
            aq = ltl_ctx.make_acquisition(conn=conn)
            udf = SignalMonitorUDF()
            res = ltl_ctx.run_udf(dataset=aq, udf=udf)
            assert np.all(res['intensity'].data == data[-1, -1])


@pytest.mark.parametrize(
    'backend', tuple(PartitionMonitorUDF().get_backends()) + (None, )
)
@pytest.mark.parametrize(
    'udf_kwargs', ({}, {'dtype': np.complex128})
)
def test_partition_monitor(ltl_ctx: LiveContext, backend, udf_kwargs):
    with set_device_class(get_device_class(backend)):
        if backend is None:
            backends = None
        else:
            backends = (backend, )
        data = np.random.random((13, 17, 19, 23))

        conn = ltl_ctx.make_connection('memory').open(
            data=data,
            extra_kwargs=dict(array_backends=backends)
        )
        with conn:
            aq = ltl_ctx.make_acquisition(conn=conn)
            udf = PartitionMonitorUDF(**udf_kwargs)

            dtype = udf_kwargs.get('dtype')
            res = np.zeros(aq.shape.sig, dtype=dtype)
            for part_res in ltl_ctx.run_udf_iter(dataset=aq, udf=udf):
                res += part_res.buffers[0]['intensity'].raw_data

            ref = ltl_ctx.run_udf(dataset=aq, udf=SumUDF())

            assert_allclose(ref['intensity'].raw_data, res)
