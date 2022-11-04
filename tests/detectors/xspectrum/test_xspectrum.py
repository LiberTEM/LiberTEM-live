import sys
import os
from unittest import mock

import numpy as np

from libertem.udf.sumsigudf import SumSigUDF
from libertem.udf.sum import SumUDF
from libertem.common import Shape
import pytest
from libertem_live.detectors.xspectrum.acquisition import XSpectrumAcquisition
from libertem_live.detectors.xspectrum.mock import mock_xspectrum


pytestmark = pytest.mark.skipif(sys.version_info < (3, 7),
                                reason="X-Spectrum support requires Python 3.7")


# Default location, exists in simulator
HAVE_XSPECTRUM_CONFIG = os.path.exists('/opt/xsp/config/system.yml')


def test_udf_sig(ctx_pipelined):
    dataset_shape = Shape((128, 512, 512), sig_dims=2)
    data = np.random.randn(*dataset_shape).astype(np.uint8)
    with mock_xspectrum(data):
        aq = XSpectrumAcquisition(
            nav_shape=tuple(dataset_shape.nav),
            frames_per_partition=42,  # chosen not to evenly divide `dataset_shape.nav`
        )
        aq.initialize(ctx_pipelined.executor)

        udf = SumUDF()
        res = ctx_pipelined.run_udf(dataset=aq, udf=udf)

        assert np.allclose(
            res['intensity'].data,
            data.astype(np.float32).sum(axis=0),
        )


def test_udf_nav(ctx_pipelined):
    dataset_shape = Shape((128, 512, 512), sig_dims=2)
    data = np.random.randn(*dataset_shape).astype(np.uint8)
    with mock_xspectrum(data):
        trigger = mock.MagicMock(return_value=None)
        aq = XSpectrumAcquisition(
            nav_shape=tuple(dataset_shape.nav),
            frames_per_partition=42,  # chosen not to evenly divide `dataset_shape.nav`
            trigger=trigger,
        )
        aq.initialize(ctx_pipelined.executor)

        udf = SumSigUDF()
        res = ctx_pipelined.run_udf(dataset=aq, udf=udf)

        assert np.allclose(
            res['intensity'].data,
            data.sum(axis=(1, 2)),
        )
        trigger.assert_called_with(aq)


@pytest.mark.skipif(not HAVE_XSPECTRUM_CONFIG, reason="Needs X-Spectrum camera system or simulator")
def test_acquisition(ctx_pipelined):
    aq = XSpectrumAcquisition(nav_shape=(23, 5), frames_per_partition=12)
    aq = aq.initialize(ctx_pipelined.executor)
    res = ctx_pipelined.run_udf(dataset=aq, udf=[SumUDF(), SumSigUDF()])

    import pyxsp as px

    s = px.System('/opt/xsp/config/system.yml')
    s.connect()
    s.initialize()

    r = s.open_receiver(s.list_receivers()[0])
    sig_shape = r.frame_height, r.frame_width
    min_itemsize = r.frame_depth / 8

    assert aq.dtype.itemsize >= min_itemsize
    assert tuple(aq.shape.sig) == sig_shape

    assert res[0]['intensity'].data.shape == sig_shape
    assert res[1]['intensity'].data.shape == (23, 5)
