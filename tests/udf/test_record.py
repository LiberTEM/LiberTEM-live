import os

import numpy as np

from libertem_live.udf.record import RecordUDF


def test_record(tmpdir_factory, ctx_pipelined):
    datadir = tmpdir_factory.mktemp('data')
    filename = 'numpyfile.npy'
    path = os.path.join(datadir, filename)
    data = np.random.random((13, 17, 19, 23))
    aq = ctx_pipelined.prepare_acquisition('memory', trigger=None, data=data)

    udf = RecordUDF(path)

    ctx_pipelined.run_udf(dataset=aq, udf=udf)

    res = np.load(path, mmap_mode='r')

    assert res.dtype == data.dtype
    assert res.shape == data.shape
    assert np.all(res == data)
