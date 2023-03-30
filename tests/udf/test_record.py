import os

import numpy as np

from libertem_live.udf.record import RecordUDF
from libertem_live.api import LiveContext


def test_record(tmpdir_factory, ctx_pipelined: LiveContext):
    datadir = tmpdir_factory.mktemp('data')
    filename = 'numpyfile.npy'
    path = os.path.join(datadir, filename)
    data = np.random.random((13, 17, 19, 23))

    with ctx_pipelined.make_connection('memory').open(data=data) as conn:
        aq = ctx_pipelined.make_acquisition(conn=conn)

        udf = RecordUDF(path)

        ctx_pipelined.run_udf(dataset=aq, udf=udf)

        res = np.load(path, mmap_mode='r')

        assert res.dtype == data.dtype
        assert res.shape == data.shape
        assert np.all(res == data)
