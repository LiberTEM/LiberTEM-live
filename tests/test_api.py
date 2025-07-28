import contextlib

import numpy as np
import pytest

from libertem.udf.base import NoOpUDF, UDF
from libertem.utils.devices import detect

from libertem_live.api import LiveContext, Hooks
from libertem_live.udf.monitor import SignalMonitorUDF


def test_default_ctx():
    data = np.random.random((13, 17, 19, 23))
    with LiveContext() as ctx:
        with ctx.make_connection('memory').open(data=data) as conn:
            aq = ctx.make_acquisition(conn=conn)
            udf1 = NoOpUDF()
            ctx.run_udf(dataset=aq, udf=udf1)
            for _ in ctx.run_udf_iter(dataset=aq, udf=udf1):
                pass


def test_trigger(ltl_ctx):
    data = np.random.random((13, 17, 19, 23))
    triggered = np.array((False,))

    class MyHooks(Hooks):
        def on_ready_for_data(self, env):
            triggered[:] = True
            assert tuple(env.aq.shape.nav) == data.shape[:2]

    with ltl_ctx.make_connection('memory').open(data=data) as conn:
        aq = ltl_ctx.make_acquisition(conn=conn, hooks=MyHooks())

        udf1 = SignalMonitorUDF()
        udf2 = NoOpUDF()
        res = ltl_ctx.run_udf(dataset=aq, udf=[udf1, udf2])

        assert np.all(res[0]['intensity'].data == data[-1, -1])
        assert triggered


def test_dataset(ltl_ctx):
    data = np.random.random((13, 17, 19, 23))

    ds = ltl_ctx.load('memory', data=data)

    udf = SignalMonitorUDF()

    res = ltl_ctx.run_udf(dataset=ds, udf=udf)

    assert np.all(res['intensity'].data == data[-1, -1])


def test_removed_api(ltl_ctx):
    with pytest.raises(RuntimeError):
        ltl_ctx.prepare_acquisition('merlin')


def test_update_parameters_iter_sync(ltl_ctx):
    data = np.random.random((13, 17, 19, 23))

    with ltl_ctx.make_connection('memory').open(data=data) as conn:
        aq = ltl_ctx.make_acquisition(conn=conn)

        udf1 = SignalMonitorUDF()
        udf2 = NoOpUDF()
        res_iter = ltl_ctx.run_udf_iter(dataset=aq, udf=[udf1, udf2], sync=True)
        with contextlib.closing(res_iter):
            for item in res_iter:
                res_iter.update_parameters_experimental([{}, {}])


class CheckXPUDF(UDF):
    def get_result_buffers(self):
        return {
            'process_cupy': self.buffer(kind='nav', dtype='bool'),
            'merge_cupy': self.buffer(kind='nav', dtype='bool'),
            'result_cupy': self.buffer(
                kind='single', dtype='bool', extra_shape=(1, ), use='result_only'
            ),
        }

    def xp_is_cupy(self):
        obj = str(self.xp)
        is_cupy = 'cupy' in obj
        is_numpy = 'numpy' in obj
        assert is_cupy != is_numpy
        return is_cupy

    def process_frame(self, frame):
        self.results.process_cupy[:] = self.xp_is_cupy()

    def merge(self, dest, src):
        dest.process_cupy[:] = src.process_cupy[:]
        dest.merge_cupy[:] = self.xp_is_cupy()

    def get_results(self):
        res = np.zeros((1, ), dtype='bool')
        res[0] = self.xp_is_cupy()
        return {
            'result_cupy': res,
        }


@pytest.mark.slow
def test_default_context_with_main_gpu():
    try:
        from libertem.executor.base import make_canonical  # NOQA: F401
        has_feature = True
    except (ImportError, ModuleNotFoundError):
        has_feature = False
    d = detect()
    use_cupy = d['has_cupy'] and d['cudas'] and has_feature
    ctx = LiveContext()
    try:
        udf = CheckXPUDF()
        ds = ctx.load('memory', data=np.zeros((1, 1, 1, 1)))
        res = ctx.run_udf(dataset=ds, udf=udf)
        if use_cupy:
            assert np.all(res['merge_cupy'])
            assert np.all(res['result_cupy'])
        else:
            assert not np.any(res['merge_cupy'])
            assert not np.any(res['result_cupy'])
        assert not np.any(res['process_cupy'])
    finally:
        ctx.close()
