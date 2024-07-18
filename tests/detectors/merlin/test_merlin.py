import os
import concurrent.futures
from typing import Optional

import pytest
import numpy as np
from numpy.testing import assert_allclose

from libertem.udf import UDF
from libertem.udf.sum import SumUDF
from libertem.contrib.daskadapter import make_dask_array
from libertem.io.dataset.base import TilingScheme, DataSet
from libertem.common import Shape

from libertem_live.api import Hooks, LiveContext
from libertem_live.hooks import ReadyForDataEnv
from libertem_live.detectors.merlin import (
    MerlinControl,
)
from libertem_live.detectors.merlin.sim import TriggerClient

from utils import get_testdata_path


MIB_TESTDATA_PATH = os.path.join(get_testdata_path(), 'default.mib')
HAVE_MIB_TESTDATA = os.path.exists(MIB_TESTDATA_PATH)

PTYCHO_TESTDATA_PATH = os.path.join(get_testdata_path(), '20200518 165148', 'default.hdr')
HAVE_PTYCHO_TESTDATA = os.path.exists(PTYCHO_TESTDATA_PATH)

pytestmark = [
    pytest.mark.skipif(not HAVE_MIB_TESTDATA, reason="need .mib testdata"),
    pytest.mark.data,
]


class MyHooks(Hooks):
    def __init__(self, triggered: np.ndarray, merlin_ds: "DataSet"):
        self.ds = merlin_ds
        self.triggered = triggered

    def on_ready_for_data(self, env: ReadyForDataEnv):
        self.triggered[:] = True
        assert env.aq.shape.nav == self.ds.shape.nav


def test_acquisition(
    ctx_pipelined: LiveContext,
    merlin_ds,
    default_conn,
):
    triggered = np.array((False,))

    aq = ctx_pipelined.make_acquisition(
        conn=default_conn,
        hooks=MyHooks(triggered=triggered, merlin_ds=merlin_ds),
        nav_shape=(32, 32),
    )
    udf = SumUDF()

    assert not triggered[0]
    res = ctx_pipelined.run_udf(dataset=aq, udf=udf)
    assert triggered[0]

    ref = ctx_pipelined.run_udf(dataset=merlin_ds, udf=udf)

    assert_allclose(res['intensity'], ref['intensity'])


def test_small_shm(
    ctx_pipelined: LiveContext,
    merlin_detector_sim,
    merlin_control_sim,  # XXX not really the matching control, but it kinda works...
    merlin_ds,
):
    host, port = merlin_detector_sim
    api_host, api_port = merlin_control_sim

    with ctx_pipelined.make_connection('merlin').open(
        data_host=host,
        data_port=port,
        api_host=api_host,
        api_port=api_port,
        drain=False,
    ) as conn:
        import libertem_qd_mpx
        # monkey-patch in a different underlying connection:
        conn._data_socket.close()
        conn._data_socket = libertem_qd_mpx.QdConnection(
            data_host=host,
            data_port=port,
            frame_stack_size=1,
            num_slots=4,  # NOTE this ridiculously small value
            shm_handle_path=conn._make_socket_path(),
            drain=False,
        )
        conn._data_socket.start_passive()

        aq = ctx_pipelined.make_acquisition(
            conn=conn,
            nav_shape=(32, 32),
        )
        udf = SumUDF()

        res = ctx_pipelined.run_udf(dataset=aq, udf=udf)
        ref = ctx_pipelined.run_udf(dataset=merlin_ds, udf=udf)

        assert_allclose(res['intensity'], ref['intensity'])


def test_passive_acquisition(
    ctx_pipelined: LiveContext,
    merlin_ds,
    default_conn,
):
    triggered = np.array((False,))

    pending_aq = default_conn.wait_for_acquisition(10.0)
    assert pending_aq is not None

    aq = ctx_pipelined.make_acquisition(
        conn=default_conn,
        hooks=MyHooks(triggered=triggered, merlin_ds=merlin_ds),
        pending_aq=pending_aq,
        nav_shape=(32, 32),
    )
    udf = SumUDF()

    assert not triggered[0]
    res = ctx_pipelined.run_udf(dataset=aq, udf=udf)
    assert not triggered[0]  # in passive mode, we don't call the on_ready_for_data hook

    ref = ctx_pipelined.run_udf(dataset=merlin_ds, udf=udf)

    assert_allclose(res['intensity'], ref['intensity'])


def test_passive_timeout(
    ctx_pipelined: LiveContext,
    conn_triggered,
):
    with conn_triggered as conn:
        # without external interaction, we don't get an acquisition:
        pending_aq = conn.wait_for_acquisition(0.5)
        assert pending_aq is None


class ProcessPartitionUDF(UDF):
    def get_result_buffers(self):
        return {
            'result': self.buffer(kind='nav', dtype=np.float32),
        }

    def process_partition(self, partition):
        self.results.result[:] = partition.sum(axis=(-1, -2))


def test_process_partition(
    ctx_pipelined,
    default_conn,
    merlin_ds
):
    triggered = np.array((False,))

    aq = ctx_pipelined.make_acquisition(
        conn=default_conn,
        hooks=MyHooks(triggered=triggered, merlin_ds=merlin_ds),
        nav_shape=(32, 32),
    )
    udf = ProcessPartitionUDF()

    res = ctx_pipelined.run_udf(dataset=aq, udf=udf)
    ref = ctx_pipelined.run_udf(dataset=merlin_ds, udf=udf)

    assert_allclose(res['result'], ref['result'])


def test_acquisition_dry_run(
    ctx_pipelined,
    default_conn,
    merlin_ds,
):
    triggered = np.array((False,))

    aq = ctx_pipelined.make_acquisition(
        conn=default_conn,
        hooks=MyHooks(triggered=triggered, merlin_ds=merlin_ds),
        nav_shape=(32, 32),
    )
    udf = SumUDF()

    assert not triggered[0]
    runner_cls = ctx_pipelined.executor.get_udf_runner()
    runner_cls.dry_run([udf], aq, None)
    assert not triggered[0]
    ctx_pipelined.run_udf(dataset=aq, udf=udf)
    assert triggered[0]


def test_acquisition_iter(
    ctx_pipelined,
    default_conn,
    merlin_ds
):
    triggered = np.array((False,))

    aq = ctx_pipelined.make_acquisition(
        conn=default_conn,
        hooks=MyHooks(triggered=triggered, merlin_ds=merlin_ds),
        nav_shape=(32, 32),
    )
    udf = SumUDF()

    res = None
    assert not triggered[0]
    for res in ctx_pipelined.run_udf_iter(dataset=aq, udf=udf, sync=True):
        assert triggered[0]
        pass

    ref = ctx_pipelined.run_udf(dataset=merlin_ds, udf=udf)

    assert res is not None
    assert_allclose(res.buffers[0]['intensity'], ref['intensity'])


@pytest.mark.asyncio
async def test_acquisition_async(
    ctx_pipelined,
    default_conn,
    merlin_ds,
):
    triggered = triggered = np.array((False,))

    with default_conn:
        aq = ctx_pipelined.make_acquisition(
            conn=default_conn,
            hooks=MyHooks(triggered=triggered, merlin_ds=merlin_ds),
            nav_shape=(32, 32),
        )

        udf = SumUDF()

        assert not triggered[0]
        res = await ctx_pipelined.run_udf(dataset=aq, udf=udf, sync=False)
        assert triggered[0]
    ref = ctx_pipelined.run_udf(dataset=merlin_ds, udf=udf)

    assert_allclose(res['intensity'], ref['intensity'])

    triggered[:] = False

    with default_conn:
        aq = ctx_pipelined.make_acquisition(
            conn=default_conn,
            hooks=MyHooks(triggered=triggered, merlin_ds=merlin_ds),
            nav_shape=(32, 32),
        )
        async for res in ctx_pipelined.run_udf_iter(dataset=aq, udf=udf, sync=False):
            pass

    assert triggered[0]

    assert_allclose(res.buffers[0]['intensity'], ref['intensity'])


class ValidationUDF(UDF):
    def __init__(self, ref_da):
        super().__init__(ref_da=ref_da)

    def get_result_buffers(self):
        return {
            'nav': self.buffer(kind='nav'),
        }

    def process_partition(self, partition):
        ref_tile_data = self.params.ref_da[self.meta.coordinates.reshape((-1,))].compute()
        # assert False
        assert np.allclose(partition, ref_tile_data)


# use inline executor here to not use too much memory
def test_get_tiles_comparison(
    ltl_ctx,
    merlin_detector_sim_ptycho,
    merlin_control_sim_ptycho,
    merlin_ds_ptycho_flat
):
    merlin_ds = merlin_ds_ptycho_flat
    da, _ = make_dask_array(merlin_ds)
    p = next(merlin_ds.get_partitions())
    host, port = merlin_detector_sim_ptycho

    host, port = merlin_detector_sim_ptycho
    api_host, api_port = merlin_control_sim_ptycho
    with ltl_ctx.make_connection('merlin').open(
        data_host=host,
        data_port=port,
        api_host=api_host,
        api_port=api_port,
        drain=False,
    ) as conn:
        aq = ltl_ctx.make_acquisition(
            conn=conn,
            nav_shape=merlin_ds.shape.nav,
            frames_per_partition=p.slice.shape[0]
        )
        _ = TilingScheme.make_for_shape(
            tileshape=Shape((7, 256, 256), sig_dims=2),
            dataset_shape=aq.shape
        )

        ltl_ctx.run_udf(dataset=aq, udf=ValidationUDF(ref_da=da))


def test_acquisition_triggered_garbage(
    ctx_pipelined: LiveContext,
    merlin_control_sim,
    trigger_sim,
    garbage_sim,
    merlin_ds,
):
    pool = concurrent.futures.ThreadPoolExecutor(1)

    trig_res = {
        0: None
    }

    class _MyHooks(Hooks):
        def on_ready_for_data(self, env: ReadyForDataEnv):
            control = MerlinControl(*merlin_control_sim)
            with control:
                control.cmd('STARTACQUISITION')
            tr = TriggerClient(*trigger_sim)
            print("Trigger connection:", trigger_sim)
            tr.connect()
            tr.trigger()

            def do_scan():
                '''
                Emulated blocking scan function using the Merlin simulator
                '''
                print("do_scan()")

            fut = pool.submit(do_scan)
            trig_res[0] = fut
            tr.close()

    host, port = garbage_sim
    api_host, api_port = merlin_control_sim
    with ctx_pipelined.make_connection('merlin').open(
        data_host=host,
        data_port=port,
        api_host=api_host,
        api_port=api_port,
        drain=True,
    ) as conn:
        aq = ctx_pipelined.make_acquisition(
            conn=conn,
            hooks=_MyHooks(),
            nav_shape=(32, 32),
        )
        udf = SumUDF()

        res = ctx_pipelined.run_udf(dataset=aq, udf=udf)
        assert trig_res[0].result() is None

        ref = ctx_pipelined.run_udf(dataset=merlin_ds, udf=udf)

        assert_allclose(res['intensity'], ref['intensity'])


def test_acquisition_triggered_control(
    ctx_pipelined: "LiveContext",
    merlin_ds,
    merlin_triggered_sim_threads,
):
    pool = concurrent.futures.ThreadPoolExecutor(1)
    trig_res = {
        0: None
    }
    host, port = merlin_triggered_sim_threads.server_t.sockname
    api_host, api_port = merlin_triggered_sim_threads.control_t.sockname

    class _MyHooks(Hooks):
        def on_ready_for_data(self, env: ReadyForDataEnv):
            control = MerlinControl(host=api_host, port=api_port)
            with control:
                control.cmd('STARTACQUISITION')

            def do_scan():
                '''
                Emulated blocking scan function using the Merlin simulator
                '''
                print("do_scan()")
                with control:
                    control.cmd('SOFTTRIGGER')

            fut = pool.submit(do_scan)
            trig_res[0] = fut

    with ctx_pipelined.make_connection('merlin').open(
        data_host=host,
        data_port=port,
        api_host=api_host,
        api_port=api_port,
        drain=False,
    ) as conn:
        aq = ctx_pipelined.make_acquisition(
            conn=conn,
            hooks=_MyHooks(),
            nav_shape=(32, 32),
        )

        udf = SumUDF()

        res = ctx_pipelined.run_udf(dataset=aq, udf=udf)
        assert trig_res[0].result() is None

        ref = ctx_pipelined.run_udf(dataset=merlin_ds, udf=udf)

        assert_allclose(res['intensity'], ref['intensity'])


def test_control(merlin_control_sim, tmp_path):
    path = tmp_path / 'cmd.txt'
    with path.open('w') as f:
        f.write("SET,COUNTERDEPTH,12\n")
        f.write("\n")
        f.write("SET,RUNHEADLESS,1\n")
    c = MerlinControl(*merlin_control_sim)
    with c:
        c.set("NUMFRAMESTOACQUIRE", 23)
        c.cmd('STARTACQUISITION')
        assert c.get("NUMFRAMESTOACQUIRE") == b'23'
        c.send_command_file(path)
        assert c.get("COUNTERDEPTH") == b'12'


@pytest.mark.parametrize(
    ['shape_hint', 'expected'],
    [
        (None, (32, 32)),
        ((-1, -1), (32, 32)),
        ((16, -1), (16, 64)),
        ((-1, 16, 16), (4, 16, 16)),
        ((4, -1, -1), (4, 16, 16)),
    ]
)
def test_passive_auto_nav_shape(
    ctx_pipelined: LiveContext,
    default_conn,

    shape_hint: Optional[tuple[int, ...]],
    expected: tuple[int, ...],
):
    pending_aq = default_conn.wait_for_acquisition(10.0)
    assert pending_aq is not None

    aq = ctx_pipelined.make_acquisition(
        conn=default_conn,
        pending_aq=pending_aq,
        nav_shape=shape_hint,
    )
    assert tuple(aq.shape.nav) == expected
    _ = ctx_pipelined.run_udf(dataset=aq, udf=SumUDF())


def test_passive_trigger_multi(
    ctx_pipelined: LiveContext,
    merlin_ds,
    merlin_triggered_sim_threads,
):
    host, port = merlin_triggered_sim_threads.server_t.sockname
    api_host, api_port = merlin_triggered_sim_threads.control_t.sockname
    with ctx_pipelined.make_connection('merlin').open(
        data_host=host,
        data_port=port,
        api_host=api_host,
        api_port=api_port,
        drain=False,
    ) as conn_triggered:
        udf = SumUDF()
        ref = ctx_pipelined.run_udf(dataset=merlin_ds, udf=udf)

        trigger_host, trigger_port = merlin_triggered_sim_threads.trigger_t.sockname
        tr = TriggerClient(host=trigger_host, port=trigger_port)

        for i in range(3):
            print(f"Arming detector... {i}")
            control = MerlinControl(host=api_host, port=api_port)
            with control:
                control.cmd('STARTACQUISITION')
            print(f"triggering {i}")
            with tr:
                tr.trigger()
            pending_aq = conn_triggered.wait_for_acquisition(10)
            assert pending_aq is not None
            print(f"starting {i}")
            aq = ctx_pipelined.make_acquisition(
                conn=conn_triggered,
                nav_shape=(32, 32),
                pending_aq=pending_aq,
            )
            res = ctx_pipelined.run_udf(dataset=aq, udf=udf)
            assert_allclose(res['intensity'].data, ref['intensity'].data)
            print(f"done {i}")
            import time
            time.sleep(1)
        tr.close()
