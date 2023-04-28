import numpy as np
from numpy.testing import assert_allclose

from libertem.udf.sum import SumUDF
from libertem.io.dataset.base import DataSet

from libertem_live.api import Hooks, LiveContext
from libertem_live.hooks import ReadyForDataEnv
from libertem_live.detectors.merlin.control import MerlinControl
from libertem_live.detectors.merlin.sim import TriggerClient


class MyHooks(Hooks):
    def __init__(self, triggered: np.ndarray, merlin_ds: "DataSet"):
        self.ds = merlin_ds
        self.triggered = triggered

    def on_ready_for_data(self, env: ReadyForDataEnv):
        self.triggered[:] = True
        assert env.aq.shape.nav == self.ds.shape.nav


def test_acquisition(
    ctx_pipelined: LiveContext,
    mock_merlin_ds,
    mock_ds_conn,
):
    triggered = np.array((False,))

    aq = ctx_pipelined.make_acquisition(
        conn=mock_ds_conn,
        hooks=MyHooks(triggered=triggered, merlin_ds=mock_merlin_ds),
        nav_shape=mock_merlin_ds.shape.nav,
    )
    udf = SumUDF()

    assert not triggered[0]
    res = ctx_pipelined.run_udf(dataset=aq, udf=udf)
    assert triggered[0]

    ref = ctx_pipelined.run_udf(dataset=mock_merlin_ds, udf=udf)

    assert_allclose(res['intensity'], ref['intensity'])


def test_passive_acquisition(
    ctx_pipelined: LiveContext,
    mock_merlin_ds,
    mock_ds_conn,
):
    triggered = np.array((False,))

    pending_aq = mock_ds_conn.wait_for_acquisition(10.0)
    assert pending_aq is not None

    aq = ctx_pipelined.make_acquisition(
        conn=mock_ds_conn,
        hooks=MyHooks(triggered=triggered, merlin_ds=mock_merlin_ds),
        pending_aq=pending_aq,
        nav_shape=mock_merlin_ds.shape.nav,
    )
    udf = SumUDF()

    assert not triggered[0]
    res = ctx_pipelined.run_udf(dataset=aq, udf=udf)
    assert not triggered[0]  # in passive mode, we don't call the on_ready_for_data hook

    ref = ctx_pipelined.run_udf(dataset=mock_merlin_ds, udf=udf)

    assert_allclose(res['intensity'], ref['intensity'])


class TriggerHooks(Hooks):
    def __init__(self, control_t, trigger_t):
        self._control_t = control_t
        self._trigger_t = trigger_t

    def on_ready_for_data(self, env: ReadyForDataEnv):
        control = MerlinControl(*self._control_t)
        with control:
            control.cmd('STARTACQUISITION')

        with TriggerClient(*self._trigger_t) as tr:
            print("Trigger connection:", self._trigger_t)
            tr.trigger()


def test_acquisition_triggered_garbage(
    ctx_pipelined: LiveContext,
    mock_merlin_ds,
    mock_merlin_detector_sim_garbage,
    mock_merlin_trigger_sim_garbage,
    mock_merlin_control_sim_garbage,
):
    host, port = mock_merlin_detector_sim_garbage
    api_host, api_port = mock_merlin_control_sim_garbage
    with ctx_pipelined.make_connection('merlin').open(
        data_host=host,
        data_port=port,
        api_host=api_host,
        api_port=api_port,
        drain=True,
    ) as conn:
        aq = ctx_pipelined.make_acquisition(
            conn=conn,
            hooks=TriggerHooks(
                mock_merlin_control_sim_garbage,
                mock_merlin_trigger_sim_garbage
            ),
            nav_shape=mock_merlin_ds.shape.nav,
        )
        udf = SumUDF()

        res = ctx_pipelined.run_udf(dataset=aq, udf=udf)
        ref = ctx_pipelined.run_udf(dataset=mock_merlin_ds, udf=udf)
        assert_allclose(res['intensity'], ref['intensity'])
