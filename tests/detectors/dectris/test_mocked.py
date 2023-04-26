import pytest

from libertem_live.api import LiveContext
from libertem.udf.sum import SumUDF

from utils import run_camera_sim


def run_dectris_sim(*args, **kwargs):
    from libertem_live.detectors.dectris.sim import DectrisSim
    return run_camera_sim(
        cls=DectrisSim, verbose=True, port=0, zmqport=0,
        tolerate_timeouts=False,
        num_mock_frames=32*32,
        **kwargs
    )


@pytest.fixture(scope='module')
def dectris_runner():
    yield from run_dectris_sim(dwelltime=50)


@pytest.fixture(scope='module')
def dectris_sim(dectris_runner):
    '''
    port, zmqport tuple of the simulator
    '''
    return (dectris_runner.port, dectris_runner.zmqport)


def test_udf_sig(ctx_pipelined: LiveContext, dectris_sim):
    api_port, data_port = dectris_sim

    conn = ctx_pipelined.make_connection('dectris').open(
        api_host='127.0.0.1',
        api_port=api_port,
        data_host='127.0.0.1',
        data_port=data_port,
    )

    try:
        aq = ctx_pipelined.make_acquisition(
            conn=conn,
            nav_shape=(32, 32),
            frames_per_partition=64,
            controller=conn.get_active_controller(
                trigger_mode='exte',
            ),
        )
        udf = SumUDF()
        ctx_pipelined.run_udf(dataset=aq, udf=udf)
    finally:
        conn.close()


def test_passive_acquisition(ctx_pipelined: LiveContext, dectris_sim):
    api_port, data_port = dectris_sim
    conn = ctx_pipelined.make_connection('dectris').open(
        api_host='127.0.0.1',
        api_port=api_port,
        data_host='127.0.0.1',
        data_port=data_port,
    )

    try:
        # this can happen wherever, maybe from another computer in the network:
        ec = conn.get_api_client()
        ec.sendDetectorCommand('arm')

        pending_aq = conn.wait_for_acquisition(10.0)
        assert pending_aq is not None
        assert pending_aq.series == 34
        assert pending_aq.detector_config is not None
        assert pending_aq.nimages == 32 * 32

        aq = ctx_pipelined.make_acquisition(
            pending_aq=pending_aq,
            conn=conn,
            nav_shape=(32, 32),
            frames_per_partition=64,
        )

        # FIXME verify result
        _ = ctx_pipelined.run_udf(dataset=aq, udf=SumUDF())
    finally:
        conn.close()


def test_context_manager_multi_block(ctx_pipelined: LiveContext, dectris_sim):
    api_port, data_port = dectris_sim
    conn = ctx_pipelined.make_connection('dectris').open(
        api_host='127.0.0.1',
        api_port=api_port,
        data_host='127.0.0.1',
        data_port=data_port,
    )
    with conn:
        # this can happen wherever, maybe from another computer in the network:
        ec = conn.get_api_client()
        ec.sendDetectorCommand('arm')

        pending_aq = conn.wait_for_acquisition(10.0)
        assert pending_aq is not None

        aq = ctx_pipelined.make_acquisition(
            pending_aq=pending_aq,
            conn=conn,
            nav_shape=(32, 32),
            frames_per_partition=64,
        )

        _ = ctx_pipelined.run_udf(dataset=aq, udf=SumUDF())

    assert conn._conn is None

    with conn:
        # this can happen wherever, maybe from another computer in the network:
        ec = conn.get_api_client()
        ec.sendDetectorCommand('arm')

        pending_aq = conn.wait_for_acquisition(10.0)
        assert pending_aq is not None

        aq = ctx_pipelined.make_acquisition(
            pending_aq=pending_aq,
            conn=conn,
            nav_shape=(32, 32),
            frames_per_partition=64,
        )

        _ = ctx_pipelined.run_udf(dataset=aq, udf=SumUDF())

    assert conn._conn is None
