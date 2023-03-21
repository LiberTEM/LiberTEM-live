import os
import sys
import sparse

from libertem.udf.sumsigudf import SumSigUDF
from libertem.udf.sum import SumUDF
from libertem.executor.pipelined import PipelinedExecutor
import pytest
from libertem_live.api import LiveContext
from libertem_live.detectors.dectris.acquisition import (
    DectrisAcquisition, DectrisDetectorConnection,
)
from libertem.io.corrections import CorrectionSet
import libertem

from utils import get_testdata_path, run_camera_sim


DECTRIS_TESTDATA_PATH = os.path.join(
    get_testdata_path(),
    'dectris', 'zmqdump.dat.128x128-id34-exte-bslz4'
)
HAVE_DECTRIS_TESTDATA = os.path.exists(DECTRIS_TESTDATA_PATH)

pytestmark = pytest.mark.skipif(sys.version_info < (3, 7),
                                reason="DECTRIS support requires Python 3.7")


def run_dectris_sim(*args, path=DECTRIS_TESTDATA_PATH, **kwargs):
    from libertem_live.detectors.dectris.sim import DectrisSim
    return run_camera_sim(cls=DectrisSim, path=path, port=0, zmqport=0, **kwargs)


@pytest.fixture(scope='module')
def dectris_runner():
    yield from run_dectris_sim()


@pytest.fixture(scope='module')
def dectris_sim(dectris_runner):
    '''
    port, zmqport tuple of the simulator
    '''
    return (dectris_runner.port, dectris_runner.zmqport)


@pytest.fixture()
def skipped_dectris_runner():

    item_counter = {0: 0}

    def skip(data):
        result = None
        # print(data[:10])
        if item_counter[0] not in (10, 11, 12, 13):
            result = data
        item_counter[0] += 1
        return result

    yield from run_dectris_sim(data_filter=skip)


@pytest.fixture()
def skipped_dectris_sim(skipped_dectris_runner):
    '''
    port, zmqport tuple of the simulator
    '''
    return (skipped_dectris_runner.port, skipped_dectris_runner.zmqport)


@pytest.mark.skipif(not HAVE_DECTRIS_TESTDATA, reason="need DECTRIS testdata")
@pytest.mark.data
def test_udf_sig(ctx_pipelined, dectris_sim):
    api_port, data_port = dectris_sim

    conn = DectrisDetectorConnection(
        api_host='127.0.0.1',
        api_port=api_port,
        data_host='127.0.0.1',
        data_port=data_port,
        num_slots=2000,
        bytes_per_frame=512*512,
    )

    aq = DectrisAcquisition(
        conn=conn,
        nav_shape=(128, 128),
        trigger=lambda aq: None,
        frames_per_partition=512,
        controller=conn.get_active_controller(trigger_mode='exte'),
    )
    aq.initialize(ctx_pipelined.executor)

    bad_y = (168, 291, 326, 301, 343, 292,   0,   0,   0,   0,   0, 511)
    bad_x = (496, 458, 250, 162, 426, 458, 393, 396, 413, 414, 342, 491)

    corr = CorrectionSet(
        excluded_pixels=sparse.COO(
            coords=(bad_y, bad_x),
            data=1,
            shape=aq.shape.sig
        )
    )

    udf = SumUDF()
    ctx_pipelined.run_udf(dataset=aq, udf=udf, corrections=corr)

    conn.close()


@pytest.mark.skipif(not HAVE_DECTRIS_TESTDATA, reason="need DECTRIS testdata")
@pytest.mark.data
def test_udf_nav(ctx_pipelined, dectris_sim):
    api_port, data_port = dectris_sim
    conn = DectrisDetectorConnection(
        api_host='127.0.0.1',
        api_port=api_port,
        data_host='127.0.0.1',
        data_port=data_port,
        num_slots=2000,
        bytes_per_frame=512*512,
    )
    aq = DectrisAcquisition(
        conn=conn,
        nav_shape=(128, 128),
        trigger=lambda aq: None,
        frames_per_partition=512,
        controller=conn.get_active_controller(trigger_mode='exte'),
    )
    aq.initialize(ctx_pipelined.executor)
    udf = SumSigUDF()
    ctx_pipelined.run_udf(dataset=aq, udf=udf)
    conn.close()


# TODO: pytest.mark.slow? this test is mostly useful for debugging
@pytest.mark.skipif(not HAVE_DECTRIS_TESTDATA, reason="need DECTRIS testdata")
@pytest.mark.data
def test_udf_nav_inline(ltl_ctx, dectris_sim):
    api_port, data_port = dectris_sim
    from libertem.common.tracing import maybe_setup_tracing
    maybe_setup_tracing("test_udf_nav_inline")
    conn = DectrisDetectorConnection(
        api_host='127.0.0.1',
        api_port=api_port,
        data_host='127.0.0.1',
        data_port=data_port,
        num_slots=2000,
        bytes_per_frame=512*512,
    )
    aq = DectrisAcquisition(
        conn=conn,
        nav_shape=(128, 128),
        trigger=lambda aq: None,
        frames_per_partition=32,
        controller=conn.get_active_controller(trigger_mode='exte'),
    )
    aq.initialize(ltl_ctx.executor)
    udf = SumSigUDF()
    ltl_ctx.run_udf(dataset=aq, udf=udf)
    conn.close()


@pytest.mark.skipif(not HAVE_DECTRIS_TESTDATA, reason="need DECTRIS testdata")
@pytest.mark.data
def test_sum(ctx_pipelined, dectris_sim):
    api_port, data_port = dectris_sim
    conn = DectrisDetectorConnection(
        api_host='127.0.0.1',
        api_port=api_port,
        data_host='127.0.0.1',
        data_port=data_port,
        num_slots=2000,
        bytes_per_frame=512*512,
    )
    aq = ctx_pipelined.prepare_acquisition(
        'dectris',
        conn=conn,
        nav_shape=(128, 128),
        trigger=lambda aq: None,
        frames_per_partition=32,
        controller=conn.get_active_controller(trigger_mode='exte'),
    )
    # FIXME verify result
    _ = ctx_pipelined.run_udf(dataset=aq, udf=SumUDF())
    conn.close()


@pytest.mark.skipif(not HAVE_DECTRIS_TESTDATA, reason="need DECTRIS testdata")
@pytest.mark.data
def test_passive_acquisition(ctx_pipelined, dectris_sim):
    api_port, data_port = dectris_sim
    conn = DectrisDetectorConnection(
        api_host='127.0.0.1',
        api_port=api_port,
        data_host='127.0.0.1',
        data_port=data_port,
        num_slots=2000,
        bytes_per_frame=512*512,
    )

    # this can happen wherever, maybe from another computer in the network:
    ec = conn.get_api_client()
    ec.sendDetectorCommand('arm')

    pending_aq = conn.wait_for_acquisition(10.0)
    assert pending_aq is not None
    assert pending_aq.series == 34
    assert pending_aq.detector_config is not None
    # assert pending_aq.detector_config.nimages == 1
    # assert pending_aq.detector_config.ntrigger == 128 * 128
    # assert pending_aq.detector_config.x_pixels_in_detector == 512
    # assert pending_aq.detector_config.y_pixels_in_detector == 512
    # assert pending_aq.detector_config.bit_depth_image == 16

    aq = ctx_pipelined.prepare_from_pending(
        conn=conn,
        nav_shape=(128, 128),
        trigger=lambda aq: None,
        frames_per_partition=32,
        pending_acquisition=pending_aq,
    )

    # FIXME verify result
    _ = ctx_pipelined.run_udf(dataset=aq, udf=SumUDF())
    conn.close()


@pytest.mark.skipif(not HAVE_DECTRIS_TESTDATA, reason="need DECTRIS testdata")
@pytest.mark.data
def test_passive_timeout(dectris_sim):
    api_port, data_port = dectris_sim
    conn = DectrisDetectorConnection(
        api_host='127.0.0.1',
        api_port=api_port,
        data_host='127.0.0.1',
        data_port=data_port,
        num_slots=2000,
        bytes_per_frame=512*512,
    )

    # if we don't arm the detector externally, nothing happens
    # and we run into a timeout:
    pending_aq = conn.wait_for_acquisition(1.0)
    assert pending_aq is None


@pytest.mark.skipif(not HAVE_DECTRIS_TESTDATA, reason="need DECTRIS testdata")
@pytest.mark.data
@pytest.mark.timeout(120)  # May lock up because of an executor bug
def test_frame_skip(skipped_dectris_sim, dectris_sim):
    # Only newer 0.10.0 because of a bug in the PipelinedExecutor
    _version_bits = libertem.__version__.split('.')
    version_tuple = tuple(int(b) for b in _version_bits[:3])
    if version_tuple <= (0, 10, 0):
        pytest.skip(reason="LiberTEM version too old")

    # uses its own executor to not potentially bring
    # the `ctx_pipelined` executor into a bad state
    executor = None
    conn = None
    try:
        api_port, data_port = skipped_dectris_sim
        conn = DectrisDetectorConnection(
            api_host='127.0.0.1',
            api_port=api_port,
            data_host='127.0.0.1',
            data_port=data_port,
            num_slots=2000,
            bytes_per_frame=512*512,
        )
        executor = PipelinedExecutor(
            spec=PipelinedExecutor.make_spec(cpus=range(4), cudas=[]),
            # to prevent issues in already-pinned situations (i.e. containerized
            # environments), don't pin our worker processes in testing:
            pin_workers=False,
            cleanup_timeout=0.5,
        )
        aq = DectrisAcquisition(
            conn=conn,
            nav_shape=(128, 128),
            trigger=lambda aq: None,
            frames_per_partition=32,
            controller=conn.get_active_controller(trigger_mode='exte'),
        )
        ctx = LiveContext(executor=executor)
        aq.initialize(ctx.executor)
        # Originally an AssertionError, but may cause downstream issues
        # in the executor, TODO revisit after some time if executor behavior changed
        with pytest.raises(Exception):
            _ = ctx.run_udf(dataset=aq, udf=SumUDF())
        # Ensure the executor is still alive
        api_port, data_port = dectris_sim
        aq2 = DectrisAcquisition(
            conn=conn,
            nav_shape=(128, 128),
            trigger=lambda aq: None,
            frames_per_partition=32,
            controller=conn.get_active_controller(trigger_mode='exte'),
        )
        aq2.initialize(ctx.executor)
        _ = ctx.run_udf(dataset=aq2, udf=SumUDF())
    finally:
        if executor is not None:
            executor.close()
        if conn is not None:
            conn.close()
