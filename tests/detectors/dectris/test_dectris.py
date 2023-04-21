import os
from typing import Optional, Tuple

import numpy as np
import sparse

from libertem.udf.sumsigudf import SumSigUDF
from libertem.udf.sum import SumUDF
from libertem.executor.pipelined import PipelinedExecutor
import pytest
from libertem_live.api import LiveContext, Hooks
from libertem_live.hooks import DetermineNavShapeEnv
from libertem.io.corrections import CorrectionSet

from utils import get_testdata_path, run_camera_sim


DECTRIS_TESTDATA_PATH = os.path.join(
    get_testdata_path(),
    'dectris', 'zmqdump.dat.128x128-id34-exte-bslz4'
)
HAVE_DECTRIS_TESTDATA = os.path.exists(DECTRIS_TESTDATA_PATH)

pytestmark = [
    pytest.mark.data,
]


def run_dectris_sim(*args, path=DECTRIS_TESTDATA_PATH, **kwargs):
    from libertem_live.detectors.dectris.sim import DectrisSim
    return run_camera_sim(
        cls=DectrisSim, verbose=True, path=path, port=0, zmqport=0,
        tolerate_timeouts=False,
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
            nav_shape=(128, 128),
            frames_per_partition=512,
            controller=conn.get_active_controller(
                trigger_mode='exte',
            ),
        )

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
    finally:
        conn.close()


@pytest.mark.skipif(not HAVE_DECTRIS_TESTDATA, reason="need DECTRIS testdata")
@pytest.mark.data
def test_conn_parameters(ctx_pipelined: LiveContext, dectris_sim):
    api_port, data_port = dectris_sim

    conn = ctx_pipelined.make_connection('dectris').open(
        api_host='127.0.0.1',
        api_port=api_port,
        data_host='127.0.0.1',
        data_port=data_port,
        buffer_size=2048,
        bytes_per_frame=512*512//8,
        frame_stack_size=64,
        huge_pages=False,
    )
    try:
        aq = ctx_pipelined.make_acquisition(
            conn=conn,
            nav_shape=(128, 128),
            frames_per_partition=512,
            controller=conn.get_active_controller(trigger_mode='exte'),
        )

        udf = SumUDF()
        ctx_pipelined.run_udf(dataset=aq, udf=udf)
    finally:
        conn.close()


@pytest.mark.skipif(not HAVE_DECTRIS_TESTDATA, reason="need DECTRIS testdata")
@pytest.mark.data
def test_udf_nav(ctx_pipelined: LiveContext, dectris_sim):
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
            nav_shape=(128, 128),
            frames_per_partition=512,
            controller=conn.get_active_controller(trigger_mode='exte'),
        )

        udf = SumSigUDF()
        ctx_pipelined.run_udf(dataset=aq, udf=udf)
    finally:
        conn.close()


# TODO: pytest.mark.slow? this test is mostly useful for debugging
@pytest.mark.skipif(not HAVE_DECTRIS_TESTDATA, reason="need DECTRIS testdata")
@pytest.mark.data
def test_udf_nav_inline(ltl_ctx: LiveContext, dectris_sim):
    api_port, data_port = dectris_sim
    from libertem.common.tracing import maybe_setup_tracing
    maybe_setup_tracing("test_udf_nav_inline")
    conn = ltl_ctx.make_connection('dectris').open(
        api_host='127.0.0.1',
        api_port=api_port,
        data_host='127.0.0.1',
        data_port=data_port,
    )
    try:
        aq = ltl_ctx.make_acquisition(
            conn=conn,
            nav_shape=(128, 128),
            frames_per_partition=32,
            controller=conn.get_active_controller(trigger_mode='exte'),
        )
        udf = SumSigUDF()
        ltl_ctx.run_udf(dataset=aq, udf=udf)
    finally:
        conn.close()


@pytest.mark.skipif(not HAVE_DECTRIS_TESTDATA, reason="need DECTRIS testdata")
@pytest.mark.data
def test_sum(ctx_pipelined: LiveContext, dectris_sim):
    api_port, data_port = dectris_sim
    conn = ctx_pipelined.make_connection(
        'dectris',
    ).open(
        api_host='127.0.0.1',
        api_port=api_port,
        data_host='127.0.0.1',
        data_port=data_port,
    )
    try:
        aq = ctx_pipelined.make_acquisition(
            conn=conn,
            nav_shape=(128, 128),
            frames_per_partition=512,
            controller=conn.get_active_controller(trigger_mode='exte'),
        )
        # FIXME verify result
        _ = ctx_pipelined.run_udf(dataset=aq, udf=SumUDF())
    finally:
        conn.close()


@pytest.mark.skipif(not HAVE_DECTRIS_TESTDATA, reason="need DECTRIS testdata")
@pytest.mark.data
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
        assert pending_aq.nimages == 128 * 128
        # assert pending_aq.detector_config.x_pixels_in_detector == 512
        # assert pending_aq.detector_config.y_pixels_in_detector == 512
        # assert pending_aq.detector_config.bit_depth_image == 16

        aq = ctx_pipelined.make_acquisition(
            pending_aq=pending_aq,
            conn=conn,
            nav_shape=(128, 128),
            frames_per_partition=512,
        )

        # FIXME verify result
        _ = ctx_pipelined.run_udf(dataset=aq, udf=SumUDF())
    finally:
        conn.close()


@pytest.mark.skipif(not HAVE_DECTRIS_TESTDATA, reason="need DECTRIS testdata")
@pytest.mark.data
def test_passive_reconnect(ctx_pipelined: LiveContext, dectris_sim):
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

        pending_aq = conn.wait_for_acquisition(2.0)
        assert pending_aq is not None

        aq = ctx_pipelined.make_acquisition(
            pending_aq=pending_aq,
            conn=conn,
            nav_shape=(128, 128),
            frames_per_partition=512,
        )

        _ = ctx_pipelined.run_udf(dataset=aq, udf=SumUDF())
    finally:
        conn.close()

    conn.reconnect()

    try:
        # this can happen wherever, maybe from another computer in the network:
        ec = conn.get_api_client()
        ec.sendDetectorCommand('arm')

        pending_aq = conn.wait_for_acquisition(2.0)
        assert pending_aq is not None

        aq = ctx_pipelined.make_acquisition(
            pending_aq=pending_aq,
            conn=conn,
            nav_shape=(128, 128),
            frames_per_partition=512,
        )

        _ = ctx_pipelined.run_udf(dataset=aq, udf=SumUDF())
    finally:
        conn.close()


@pytest.mark.skipif(not HAVE_DECTRIS_TESTDATA, reason="need DECTRIS testdata")
@pytest.mark.data
def test_context_manager(ctx_pipelined: LiveContext, dectris_sim):
    api_port, data_port = dectris_sim
    with ctx_pipelined.make_connection('dectris').open(
        api_host='127.0.0.1',
        api_port=api_port,
        data_host='127.0.0.1',
        data_port=data_port,
    ) as conn:
        # this can happen wherever, maybe from another computer in the network:
        ec = conn.get_api_client()
        ec.sendDetectorCommand('arm')

        pending_aq = conn.wait_for_acquisition(10.0)
        assert pending_aq is not None

        aq = ctx_pipelined.make_acquisition(
            pending_aq=pending_aq,
            conn=conn,
            nav_shape=(128, 128),
            frames_per_partition=512,
        )

        _ = ctx_pipelined.run_udf(dataset=aq, udf=SumUDF())


@pytest.mark.skipif(not HAVE_DECTRIS_TESTDATA, reason="need DECTRIS testdata")
@pytest.mark.data
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
            nav_shape=(128, 128),
            frames_per_partition=512,
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
            nav_shape=(128, 128),
            frames_per_partition=512,
        )

        _ = ctx_pipelined.run_udf(dataset=aq, udf=SumUDF())

    assert conn._conn is None


@pytest.mark.skipif(not HAVE_DECTRIS_TESTDATA, reason="need DECTRIS testdata")
@pytest.mark.data
def test_passive_timeout(dectris_sim):
    from libertem_live.detectors.dectris import DectrisDetectorConnection
    api_port, data_port = dectris_sim
    conn = DectrisDetectorConnection(
        api_host='127.0.0.1',
        api_port=api_port,
        data_host='127.0.0.1',
        data_port=data_port,
    )

    try:
        # if we don't arm the detector externally, nothing happens
        # and we run into a timeout:
        pending_aq = conn.wait_for_acquisition(0.5)
        assert pending_aq is None
    finally:
        conn.close()


@pytest.mark.skipif(not HAVE_DECTRIS_TESTDATA, reason="need DECTRIS testdata")
@pytest.mark.data
@pytest.mark.timeout(120)  # May lock up because of an executor bug
def test_frame_skip(skipped_dectris_sim, dectris_sim):
    # uses its own executor to not potentially bring
    # the `ctx_pipelined` executor into a bad state
    executor = None
    conn = None
    try:
        api_port, data_port = skipped_dectris_sim
        executor = PipelinedExecutor(
            spec=PipelinedExecutor.make_spec(cpus=range(4), cudas=[]),
            # to prevent issues in already-pinned situations (i.e. containerized
            # environments), don't pin our worker processes in testing:
            pin_workers=False,
            cleanup_timeout=0.5,
        )
        ctx = LiveContext(executor=executor)
        conn = ctx.make_connection('dectris').open(
            api_host='127.0.0.1',
            api_port=api_port,
            data_host='127.0.0.1',
            data_port=data_port,
        )
        aq = ctx.make_acquisition(
            conn=conn,
            nav_shape=(128, 128),
            frames_per_partition=512,
            controller=conn.get_active_controller(trigger_mode='exte'),
        )

        # Originally an AssertionError, but may cause downstream issues
        # in the executor, TODO revisit after some time if executor behavior changed
        with pytest.raises(Exception):
            _ = ctx.run_udf(dataset=aq, udf=SumUDF())
        # Ensure the executor is still alive
        api_port, data_port = dectris_sim
        aq2 = ctx.make_acquisition(
            conn=conn,
            nav_shape=(128, 128),
            frames_per_partition=512,
            controller=conn.get_active_controller(trigger_mode='exte'),
        )
        _ = ctx.run_udf(dataset=aq2, udf=SumUDF())
    finally:
        if executor is not None:
            executor.close()
        if conn is not None:
            conn.close()


@pytest.mark.skipif(not HAVE_DECTRIS_TESTDATA, reason="need DECTRIS testdata")
@pytest.mark.data
def test_hooks(ctx_pipelined: LiveContext, dectris_sim):
    api_port, data_port = dectris_sim
    conn = ctx_pipelined.make_connection('dectris').open(
        api_host='127.0.0.1',
        api_port=api_port,
        data_host='127.0.0.1',
        data_port=data_port,
    )

    class _MyHooks(Hooks):
        def __init__(self):
            self._ready_called = False

        def on_ready_for_data(self, env):
            self._ready_called = True

    with conn:
        hook_instance = _MyHooks()
        aq = ctx_pipelined.make_acquisition(
            conn=conn,
            nav_shape=(128, 128),
            frames_per_partition=512,
            hooks=hook_instance,
        )

        assert not hook_instance._ready_called
        for res in ctx_pipelined.run_udf_iter(dataset=aq, udf=SumUDF()):
            assert hook_instance._ready_called


@pytest.mark.skipif(not HAVE_DECTRIS_TESTDATA, reason="need DECTRIS testdata")
@pytest.mark.data
def test_concrete_nav_shape(ctx_pipelined: LiveContext, dectris_sim):
    api_port, data_port = dectris_sim
    conn = ctx_pipelined.make_connection('dectris').open(
        api_host='127.0.0.1',
        api_port=api_port,
        data_host='127.0.0.1',
        data_port=data_port,
    )

    try:
        ec = conn.get_api_client()
        ec.sendDetectorCommand('arm')

        pending_aq = conn.wait_for_acquisition(10.0)
        assert pending_aq is not None

        aq = ctx_pipelined.make_acquisition(
            pending_aq=pending_aq,
            conn=conn,
            # non-square to make sure we hit the right test case:
            nav_shape=(256, 64),
        )
        assert tuple(aq.shape.nav) == (256, 64)
        # run a UDF to "drain" the connection:
        _ = ctx_pipelined.run_udf(dataset=aq, udf=SumUDF())
    finally:
        conn.close()


@pytest.mark.skipif(not HAVE_DECTRIS_TESTDATA, reason="need DECTRIS testdata")
@pytest.mark.data
def test_shape_from_hook(ctx_pipelined: LiveContext, dectris_sim):
    api_port, data_port = dectris_sim
    conn = ctx_pipelined.make_connection('dectris').open(
        api_host='127.0.0.1',
        api_port=api_port,
        data_host='127.0.0.1',
        data_port=data_port,
    )

    class _MyHooks(Hooks):
        def on_determine_nav_shape(self, env: DetermineNavShapeEnv) -> Optional[Tuple[int, ...]]:
            return (64, 256)

    try:
        ec = conn.get_api_client()
        ec.sendDetectorCommand('arm')

        pending_aq = conn.wait_for_acquisition(10.0)
        assert pending_aq is not None

        hook_instance = _MyHooks()

        aq = ctx_pipelined.make_acquisition(
            pending_aq=pending_aq,
            conn=conn,
            hooks=hook_instance,
        )
        assert tuple(aq.shape.nav) == (64, 256)
        # run a UDF to "drain" the connection:
        _ = ctx_pipelined.run_udf(dataset=aq, udf=SumUDF())
    finally:
        conn.close()


@pytest.mark.skipif(not HAVE_DECTRIS_TESTDATA, reason="need DECTRIS testdata")
@pytest.mark.data
def test_shape_bad_hook(ctx_pipelined: LiveContext, dectris_sim):
    api_port, data_port = dectris_sim
    conn = ctx_pipelined.make_connection('dectris').open(
        api_host='127.0.0.1',
        api_port=api_port,
        data_host='127.0.0.1',
        data_port=data_port,
    )

    class _MyHooks(Hooks):
        def on_determine_nav_shape(self, env: DetermineNavShapeEnv) -> Optional[Tuple[int, ...]]:
            return (256, 256)  # that's too much for the number of images in the acquisition!

    try:
        ec = conn.get_api_client()
        ec.sendDetectorCommand('arm')

        pending_aq = conn.wait_for_acquisition(10.0)
        assert pending_aq is not None

        hook_instance = _MyHooks()

        with pytest.raises(ValueError) as m:
            aq = ctx_pipelined.make_acquisition(
                pending_aq=pending_aq,
                conn=conn,
                hooks=hook_instance,
            )

        m.match(r'^Result.* is not compatible with number of images \(16384\)$')

        # again, without the bad hooks:
        aq = ctx_pipelined.make_acquisition(
            pending_aq=pending_aq,
            conn=conn,
        )
        # run a UDF to "drain" the connection:
        _ = ctx_pipelined.run_udf(dataset=aq, udf=SumUDF())
    finally:
        conn.close()


@pytest.mark.parametrize(
    ['shape_hint', 'expected'],
    [
        ((-1, -1), (128, 128)),
        ((64, -1), (64, 256)),
        ((-1, 64, 64), (4, 64, 64)),
        ((4, -1, -1), (4, 64, 64)),
    ]
)
@pytest.mark.skipif(not HAVE_DECTRIS_TESTDATA, reason="need DECTRIS testdata")
@pytest.mark.data
def test_shape_with_placeholders(
    ctx_pipelined: LiveContext,
    dectris_sim,
    shape_hint: Tuple[int, ...],
    expected: Tuple[int, ...],
):
    api_port, data_port = dectris_sim
    conn = ctx_pipelined.make_connection('dectris').open(
        api_host='127.0.0.1',
        api_port=api_port,
        data_host='127.0.0.1',
        data_port=data_port,
    )

    try:
        ec = conn.get_api_client()
        ec.sendDetectorCommand('arm')

        pending_aq = conn.wait_for_acquisition(10.0)
        assert pending_aq is not None

        aq = ctx_pipelined.make_acquisition(
            pending_aq=pending_aq,
            conn=conn,
            nav_shape=shape_hint,
        )
        assert tuple(aq.shape.nav) == expected
        # run a UDF to "drain" the connection:
        _ = ctx_pipelined.run_udf(dataset=aq, udf=SumUDF())
    finally:
        conn.close()


@pytest.mark.skipif(not HAVE_DECTRIS_TESTDATA, reason="need DECTRIS testdata")
@pytest.mark.data
def test_shape_with_placeholders_2d(ctx_pipelined: LiveContext, dectris_sim):
    pass


@pytest.mark.skipif(not HAVE_DECTRIS_TESTDATA, reason="need DECTRIS testdata")
@pytest.mark.data
def test_shape_from_controller(ctx_pipelined: LiveContext, dectris_sim):
    pass


@pytest.mark.skipif(not HAVE_DECTRIS_TESTDATA, reason="need DECTRIS testdata")
@pytest.mark.data
def test_square_shape_success(ctx_pipelined: LiveContext, dectris_sim):
    pass


@pytest.mark.skipif(not HAVE_DECTRIS_TESTDATA, reason="need DECTRIS testdata")
@pytest.mark.data
def test_need_shape_in_active_mode(ctx_pipelined: LiveContext, dectris_sim):

    api_port, data_port = dectris_sim
    conn = ctx_pipelined.make_connection('dectris').open(
        api_host='127.0.0.1',
        api_port=api_port,
        data_host='127.0.0.1',
        data_port=data_port,
    )

    try:
        with pytest.raises(RuntimeError) as m:
            aq = ctx_pipelined.make_acquisition(
                conn=conn,
            )

        m.match(r'In active mode, please pass the full `nav_shape')

        # again, with a shape,
        aq = ctx_pipelined.make_acquisition(
            conn=conn,
            nav_shape=(128, 128),
        )
        # run a UDF to "drain" the connection:
        _ = ctx_pipelined.run_udf(dataset=aq, udf=SumUDF())
    finally:
        conn.close()


@pytest.mark.timeout(120)  # may wait forever in case of issues
@pytest.mark.skipif(not HAVE_DECTRIS_TESTDATA, reason="need DECTRIS testdata")
@pytest.mark.data
def test_timeout_none(ctx_pipelined: LiveContext, dectris_sim):
    api_port, data_port = dectris_sim
    conn = ctx_pipelined.make_connection('dectris').open(
        api_host='127.0.0.1',
        api_port=api_port,
        data_host='127.0.0.1',
        data_port=data_port,
    )

    try:
        ec = conn.get_api_client()
        ec.sendDetectorCommand('arm')

        # should wait indefinitely:
        pending_aq = conn.wait_for_acquisition()
        aq = ctx_pipelined.make_acquisition(
            pending_aq=pending_aq,
            conn=conn,
            nav_shape=(128, 128),
            frames_per_partition=512,
        )
        _ = ctx_pipelined.run_udf(dataset=aq, udf=SumUDF())
    finally:
        conn.close()


@pytest.mark.timeout(120)
@pytest.mark.skipif(not HAVE_DECTRIS_TESTDATA, reason="need DECTRIS testdata")
@pytest.mark.data
def test_active_after_passive_mode(ctx_pipelined: LiveContext, dectris_sim):
    api_port, data_port = dectris_sim
    conn = ctx_pipelined.make_connection('dectris').open(
        api_host='127.0.0.1',
        api_port=api_port,
        data_host='127.0.0.1',
        data_port=data_port,
    )

    try:
        ec = conn.get_api_client()
        ec.sendDetectorCommand('arm')

        # passive:
        pending_aq = conn.wait_for_acquisition(1.0)
        aq = ctx_pipelined.make_acquisition(
            pending_aq=pending_aq,
            conn=conn,
            nav_shape=(128, 128),
            frames_per_partition=512,
        )
        res1 = ctx_pipelined.run_udf(dataset=aq, udf=SumUDF())

        # active:
        aq = ctx_pipelined.make_acquisition(
            conn=conn,
            nav_shape=(128, 128),
            frames_per_partition=512,
        )
        res2 = ctx_pipelined.run_udf(dataset=aq, udf=SumUDF())

        assert np.allclose(
            res1['intensity'].data,
            res2['intensity'].data,
        )
    finally:
        conn.close()
