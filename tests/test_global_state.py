import pytest

from libertem.udf.sum import SumUDF
from libertem_live.detectors.k2is.state import (
    ProcessingDoneEvent, StopProcessingEvent, cam_server_reducer, Store,
    LifecycleState, CamServerState, CamConnectionState, ProcessingState,
    StartupCompleteEvent, CamConnectedEvent, SetUDFsEvent, CamErrorEvent,
    CamDisconnectedEvent, SetNavShapeEvent, StartProcessingEvent,
)


@pytest.fixture
def store():
    initial_state = CamServerState(
        lifecycle=LifecycleState.STARTING,
        cam_connection=CamConnectionState.DISCONNECTED,
        processing=ProcessingState.IDLE,
        udfs=[],
        nav_shape=(),
        continuous=False,
        sectors_done=set(),
    )
    store = Store(reducer=cam_server_reducer, initial_state=initial_state)
    return store


def test_serial_inc(store):
    assert store.state.serial == 0
    store.dispatch(StartupCompleteEvent())
    assert store.state.serial == 1


def test_startup_complete(store):
    assert store.state.lifecycle == LifecycleState.STARTING
    store.dispatch(StartupCompleteEvent())
    assert store.state.lifecycle == LifecycleState.STARTUP_COMPLETE


def test_cam_connection(store):
    # startup doesn't change cam state:
    assert store.state.cam_connection == CamConnectionState.DISCONNECTED
    store.dispatch(StartupCompleteEvent())
    assert store.state.cam_connection == CamConnectionState.DISCONNECTED

    # but connecting does:
    store.dispatch(CamConnectedEvent())
    assert store.state.cam_connection == CamConnectionState.CONNECTED


def test_processing_state_initial(store):
    assert store.state.processing == ProcessingState.IDLE
    store.dispatch(StartupCompleteEvent())
    assert store.state.processing == ProcessingState.IDLE


def test_processing_start_1(store):
    store.dispatch(SetUDFsEvent(udfs=[SumUDF()]))
    store.dispatch(SetNavShapeEvent(nav_shape=(128, 128)))
    assert store.state.processing == ProcessingState.READY
    store.dispatch(StartProcessingEvent(continuous=False))
    assert store.state.processing == ProcessingState.RUNNING


def test_processing_start_2(store):
    store.dispatch(SetNavShapeEvent(nav_shape=(128, 128)))
    store.dispatch(SetUDFsEvent(udfs=[SumUDF()]))
    assert store.state.processing == ProcessingState.READY
    store.dispatch(StartProcessingEvent(continuous=False))
    assert store.state.processing == ProcessingState.RUNNING
    assert not store.state.continuous


def test_processing_start_continuous(store):
    store.dispatch(SetNavShapeEvent(nav_shape=(128, 128)))
    store.dispatch(SetUDFsEvent(udfs=[SumUDF()]))
    assert store.state.processing == ProcessingState.READY
    store.dispatch(StartProcessingEvent(continuous=True))
    assert store.state.processing == ProcessingState.RUNNING
    assert store.state.continuous


def test_procesing_done_sector_set(store):
    store.dispatch(SetNavShapeEvent(nav_shape=(128, 128)))
    store.dispatch(SetUDFsEvent(udfs=[SumUDF()]))
    store.dispatch(StartProcessingEvent(continuous=False))
    store.dispatch(ProcessingDoneEvent(idx=1))
    done = store.state.sectors_done
    assert len(done) == 1
    assert 1 in done

    # after stopping, the sectors_done set is kept the same:
    store.dispatch(StopProcessingEvent())
    assert store.state.processing == ProcessingState.READY
    assert len(store.state.sectors_done) == 1

    # only after starting processing again, the sectors_done set is cleared:
    store.dispatch(StartProcessingEvent(continuous=False))
    assert store.state.sectors_done == set()


def test_stop_processing_on_cam_error(store):
    store.dispatch(SetUDFsEvent(udfs=[SumUDF()]))
    store.dispatch(CamErrorEvent())
    assert store.state.processing == ProcessingState.IDLE


def test_stop_processing_on_cam_disconnect(store):
    store.dispatch(SetUDFsEvent(udfs=[SumUDF()]))
    store.dispatch(CamConnectedEvent())
    store.dispatch(CamDisconnectedEvent())
    assert store.state.processing == ProcessingState.IDLE


def test_nav_shape(store):
    store.dispatch(SetNavShapeEvent(nav_shape=(128, 128)))
    assert store.state.nav_shape == (128, 128)
