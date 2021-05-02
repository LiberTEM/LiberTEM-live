import enum
import time
from typing import (
    Union, Optional, Tuple, List, Callable, DefaultDict, NoReturn,
)
from typing_extensions import Literal
from collections import defaultdict
import logging
import threading
import queue

from libertem.udf.base import UDF
from libertem_live.detectors.common import ErrThreadMixin, send_serialized
from ..common import recv_serialized

import zmq
import pydantic

logger = logging.getLogger(__name__)


def _assert_never(x: NoReturn) -> NoReturn:
    """
    Check for exhaustive matching. mypy will complain
    about a type mismatch in the argument of this function
    if the previous if/elif didn't match all of the possible
    enum values. Note: the matching needs to use `is`.
    """
    assert False, "Unhandled type: {}".format(type(x).__name__)


class EventType(str, enum.Enum):
    # XXX
    # XXX External events (that are sent to us via zmq clients):
    # XXX

    # overwrite the list of enabled UDFs with a new one.
    # The list can be empty to stop processing.
    SET_UDFS = 'SET_UDFS'

    # change the current nav result shape
    SET_NAV_SHAPE = 'SET_NAV_SHAPE'

    # An exception was thrown when running the UDF
    UDF_ERROR = 'UDF_ERROR'

    # XXX
    # XXX Internal events (that come from our "own" processes):
    # XXX

    # emitted when starting, or re-starting threads and processes:
    STARTING = 'STARTING'

    # emitted when our resources have started and are available:
    STARTUP_COMPLETE = 'STARTUP_COMPLETE'

    # emitted when we disconnect from the camera, voluntarily or not:
    CAM_DISCONNECTED = 'CAM_DISCONNECTED'

    # emitted when the connection to the camera has been estabilshed,
    # and we either are receiving a stream of data, or the camera is ready
    # to be triggered:
    # TODO: maybe split "connected" and "primed" events?
    CAM_CONNECTED = 'CAM_CONNECTED'

    # Some camera-related error happened:
    CAM_ERROR = 'CAM_ERROR'

    # UDFs are being run on the data stream:
    PROCESSING_STARTED = 'PROCESSING_STARTED'

    # stop running UDFs
    STOP_PROCESSING = 'STOP_PROCESSING'

    # UDFs are no longer being run on the data stream:
    PROCESSING_STOPPED = 'PROCESSING_STOPPED'

    # the server should shut down:
    STOP = 'STOP'

    # the server is shutting down:
    STOPPED = 'STOPPED'


class SetUDFsEvent(pydantic.BaseModel):
    typ: Literal[EventType.SET_UDFS] = EventType.SET_UDFS
    udfs: List[UDF]

    class Config:
        # needed to support Python types
        # as event payload, for example List[UDF]:
        arbitrary_types_allowed = True


class SetNavShapeEvent(pydantic.BaseModel):
    typ: Literal[EventType.SET_NAV_SHAPE] = EventType.SET_NAV_SHAPE
    nav_shape: Tuple[int, ...]


class StopProcessingEvent(pydantic.BaseModel):
    typ: Literal[EventType.STOP_PROCESSING] = EventType.STOP_PROCESSING


class StartingEvent(pydantic.BaseModel):
    typ: Literal[EventType.STARTING] = EventType.STARTING


class StoppedEvent(pydantic.BaseModel):
    typ: Literal[EventType.STOPPED] = EventType.STOPPED


class StopEvent(pydantic.BaseModel):
    typ: Literal[EventType.STOP] = EventType.STOP


class StartupCompleteEvent(pydantic.BaseModel):
    typ: Literal[EventType.STARTUP_COMPLETE] = EventType.STARTUP_COMPLETE


class CamConnectedEvent(pydantic.BaseModel):
    typ: Literal[EventType.CAM_CONNECTED] = EventType.CAM_CONNECTED


class CamDisconnectedEvent(pydantic.BaseModel):
    typ: Literal[EventType.CAM_DISCONNECTED] = EventType.CAM_DISCONNECTED


class CamErrorEvent(pydantic.BaseModel):
    typ: Literal[EventType.CAM_ERROR] = EventType.CAM_ERROR


Event = Union[
    SetUDFsEvent,
    SetNavShapeEvent,
    StopProcessingEvent,
    StartupCompleteEvent,
    StartingEvent,
    StopEvent,
    StoppedEvent,
]


class EffectSink:
    def emit(self, event: Event):
        raise NotImplementedError(
            "subclasses need to override this function"
        )


class BaseState(pydantic.BaseModel):
    serial: int

    def update(self, **kwargs):
        data = self.dict()
        data.update(kwargs)
        return self.__class__(**data)


StoreListener = Callable[[
    BaseState,
    BaseState,
    Event,
    EffectSink,
], None]
StoreListenersType = DefaultDict[Optional[EventType], List[StoreListener]]
ReducerType = Callable[[BaseState, Event], BaseState]


class Store(EffectSink):
    """
    The Store keeps track of the current State, and allows callbacks to be
    registered for all incoming events.
    """
    def __init__(self, reducer, initial_state):
        self.state = initial_state
        self.reducer: ReducerType = reducer
        self.listeners: StoreListenersType = defaultdict(lambda: [])

    def dispatch(self, event: Event):
        new_state = self.reducer(self.state, event)
        new_state = new_state.update(serial=new_state.serial + 1)
        self._call_listeners(new_state, event)
        self.state = new_state

    def _call_listeners(self, new_state, event):
        listeners = self.listeners[event.typ] + self.listeners[None]
        logger.debug(f"dispatch: {event.typ} to {len(listeners)} listeners")
        for listener in listeners:
            listener(self.state, new_state, event, self)

    def listen(self, typ: EventType, callback: StoreListener):
        self.listeners[typ].append(callback)

    def listen_all(self, callback: StoreListener):
        """
        Add a callback that will be called for all events

        Parameters
        ----------
        callback : StoreListener
            The callback to be added
        """
        self.listeners[None].append(callback)

    def remove_callback(self, callback: StoreListener):
        for typ in self.listeners:
            cbs = self.listeners[typ]
            if callback in cbs:
                cbs.remove(callback)

    def emit(self, event: Event):
        self.dispatch(event)


class StateUpdate(pydantic.BaseModel):
    event: Event
    state: BaseState


class ReplicatedStore(Store):
    def __init__(self):
        self.state: Optional[BaseState] = None
        self.reducer = None  # we don't use a reducer in the replicated store:
        self.listeners: StoreListenersType = defaultdict(lambda: [])
        self.c = zmq.Context()
        self.control = self.c.socket(zmq.REQ)
        self.control.connect("tcp://127.0.0.1:7204")

    def dispatch_update(self, update: StateUpdate):
        self._call_listeners(update.state, update.event)
        self.state = update.state

    def dispatch(self, event: Event):
        send_serialized(self.control, event)
        recv_serialized(self.control)


class EventReplicaServer(ErrThreadMixin, threading.Thread):
    """
    Pump out events and state snapshots from a threading :class:`queue.Queue`
    into a zeromq PUB socket.

    Note that this is not optimized for large state at all; we just broadcast
    the whole state after each event.
    """
    def __init__(self, event_queue: queue.Queue, initial_state: BaseState, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_queue = event_queue
        self.state = initial_state

    def run(self):
        try:
            c = zmq.Context()
            pub = c.socket(zmq.PUB)
            pub.bind("tcp://*:7202")

            # allow clients some time to connect before we pump out events:
            time.sleep(0.2)

            while not self.is_stopped():
                try:
                    event, new_state = self.event_queue.get(timeout=0.1)
                    self.state = new_state
                    logger.debug(f"sending event {event.typ}")
                    upd = StateUpdate(event=event, state=new_state)
                    send_serialized(pub, upd)
                except queue.Empty:
                    continue
        except Exception as e:
            return self.error(e)
        finally:
            pub.close()
            logger.info("EventPublisher shutting down")

    @classmethod
    def make_event_pump(cls, event_queue) -> StoreListener:
        def pump_into_queue(state, new_state, event, store):
            event_queue.put((event, new_state))
        return pump_into_queue


class EventReplicaClientThread(ErrThreadMixin, threading.Thread):
    def __init__(self, *args, **kwargs):
        self.have_state = threading.Event()
        self.sub = None
        super().__init__(*args, **kwargs)

    def run(self):
        try:
            self.sub = EventReplicaClient()
            try:
                while not self.is_stopped():
                    self.sub.do_events(timeout=1)
            finally:
                self.sub.close()
        except Exception as e:
            return self.error(e)

    def dispatch(self, event: Event):
        assert self.sub is not None
        return self.sub.dispatch(event)


class EventReplicaClient:
    def __init__(self):
        self.store = ReplicatedStore()
        self.c = zmq.Context()
        self.sub = self.c.socket(zmq.SUB)
        self.sub.connect("tcp://127.0.0.1:7202")
        self.sub.subscribe(b"")
        self.poller = zmq.Poller()
        self.poller.register(self.sub, zmq.POLLIN)

    def close(self):
        self.sub.close()

    def do_events(self, timeout=0):
        """
        Receive replication events

        Parameters
        ----------
        timeout : float, optional
            Timeout in milliseconds
        """
        events = dict(self.poller.poll(timeout))
        if self.sub in events:
            upd = recv_serialized(self.sub)
            self.store.dispatch_update(upd)

    def dispatch(self, event: Event):
        return self.store.dispatch(event)


class LifecycleState(enum.IntEnum):
    STARTING = 1
    STARTUP_COMPLETE = 2
    STOPPING = 3
    STOPPED = 4


class CamConnectionState(enum.IntEnum):
    DISCONNECTED = 1
    CONNECTED = 2


class ProcessingState(enum.IntEnum):
    IDLE = 1
    RUNNING = 2


class CamServerState(BaseState):
    serial: int = 0
    lifecycle: LifecycleState
    cam_connection: CamConnectionState
    processing: ProcessingState
    udfs: List[UDF]
    nav_shape: Tuple[int, ...]

    class Config:
        # needed to support Python types
        # as event payload, for example List[UDF]:
        arbitrary_types_allowed = True


def cam_server_effects(
    state: CamServerState, new_state: CamServerState, event: Event, effects: EffectSink
):
    if event.typ is EventType.CAM_DISCONNECTED:
        if state.processing is ProcessingState.RUNNING:
            effects.emit(StopProcessingEvent())
    elif event.typ is EventType.CAM_ERROR:
        if state.processing is ProcessingState.RUNNING:
            effects.emit(StopProcessingEvent())


def cam_server_reducer(state: CamServerState, event: Event) -> CamServerState:

    if event.typ is EventType.STARTUP_COMPLETE:
        assert state.lifecycle is LifecycleState.STARTING
        return state.update(
            lifecycle=LifecycleState.STARTUP_COMPLETE,
        )
    elif event.typ is EventType.STARTING:
        return state  # FIXME

    elif event.typ is EventType.PROCESSING_STOPPED:
        return state
    elif event.typ is EventType.PROCESSING_STARTED:
        return state
    elif event.typ is EventType.STOP_PROCESSING:
        # someone else should listen to this event and stop running the UDFs
        return state

    elif event.typ is EventType.CAM_CONNECTED:
        assert state.cam_connection is CamConnectionState.DISCONNECTED
        return state.update(
            cam_connection=CamConnectionState.CONNECTED
        )
    elif event.typ is EventType.CAM_DISCONNECTED:
        return state  # FIXME
    elif event.typ is EventType.CAM_ERROR:
        return state  # FIXME

    elif event.typ is EventType.SET_UDFS:
        if state.processing is ProcessingState.IDLE:
            # transition: IDLE -> RUNNING
            # TODO: do we need an intermediate state,
            # like "starting to run"?
            return state.update(
                processing=ProcessingState.RUNNING,
                udfs=event.udfs,
            )
        else:
            # RUNNING -> RUNNING
            return state.update(
                udfs=event.udfs,
            )
    elif event.typ is EventType.UDF_ERROR:
        return state  # FIXME: should explicitly stop running UDFs

    elif event.typ is EventType.SET_NAV_SHAPE:
        return state.update(
            nav_shape=event.nav_shape
        )

    elif event.typ is EventType.STOP:
        return state.update(lifecycle=LifecycleState.STOPPING)
    elif event.typ is EventType.STOPPED:
        return state.update(lifecycle=LifecycleState.STOPPED)

    else:
        _assert_never(event.typ)
