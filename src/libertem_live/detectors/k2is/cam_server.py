import sys
import queue
import logging

import zmq
import click

from libertem_live.detectors.common import (
    recv_serialized, SerializedQueue,
)
from .proto import MySubProcess, SyncState
from .state import (
    Event, Store, cam_server_reducer, cam_server_effects,
    EventReplicaServer, CamServerState, LifecycleState, CamConnectionState,
    ProcessingState,

    StartingEvent, StartupCompleteEvent, StoppedEvent,
)


logger = logging.getLogger(__name__)


def main_loop() -> None:
    event_queue: queue.Queue[Event] = queue.Queue()
    initial_state = CamServerState(
        lifecycle=LifecycleState.STARTING,
        cam_connection=CamConnectionState.DISCONNECTED,
        processing=ProcessingState.IDLE,
        udfs=[],
        nav_shape=(),
    )
    store: Store = Store(
        reducer=cam_server_reducer,
        initial_state=initial_state
    )
    event_pump = EventReplicaServer.make_event_pump(event_queue)
    store.listen_all(event_pump)
    store.listen_all(cam_server_effects)

    event_publisher = EventReplicaServer(
        event_queue,
        initial_state=initial_state
    )
    event_publisher.start()

    zctx = zmq.Context()
    control = zctx.socket(zmq.REP)
    control.bind("tcp://*:7204")
    poller = zmq.Poller()
    poller.register(control, zmq.POLLIN)

    processes = []
    oqs = []

    try:
        store.dispatch(StartingEvent())

        # FIXME: this is detector-specific, needs to be abstracted
        # FIXME: magic constant?
        ss = SyncState(num_processes=8)
        for idx in range(8):
            oq = SerializedQueue()
            p = MySubProcess(
                idx=idx,
                sync_state=ss,
                out_queue=oq
            )
            p.start()
            processes.append(p)
            oqs.append(oq)

        store.dispatch(StartupCompleteEvent())

        while store.state.lifecycle is not LifecycleState.STOPPING:
            event_publisher.maybe_raise()
            poll_events = dict(poller.poll(1))
            if control in poll_events:
                event = recv_serialized(control)
                store.dispatch(event)
                control.send_pyobj(None)
    finally:
        logging.debug("main_loop: finally")
        try:
            store.dispatch(StoppedEvent())
        except Exception:
            pass
        store.remove_callback(event_pump)
        event_publisher.stop()
        event_publisher.join()
        for p in processes:
            p.stop()
        for p in processes:
            p.join(timeout=1)
            # we can't determine join timeout status by return value,
            # so we explicitly check here with `is_alive` and cleanup
            # a bit more forecully :
        for p in processes:
            if p.is_alive():
                print("still alive, using the force")
                p.terminate()
                p.join()


@click.command()
# @click.argument('path', type=click.Path(exists=True))
# @click.option('--continuous', default=False, is_flag=True)
# @click.option('--cached', default='NONE', type=click.Choice(
#     ['NONE', 'MEM', 'MEMFD'], case_sensitive=False)
# )
# FIXME: what is the IANA guideline on unregistered port numbers?
# @click.option('--port', type=int, default=7201)
# @click.option('--max-runs', type=int, default=-1)
def main():
    logging.basicConfig(level=logging.INFO)
    main_loop()
    sys.exit(0)


if __name__ == "__main__":
    main()
