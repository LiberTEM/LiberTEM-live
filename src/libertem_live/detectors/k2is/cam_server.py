import sys
import queue
import logging
import datetime as dt

import zmq
import click

from libertem_live.detectors.common import recv_serialized
from .proto import K2ListenerProcess, SyncState
from .state import (
    Event, Store, cam_server_reducer,
    EventReplicaServer, CamServerState, LifecycleState, CamConnectionState,
    ProcessingState,

    StartingEvent, StartupCompleteEvent, StoppedEvent,
)


logger = logging.getLogger(__name__)


def debug_events(state, new_state, event, store) -> None:
    logger.info("got event: %r", event)


def main_loop(force: bool, enable_tracing: bool, pdb: bool, profile: bool) -> None:
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
    store.listen_all(debug_events)

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

    try:
        store.dispatch(StartingEvent())

        # FIXME: this is detector-specific, needs to be abstracted
        # FIXME: magic constant?
        ss = SyncState(num_processes=8)
        for idx in range(8):
            p = K2ListenerProcess(
                idx=idx,
                sync_state=ss,
                enable_tracing=enable_tracing,
                pdb=pdb,
                profile=profile,
            )
            p.start()
            processes.append(p)

        store.dispatch(StartupCompleteEvent())

        while store.state.lifecycle is not LifecycleState.STOPPING:
            event_publisher.maybe_raise()
            poll_events = dict(poller.poll(1))
            if control in poll_events:
                event = recv_serialized(control)
                store.dispatch(event)
                control.send_pyobj(None)
            if any(p.is_stopped() for p in processes):
                break
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
            p.join(timeout=2)
            # we can't determine join timeout status by return value,
            # so we explicitly check here with `is_alive` and cleanup
            # a bit more forecully :
        for p in processes:
            if p.is_alive():
                if force:
                    print("still alive, using the force")
                    p.terminate()
                p.join()


# inspired by https://stackoverflow.com/a/6290946/540644
class MicrosecondFormatter(logging.Formatter):
    converter = dt.datetime.fromtimestamp

    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)

        if datefmt:
            s = ct.strftime(datefmt)
        else:
            t = ct.strftime("%Y-%m-%d %H:%M:%S")
            s = "%s,%03d" % (t, record.msecs)
        return s


@click.command()
# @click.argument('path', type=click.Path(exists=True))
# @click.option('--cached', default='NONE', type=click.Choice(
#     ['NONE', 'MEM', 'MEMFD'], case_sensitive=False)
# )
# FIXME: what is the IANA guideline on unregistered port numbers?
# @click.option('--port', type=int, default=7201)
# @click.option('--max-runs', type=int, default=-1)
@click.option('--force/--no-force', default=False, is_flag=True,
              help='forcefully stop processes')
@click.option('--enable-tracing/--disable-tracing', default=False, is_flag=True)
@click.option('--pdb/--no-pdb', default=False, is_flag=True)
@click.option('--profile/--no-profile', default=False, is_flag=True)
def main(force, enable_tracing, pdb, profile):
    # logging.basicConfig(
    #     level=logging.INFO,
    #     format='%(asctime)s %(levelname)-8s %(message)s',
    #     datefmt='%Y-%m-%d %H:%M:%S.%f',
    # )

    console = logging.StreamHandler()
    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(console)

    formatter = MicrosecondFormatter(
        fmt='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d,%H:%M:%S.%f'
    )
    console.setFormatter(formatter)

    main_loop(force=force, enable_tracing=enable_tracing, pdb=pdb, profile=profile)
    sys.exit(0)


if __name__ == "__main__":
    main()
