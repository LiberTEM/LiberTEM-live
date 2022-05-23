import typing
from typing import List
from contextlib import contextmanager

import numpy as np

from libertem.common.controller import WorkerController, MainController
from libertem.common.executor import TaskProtocol

from libertem_live.channel import ShmQueue, WorkerQueues
from libertem_live.detectors.dectris.acquisition import RawFrame

if typing.TYPE_CHECKING:
    from libertem_live.detectors.dectris.acquisition import Receiver

# TODO:
# - [ ] enhance `TaskProtocol` to get access to the task's partition
# - [ ] function: get request queue from task


def _get_frames(request):
    """
    Consume all FRAME messages from the request queue until we get an
    END_PARTITION message (which we also consume).

    This protocol needs to be kept in sync with the
    `DectrisMainController` below.
    """
    while True:
        with request.get() as msg:
            header, payload = msg
            header_type = header["type"]
            if header_type == "FRAME":
                raw_frame = RawFrame(
                    data=payload,
                    encoding=header['encoding'],
                    dtype=header['dtype'],
                    shape=header['shape'],
                )
                frame_arr = raw_frame.decode()
                yield frame_arr
            elif header_type == "END_PARTITION":
                return
            else:
                raise RuntimeError(f"invalid header type {header}; FRAME or END_PARTITION expected")


class DectrisWorkerController(WorkerController):
    def __init__(self, queues: WorkerQueues):
        self.queues = queues

    @contextmanager
    def handle_task(self, task: TaskProtocol):
        data_source = _get_frames(self.queues.request)
        task.get_partition().set_data_source(data_source)
        yield


class DectrisMainController(MainController):
    def __init__(self, receiver: Receiver):
        # somehow, the instance of the main controller needs
        # to come from the dectris acquisition instance
        # something like `aq.get_controller()` ?
        # it then needs to be made available in the
        # executor, connecting to `JobExecutor.run_tasks`
        # in a way...
        self._receiver = receiver

    # FIXME: generic queue interface instead of shmqueue here?
    def run_tasks(self, tasks: List[TaskProtocol]):
        """
        Stream the data for one partition to the given `request_queue`.
        Which partition is processed by which worker is decided by the
        `JobExecutor` - we just get to decide how the data for
        this `task` ends up on the worker. That is, we can decide the
        messages sent, including metadata and frame encoding etc.
        """

        # FIXME: how do we prevent intermingling executor-specifics and
        # detector specifics here?
        # We really want to run this stuff in a background thread,
        # if possible continuously. One reason for this is the
        # zeromq context, which can't be moved between threads.
        # So we need a way to ask the executor for the queue
        # to talk to the worker that is responsible for a given task
        # or partition.

        for task in tasks:
            request_queue: ShmQueue = get_request_queue_for_task(task)
            partition = task.get_partition()
            request_queue.put({"type": "START_PARTITION", "partition": partition})
            for frame_idx in range(partition.shape.nav.size):
                raw_frame: RawFrame = next(self._receiver)
                request_queue.put({
                    "type": "FRAME",
                    "shape": raw_frame.shape,
                    "dtype": raw_frame.dtype,
                    "encoding": raw_frame.encoding,
                }, payload=np.frombuffer(raw_frame.data, dtype=np.uint8))
            request_queue.put({"type": "END_PARTITION"})
