from contextlib import contextmanager
import time
import multiprocessing as mp
from typing import Union
from typing_extensions import Literal

import numpy as np

from libertem.udf.base import UDFPartRunner, UDFRunner, UDFParams
from libertem.common import Shape

from libertem_live.channel import WorkerPool, WorkerQueues
from libertem_live.detectors.dectris.acquisition import AcquisitionParams, DectrisAcquisition, DetectorConfig, Receiver, TriggerMode

MSG_TYPES = Union[
    Literal['BEGIN_PARTITION'],
    Literal['FRAME'],
    Literal['END_PARTITION'],
    Literal['SHUTDOWN'],
]


class OfflineReceiver(Receiver):
    """
    Mock Receiver that reads from a numpy array
    """
    def __init__(self, data):
        self.data = data
        assert len(data.shape) == 3
        self._idx = 0

    def __next__(self) -> np.ndarray:
        if self._idx == self.data.shape[0]:
            raise StopIteration
        data = self.data[self._idx]
        self._idx += 1
        return data


class OfflineAcquisition(DectrisAcquisition):
    def __init__(self, mock_data, *args, **kwargs):
        self.data = mock_data
        super().__init__(*args, **kwargs)

    def connect(self):
        pass  # NOOP

    def get_receiver(self):
        return OfflineReceiver(data=self.data)

    def get_api_client(self):
        return None  # should not make API calls in testing!

    def get_detector_config(self) -> DetectorConfig:
        shape_y = 512
        shape_x = 512
        bit_depth = 8
        return DetectorConfig(
            x_pixels_in_detector=shape_x, y_pixels_in_detector=shape_y, bit_depth=bit_depth
        )

    @contextmanager
    def acquire(self):
        try:
            self._acq_state = AcquisitionParams(
                sequence_id=42,
                nimages=128,
                trigger_mode=self._trigger_mode
            )
            self.trigger()  # <-- this triggers, either via API or via HW trigger
            yield
        finally:
            self._acq_state = None


class DectrisUDFPartRunner(UDFPartRunner):
    pass


def decode(payload):
    return payload  # TODO: move decoding step here


def run_udf_on_frames(partition, frames, udfs, params):
    for frame in frames:
        pass  # TODO


def get_frames(partition, request):
    """
    Consume all FRAME messages until we get an END_PARTITION message
    """
    while True:
        with request.get() as msg:
            header, payload = msg
            header_type = header["type"]
            if header_type == "FRAME":
                frame_arr = decode(payload)
                yield frame_arr
            elif header_type == "END_PARTITION":
                print(f"partition {partition} done")
                return
            else:
                raise RuntimeError(f"invalid header type {header}; FRAME or END_PARTITION expected")


def worker(queues: WorkerQueues):
    udfs = []
    params = None

    while True:
        with queues.request.get() as msg:
            header, payload = msg
            header_type = header["type"]
            if header_type == "START_PARTITION":
                partition = header["partition"]
                print(f"processing partition {partition}")
                frames = get_frames(partition, queues.request)
                result = run_udf_on_frames(partition, frames, udfs, params)
                queues.response.put({"type": "RESULT", "result": result})
                continue
            elif header_type == "SET_UDFS_AND_PARAMS":
                udfs = header["udfs"]
                params = header["params"]
                continue
            elif header_type == "SHUTDOWN":
                break
            else:
                raise RuntimeError(f"unknown message {header}")


def run_on_acquisition(aq: DectrisAcquisition):
    pool = WorkerPool(processes=2, worker_fn=worker)
    from perf_utils import perf

    for qs in pool.all_worker_queues():
        qs.request.put({
            "type": "SET_UDFS_AND_PARAMS",
            "udfs": [object()],
            "params": UDFParams(corrections=None, roi=None, tiling_scheme=None, kwargs=[{}])
        })

    REPEATS = 1

    receiver = aq.get_receiver()
    receiver.start()

    # with perf("looped"):
    if True:
        for i in range(REPEATS):
            receiver = aq.get_receiver()
            receiver.start()
            t0 = time.time()
            idx = 0
            for partition in aq.get_partitions():
                qs = pool.get_worker_queues(idx)
                partition._receiver = None
                qs.request.put({"type": "START_PARTITION", "partition": partition})
                for frame_idx in range(partition.shape.nav.size):
                    frame = next(receiver)
                    qs.request.put({"type": "FRAME"}, frame)
                qs.request.put({"type": "END_PARTITION"})
                idx = (idx + 1) % pool.size
            # synchronization:
            print("waiting for response...")
            with qs.response.get() as response:
                print(response)
            t1 = time.time()
            print(t1-t0)

    for qs in pool.all_worker_queues():
        qs.request.put({"type": "SHUTDOWN"})

    pool.join_all()


if __name__ == "__main__":
    if False:
        dataset_shape = Shape((512, 512, 512), sig_dims=2)
        data = np.random.randn(*dataset_shape).astype(np.uint8)
        print(f"data size: {data.nbytes/1024/1024}MiB")
        aq = OfflineAcquisition(
            nav_shape=tuple(dataset_shape.nav),
            mock_data=data,
            frames_per_partition=42,  # chosen not to evenly divide `dataset_shape.nav`
            api_host=None,
            api_port=None,
            data_host=None,
            data_port=None,
            trigger_mode="exte",
        )
    aq = DectrisAcquisition(
        api_host="localhost",
        api_port="8910",
        data_host="localhost",
        data_port=9999,
        nav_shape=(128, 128),
        trigger_mode="exte",
    )
    aq.initialize(None)
    with aq.acquire():
        run_on_acquisition(aq)
