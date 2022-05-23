import os
import gc
import logging
from contextlib import contextmanager
from threading import Thread
from queue import Empty
import time
from typing import List, Union
from typing_extensions import Literal

import numpy as np

from libertem.io.dataset.base.tiling_scheme import TilingScheme
from libertem.udf.sum import SumUDF
from libertem.udf.stddev import StdDevUDF
from libertem.udf.base import UDFPartRunner, UDFParams, UDF
from libertem.common import Shape
from libertem.common.executor import Environment

from libertem_live.channel import WorkerPool, WorkerQueues
from libertem_live.detectors.dectris.acquisition import (
    AcquisitionParams, DectrisAcquisition, DetectorConfig, Receiver,
    RawFrame
)

try:
    import prctl
except ImportError:
    prctl = None

logger = logging.getLogger(__name__)

MSG_TYPES = Union[
    Literal['BEGIN_PARTITION'],
    Literal['FRAME'],
    Literal['END_PARTITION'],
    Literal['SHUTDOWN'],
]


def set_thread_name(name: str):
    """
    Set a thread name; mostly useful for using system tools for profiling

    Parameters
    ----------
    name : str
        The thread name
    """
    if prctl is None:
        return
    prctl.set_name(name)


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
                nimages=self.data.shape[0],
                trigger_mode=self._trigger_mode
            )
            self.trigger()  # <-- this triggers, either via API or via HW trigger
            yield
        finally:
            self._acq_state = None


def run_udf_on_frames(partition, frames, udfs, params):
    partition._receiver = frames  # FIXME  # proper interface for this one
    runner = UDFPartRunner(udfs)
    if True:
        result = runner.run_for_partition(
            partition=partition,
            params=params,
            env=Environment(threaded_executor=False, threads_per_worker=1),
        )
        for frame in frames:
            raise RuntimeError("frames should be fully consumed here!")
        return result
    if False:
        for frame in frames:
            frame.sum()


def get_frames(partition, request):
    """
    Consume all FRAME messages from the request queue until we get an
    END_PARTITION message (which we also consume)
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
                # print(f"partition {partition} done")
                return
            else:
                raise RuntimeError(f"invalid header type {header}; FRAME or END_PARTITION expected")


def worker(queues: WorkerQueues, idx: int):
    udfs: List[UDF] = []
    params = None

    set_thread_name(f"worker-{idx}")

    collect_time = 0.0

    while True:
        with queues.request.get() as msg:
            header, payload = msg
            header_type = header["type"]
            if header_type == "START_PARTITION":
                partition = header["partition"]
                logger.debug(f"processing partition {partition}")
                frames = get_frames(partition, queues.request)
                result = run_udf_on_frames(partition, frames, udfs, params)
                queues.response.put({"type": "RESULT", "result": result})
                t0 = time.time()
                gc.collect()
                t1 = time.time()
                collect_time += t1 - t0
                continue
            elif header_type == "SET_UDFS_AND_PARAMS":
                udfs = header["udfs"]
                params = header["params"]
                continue
            elif header_type == "SHUTDOWN":
                logger.debug(f"gc in worker {idx}, time sum {collect_time}s")
                queues.request.close()
                queues.response.close()
                break
            elif header_type == "WARMUP":
                env = Environment(threaded_executor=False, threads_per_worker=2)
                with env.enter():
                    pass
            else:
                raise RuntimeError(f"unknown message {header}")


def feed_workers(pool: WorkerPool, aq: DectrisAcquisition):
    set_thread_name("feed_workers")

    print("distributing work")
    t0 = time.time()
    with aq.acquire():
        print("creating and starting receiver")
        receiver = aq.get_receiver()
        receiver.start()
        idx = 0
        for partition in aq.get_partitions():
            qs = pool.get_worker_queues(idx)
            partition._receiver = None
            qs.request.put({"type": "START_PARTITION", "partition": partition})
            for frame_idx in range(partition.shape.nav.size):
                raw_frame = next(receiver)
                qs.request.put({
                    "type": "FRAME",
                    "shape": raw_frame.shape,
                    "dtype": raw_frame.dtype,
                    "encoding": raw_frame.encoding,
                }, payload=np.frombuffer(raw_frame.data, dtype=np.uint8))
            qs.request.put({"type": "END_PARTITION"})
            idx = (idx + 1) % pool.size
        t1 = time.time()
        print(f"finished feeding workers in {t1 - t0}s")


class FFTUDF(UDF):
    def get_result_buffers(self):
        return {
            'sum_of_fft': self.buffer(kind='nav', dtype=np.complex128),
        }

    def process_frame(self, frame):
        self.results.sum_of_fft[:] = np.sum(np.fft.fft2(frame))


def run_on_acquisition(aq: DectrisAcquisition):
    pool = WorkerPool(processes=22, worker_fn=worker)
    from perf_utils import perf  # NOQA

    ts = TilingScheme.make_for_shape(
        tileshape=Shape((12, 512, 512), sig_dims=2),
        dataset_shape=aq.shape,
    )

    # XXX move to executor:
    for qs in pool.all_worker_queues():
        qs.request.put({
            "type": "SET_UDFS_AND_PARAMS",
            # "udfs": [SumUDF()],
            "udfs": [SumUDF(), StdDevUDF(), FFTUDF()],
            "params": UDFParams(corrections=None, roi=None, tiling_scheme=ts, kwargs=[{}])
        })
        qs.request.put({
            "type": "WARMUP",
        })

    # with perf("multiprocess-dectris"):
    if True:
        for i in range(2):
            # XXX msg_thread is specific to the dectris
            # -> move to MainController
            msg_thread = Thread(target=feed_workers, args=(pool, aq))
            msg_thread.name = "feed_workers"
            msg_thread.daemon = True
            msg_thread.start()

            num_partitions = int(aq.shape.nav.size // aq._frames_per_partition)
            t0 = time.time()
            print("gathering responses...")

            num_responses = 0
            get_timeout = 0.1
            while num_responses < num_partitions:
                try:
                    with pool.response_queue.get(block=True, timeout=get_timeout) as response:
                        resp_header, payload = response
                        assert payload is None
                        assert resp_header['type'] == "RESULT", f"resp_header == {resp_header}"
                    num_responses += 1
                except Empty:
                    continue
            t1 = time.time()
            print(f"time for this round: {t1-t0} (-> {aq.shape.nav.size/(t1-t0)}fps)")
            print(
                f"Max SHM usage: {qs.request._psa._used/1024/1024}MiB "
                f"of {qs.request._psa._size/1024/1024}MiB"
            )

            # after a while, the msg_thread has sent all partitions and exits:
            msg_thread.join()

    pool.close_resp_queue()

    # ... and we can shut down the workers:
    for qs, p in pool.all_workers():
        qs.request.put({"type": "SHUTDOWN"})
        qs.request.close()
        qs.response.close()
        p.join()


if __name__ == "__main__":
    set_thread_name("main")
    print(f"main pid {os.getpid()}")
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
        frames_per_partition=128,
    )
    aq.initialize(None)
    print(aq.shape, aq.dtype)

    if False:
        t0 = time.time()
        with aq.acquire():
            r = aq.get_receiver()
            r.start()
            for frame in r:
                pass
        t1 = time.time()
        print(t1-t0)

    if True:
        run_on_acquisition(aq)


# Code sketches below:
class MainController:
    """
    A controller that takes care of detector-specific concerns and is
    instantiated once per acquisition.

    This is used by the `UDFRunner`, and can be swapped out per data source.

    This object can take care of, for example, receiving data from the detector.
    """
    pass


class PerWorkerController:
    """
    A persistent controller object that takes care of detector-specific concerns
    on the worker processes.

    This is wrapped around the `UDFPartRunner`, and is useful for hanging on to
    persistent resources, like shared memory, sockets or similar.
    """
    pass
