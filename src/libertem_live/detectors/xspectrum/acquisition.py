from contextlib import contextmanager
import logging
import time
from typing import NamedTuple, Optional, Sequence, Tuple

import numpy as np
from opentelemetry import trace

from libertem.common import Shape
from libertem.common.math import prod
from libertem.common.executor import TaskProtocol, WorkerQueue, TaskCommHandler
from libertem.io.dataset.base import DataSetMeta, BasePartition, DataSet

from libertem.corrections.corrset import CorrectionSet

from libertem_live.detectors.base.acquisition import AcquisitionMixin, FullframeLivePartition


tracer = trace.get_tracer(__name__)
logger = logging.getLogger(__name__)


class DetectorConnection(NamedTuple):
    system: object
    detectors: Sequence[object]
    receivers: Sequence[object]


def get_frames(request_queue):
    """
    Consume all FRAMES messages from the request queue until we get an
    END_PARTITION message (which we also consume)
    """
    while True:
        with request_queue.get() as msg:
            header, payload = msg
            header_type = header["type"]
            if header_type == "FRAMES":
                yield np.frombuffer(
                    payload, dtype=header['dtype']
                ).reshape(header['shape'])
            elif header_type == "END_PARTITION":
                # print(f"partition {partition} done")
                return
            else:
                raise RuntimeError(
                    f"invalid header type {header['type']}; FRAME or END_PARTITION expected"
                )


class XSpectrumCommHandler(TaskCommHandler):
    def __init__(self, params: DetectorConnection):
        self.params = params

    def handle_task(self, task: TaskProtocol, queue: WorkerQueue):
        with tracer.start_as_current_span("XSpectrumCommHandler.handle_task") as span:
            put_time = 0.0
            recv_time = 0.0
            # send the data for this task to the given worker
            partition = task.get_partition()
            slice_ = partition.slice
            start_idx = slice_.origin[0]
            end_idx = slice_.origin[0] + slice_.shape[0]
            span.set_attributes({
                "libertem.partition.start_idx": start_idx,
                "libertem.partition.end_idx": end_idx,
            })
            current_idx = start_idx
            sig_shape = tuple(partition.shape.sig)
            while current_idx < end_idx:
                t0 = time.perf_counter()
                # FIXME function of shutter time
                frame = self.params.receivers[0].get_frame(30000)
                if frame is None:
                    raise RuntimeError('Frame receiving timeout')
                if frame.nr - 1 != current_idx:
                    raise RuntimeError(
                        f"Frame index {frame.nr} not matching expected index {current_idx + 1}."
                    )
                t1 = time.perf_counter()
                recv_time += t1 - t0

                data = np.asarray(frame.data)
                assert data.size == prod(sig_shape)
                t0 = time.perf_counter()
                # This must be a copy, otherwise we'll
                # have to manage the release below
                header = {
                    "type": "FRAMES",
                    "status_code": frame.status_code,
                    "seq": frame.seq,
                    "shape": sig_shape,
                    "dtype": data.dtype,
                }
                with queue.put_nocopy(header, data.nbytes) as payload:
                    payload[:] = data.view(np.uint8)
                t1 = time.perf_counter()
                self.params.receivers[0].release_frame(frame)
                put_time += t1 - t0

                current_idx += 1
            span.set_attributes({
                "total_put_time": put_time,
                "total_recv_time": recv_time,
            })

    def start(self):
        pass

    def done(self):
        pass


class XSpectrumAcquisition(AcquisitionMixin, DataSet):
    def __init__(
        self,
        nav_shape: Tuple[int, ...],
        trigger=lambda aq: None,
        frames_per_partition: int = 128,
        enable_corrections: bool = False,
        name_pattern: Optional[str] = None,
    ):
        super().__init__(trigger=trigger)
        self._nav_shape = nav_shape
        self._sig_shape: Tuple[int, ...] = ()
        self._acq_state: Optional[DetectorConnection] = None
        self._frames_per_partition = min(frames_per_partition, prod(nav_shape))
        self._enable_corrections = enable_corrections
        self._name_pattern = name_pattern

    def initialize(self, executor) -> "DataSet":
        with self._connect() as connection:
            if len(connection.detectors) > 1:
                raise NotImplementedError(
                    'Only setups with a single detector are currently supported.'
                    f'Found {connection.detectors}.'
                )
            if len(connection.receivers) > 1:
                raise NotImplementedError(
                    'Only setups with a single receiver are currently supported. '
                    f'Found {connection.receivers}.'
                )
            r = connection.receivers[0]
            depth = r.frame_depth
            if depth <= 8:
                dtype = np.uint8
            elif depth <= 16:
                dtype = np.uint16
            elif depth <= 32:
                dtype = np.uint32
            elif depth <= 64:
                dtype = np.uint64
            else:
                raise RuntimeError('Invalid bit depth')

            self._sig_shape = (r.frame_height, r.frame_width)
            self._meta = DataSetMeta(
                shape=Shape(self._nav_shape + self._sig_shape, sig_dims=2),
                raw_dtype=dtype,
                dtype=dtype,
            )
            return self

    @property
    def dtype(self):
        return self._meta.dtype

    @property
    def raw_dtype(self):
        return self._meta.raw_dtype

    @property
    def shape(self):
        return self._meta.shape

    @property
    def meta(self):
        return self._meta

    def get_correction_data(self):
        if not self._enable_corrections:
            return CorrectionSet()
        else:
            raise NotImplementedError()

    @contextmanager
    def _connect(self) -> DetectorConnection:
        import pyxsp as px

        s = px.System('/opt/xsp/config/system.yml')
        if s is None:
            raise RuntimeError('X-Spectrum system creation failed, aborting.')
        try:
            s.connect()
            s.initialize()

            detectors = s.list_detectors()
            receivers = s.list_receivers()
            d = tuple(s.open_detector(id) for id in detectors)
            r = tuple(s.open_receiver(id) for id in receivers)

            yield DetectorConnection(
                system=s,
                detectors=d,
                receivers=r,
            )
        finally:
            pass
            # Quoting the pyxsp user manual 3.1: "The disconnect() method is
            # implicitly called in the destructor of the Systemobject, so that
            # it does not necessarily need to be called from the user
            # application."

            # s.disconnect()

    @contextmanager
    def acquire(self):
        with tracer.start_as_current_span('acquire') as span:
            with self._connect() as connection:
                s = connection.system
                d = connection.detectors[0]
                r = connection.receivers[0]

                if r.compression:
                    raise NotImplementedError()

                nimages = prod(self.shape.nav)
                d.number_of_frames = nimages

                while not r.ram_allocated:
                    time.sleep(0.1)
                while not d.voltage_settled(1):
                    time.sleep(0.1)
                d.start_acquisition()
                span.add_event("XSpectrumAcquisition.acquire:start_acquisition")

                try:
                    self._acq_state = connection
                    # this triggers, either via API or via HW trigger (in which case we
                    # don't need to do anything in the trigger function):
                    with tracer.start_as_current_span("XSpectrumAcquisition.trigger"):
                        self.trigger()
                    yield
                finally:
                    s.stop_acquisition()
                    self._acq_state = None

    def check_valid(self):
        pass

    def need_decode(self, read_dtype, roi, corrections):
        return True  # FIXME: we just do this to get a large tile size

    def adjust_tileshape(self, tileshape, roi):
        depth = 12
        return (depth, *self.meta.shape.sig)
        # return Shape((self._end_idx - self._start_idx, 256, 256), sig_dims=2)

    def get_max_io_size(self):
        # return 12*256*256*8
        # FIXME magic numbers?
        return 12*np.prod(self.meta.shape.sig)*8

    def get_base_shape(self, roi):
        return (1, 1, self.meta.shape.sig[-1])

    @property
    def acquisition_state(self):
        return self._acq_state

    def get_partitions(self):
        num_frames = np.prod(self._nav_shape, dtype=np.uint64)
        num_partitions = int(num_frames // self._frames_per_partition)

        slices = BasePartition.make_slices(self.shape, num_partitions)

        for part_slice, start, stop in slices:
            yield FullframeLivePartition(
                start_idx=start,
                end_idx=stop,
                get_frames=get_frames,
                meta=self._meta,
                partition_slice=part_slice,
            )

    def get_task_comm_handler(self) -> "XSpectrumCommHandler":
        assert self._acq_state is not None
        return XSpectrumCommHandler(
            params=self._acq_state,
        )
