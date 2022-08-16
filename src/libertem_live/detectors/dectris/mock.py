from contextlib import contextmanager

import numpy as np

from libertem.common.executor import TaskProtocol, WorkerQueue
from libertem_live.detectors.dectris.acquisition import (
    AcquisitionParams, DectrisAcquisition, DectrisCommHandler, DetectorConfig,
)


class OfflineCommHandler(DectrisCommHandler):
    def __init__(self, data, params: AcquisitionParams):
        self.params = params
        self.data = data

    def start(self):
        pass

    def done(self):
        pass

    def handle_task(self, task: TaskProtocol, queue: WorkerQueue):
        from libertem_dectris import headers, Frame, FrameStack, PixelType
        partition = task.get_partition()
        slice_ = partition.slice
        start_idx = slice_.origin[0]
        end_idx = slice_.origin[0] + slice_.shape[0]
        current_idx = start_idx

        assert len(self.data.shape) == 3
        iterator = iter(self.data[start_idx:end_idx])

        while current_idx < end_idx:
            frame_stack_data = next(iterator)
            frame_stack_data = frame_stack_data[np.newaxis, ...]
            first_frame = frame_stack_data[0]
            frame_stack = FrameStack.from_frame_list([
                Frame(
                    data=frame_stack_data[0].tobytes(),
                    dimage=headers.DImage(
                        frame=current_idx,
                        series=self.params.sequence_id,
                        hash="",  # currently unused
                    ),
                    dimaged=headers.DImageD(
                        shape=first_frame.shape,
                        type_=PixelType.Uint8,  # FIXME!
                        encoding="<",
                    ),
                    dconfig=headers.DConfig(0, 0, 0),  # currently unused
                )
            ])
            dtype = np.dtype(self.data.dtype)
            shape = tuple(first_frame.shape)
            serialized = frame_stack.serialize()
            queue.put({
                "type": "FRAMES",
                "dtype": dtype,
                "shape": shape,
                "encoding": '<',
            }, payload=serialized)

            current_idx += len(frame_stack_data)


class OfflineAcquisition(DectrisAcquisition):
    def __init__(self, mock_data, *args, **kwargs):
        self.data = mock_data
        super().__init__(*args, **kwargs)

    def connect(self):
        pass  # NOOP

    def get_task_comm_handler(self):
        return OfflineCommHandler(data=self.data, params=self._acq_state)

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
