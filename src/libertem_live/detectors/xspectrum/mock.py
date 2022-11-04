import sys
from unittest import mock
from contextlib import contextmanager


class MockDetectorReceiver:
    '''
    An instance of this class mocks the detector and
    the receiver of the X-Spectrum API
    simultaneously since that makes
    management of settings, counters etc easier.
    '''
    def __init__(self, data):
        assert len(data.shape) == 3
        self._data = data
        self._counter = 0
        self._seq = 0
        self.number_of_frames = 1
        self.frame_depth = data.itemsize * 8
        self._ram_allocated = False
        self._voltage_settled = False

        self._inflight = []

    def get_frame(self, timeout=None):
        if self._counter >= self.number_of_frames:
            return None
        index = self._counter % len(self._data)
        result_data = self._data[index].flatten()

        result = mock.MagicMock()
        result.seq = self._seq + 1
        result.nr = self._counter + 1
        result.data = memoryview(result_data)

        self._inflight.append(result)

        self._seq += 1
        self._counter += 1
        return result

    @property
    def frame_height(self):
        return self._data.shape[1]

    @property
    def frame_width(self):
        return self._data.shape[2]

    @property
    def compression(self):
        return False

    @property
    def ram_allocated(self):
        # First access is False, then True
        res = self._ram_allocated
        self._ram_allocated = True
        return res

    def voltage_settled(self, module_id):
        # First access is False, then True
        res = self._voltage_settled
        self._voltage_settled = True
        return res

    def start_acquisition(self):
        # only true if ready status was checked
        assert self._voltage_settled
        assert self._ram_allocated
        self.counter = 0

    def release_frame(self, frame):
        self._inflight.remove(frame)


@contextmanager
def mock_xspectrum(data):
    with mock.patch.dict(sys.modules, {'pyxsp': mock.MagicMock()}):
        import pyxsp as px

        mock_detectorreceiver = MockDetectorReceiver(data)
        mock_system = mock.MagicMock()
        mock_system.list_detectors = mock.MagicMock(return_value=('lambda',))
        mock_system.list_receivers = mock.MagicMock(return_value=('lambda/1',))

        mock_system.open_detector = mock.MagicMock(return_value=mock_detectorreceiver)
        mock_system.open_receiver = mock.MagicMock(return_value=mock_detectorreceiver)

        px.System = mock.MagicMock(return_value=mock_system)
        yield
        # Confirm all frames have been released
        assert len(mock_detectorreceiver._inflight) == 0
