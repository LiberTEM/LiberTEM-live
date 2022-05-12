from contextlib import contextmanager
import base64
import json
import logging
from typing import NamedTuple, Optional, Protocol, Tuple, Union
from typing_extensions import Literal
import numpy as np
import bitshuffle
import lz4.block
from libertem.common import Shape, Slice
from libertem.common.math import prod
from libertem.io.dataset.base import (
    DataTile, DataSetMeta, BasePartition, Partition, DataSet, TilingScheme,
)
from libertem.corrections.corrset import CorrectionSet
import zmq

from libertem_live.detectors.base.acquisition import AcquisitionMixin


logger = logging.getLogger(__name__)

TriggerMode = Union[
    Literal['exte'],  # acquire one image for each trigger, `ntrigger` times
    Literal['exts'],  # acquire series of `nimages` with a single trigger
    Literal['ints'],  # internal software triggering
    Literal['inte'],  # internal software enable -> one image for each soft-trigger
]


class AcquisitionParams(NamedTuple):
    sequence_id: int
    nimages: int
    trigger_mode: TriggerMode


class DetectorConfig(NamedTuple):
    x_pixels_in_detector: int
    y_pixels_in_detector: int
    bit_depth: int


class Receiver(Protocol):
    def __iter__(self):
        return self

    def start(self):
        pass

    def __next__(self) -> np.ndarray:
        raise NotImplementedError(f"{self.__class__.__name__}.__next__ is not implemented")


def get_darray(darray) -> np.ndarray:
    """
    Helper to decode darray-encoded arrays to numpy
    """
    data = darray['value']['data']
    if 'base64' not in darray['value']['filters']:
        raise RuntimeError("don't unterstand this encoding, bailing")
    shape = tuple(reversed(darray['value']['shape']))
    dtype = darray['value']['type']
    data = base64.decodebytes(data.encode("ascii"))
    return np.frombuffer(data, dtype=dtype).reshape(shape)


class ZeroMQReceiver(Receiver):
    def __init__(self, socket: zmq.Socket, params: Optional[AcquisitionParams]):
        self._socket = socket
        self._params = params
        self._frame_id = 0
        self._running = False

    def start(self):
        if self._running:
            return
        if self._params is None:
            raise RuntimeError("can't receive frames without acquisition parameters set!")
        header_header, header = self.receive_acquisition_header()
        self._running = True

    def recv(self):
        res = 0
        while not res:
            res = self._socket.poll(100)
        return self._socket.recv()

    def receive_acquisition_header(self):
        while True:
            data = self.recv()
            try:
                header_header = json.loads(data)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if ('header_detail' in header_header
                    and header_header['series'] == self._params.sequence_id):
                break
        header = json.loads(self.recv())
        return header_header, header

    def receive_frame(self, frame_id):
        assert self._running, "need to be running to receive frames!"
        header_header = json.loads(self.recv())
        assert header_header['series'] == self._params.sequence_id
        assert header_header['frame'] == frame_id
        header = json.loads(self.recv())
        shape = tuple(reversed(header['shape']))
        dtype = np.dtype(header['type']).newbyteorder(header['encoding'][-1])
        size = prod(shape) * dtype.itemsize
        data = self.recv()
        if header['encoding'] in ('bs32-lz4<', 'bs16-lz4<', 'bs8-lz4<'):
            decompressed = bitshuffle.decompress_lz4(
                np.frombuffer(data[12:], dtype=np.uint8),
                shape=shape,
                dtype=dtype,
                block_size=0
            )
        elif header['encoding'] == 'lz4<':
            decompressed = lz4.block.decompress(data, uncompressed_size=size)
            decompressed = np.frombuffer(decompressed, dtype=dtype).reshape(shape)
        elif header['encoding'] == '<':
            decompressed = np.frombuffer(data, dtype=dtype).reshape(shape)
        else:
            raise RuntimeError(f'Unsupported encoding {header["encoding"]}')

        footer = json.loads(self.recv())
        return header_header, header, decompressed, footer

    def receive_acquisition_footer(self):
        footer = self.recv()
        return footer

    def __next__(self) -> np.ndarray:
        if self._frame_id >= self._params.nimages:
            self.receive_acquisition_footer()
            self._running = False
            raise StopIteration()
        f_header_header, f_header, decompressed, f_footer = self.receive_frame(self._frame_id)
        self._frame_id += 1
        return decompressed


class DectrisAcquisition(AcquisitionMixin, DataSet):
    def __init__(
        self,
        api_host: str,
        api_port: int,
        data_host: str,
        data_port: int,
        trigger_mode: TriggerMode,
        trigger=lambda aq: None,
        nav_shape: Optional[Tuple[int, ...]] = None,
        frames_per_partition: int = 128,
        enable_corrections: bool = False,
    ):
        super().__init__(trigger=trigger)
        self._api_host = api_host
        self._api_port = api_port
        self._data_host = data_host
        self._data_port = data_port
        self._nav_shape = nav_shape
        self._sig_shape: Tuple[int, ...] = ()
        self._acq_state: Optional[AcquisitionParams] = None
        self._zmq_ctx: Optional[zmq.Context] = None
        self._socket: Optional[zmq.Socket] = None
        self._frames_per_partition = frames_per_partition
        self._trigger_mode = trigger_mode
        self._enable_corrections = enable_corrections

    def get_api_client(self):
        from .DEigerClient import DEigerClient
        ec = DEigerClient(self._api_host, port=self._api_port)
        return ec

    def get_detector_config(self) -> DetectorConfig:
        ec = self.get_api_client()
        # FIXME: initialize detector here, if not already initialized?
        shape_x = ec.detectorConfig("x_pixels_in_detector")['value']
        shape_y = ec.detectorConfig("y_pixels_in_detector")['value']
        bit_depth = ec.detectorConfig("bit_depth_image")['value']
        return DetectorConfig(
            x_pixels_in_detector=shape_x, y_pixels_in_detector=shape_y, bit_depth=bit_depth
        )

    def initialize(self, executor) -> "DataSet":
        dc = self.get_detector_config()
        dtypes = {
            8: np.uint8,
            16: np.uint16,
            32: np.uint32,
        }
        try:
            dtype = dtypes[dc.bit_depth]
        except KeyError:
            raise Exception(f"unknown bit depth: {dc.bit_depth}")
        self._sig_shape = (dc.y_pixels_in_detector, dc.x_pixels_in_detector)
        self._meta = DataSetMeta(
            shape=Shape(self._nav_shape + self._sig_shape, sig_dims=2),
            raw_dtype=dtype,
            dtype=dtype,
        )
        return self

    def connect(self):
        """
        Create the zeromq context and PULL socket, and bind it to `data_host`:`data_port`
        """
        self._zmq_ctx = zmq.Context()
        self._socket = self._zmq_ctx.socket(socket_type=zmq.PULL)
        self._socket.connect(f"tcp://{self._data_host}:{self._data_port}")

    def close(self):
        """
        Close the zeromq context and PULL socket
        """
        self._zmq_ctx.destroy()
        self._zmq_ctx = None
        self._socket = None

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
        ec = self.get_api_client()
        mask = ec.detectorConfig("pixel_mask")
        mask_arr = get_darray(mask)
        excluded_pixels = mask_arr > 0
        return CorrectionSet(excluded_pixels=excluded_pixels)

    @contextmanager
    def acquire(self):
        ec = self.get_api_client()
        try:
            self.connect()
            nimages = prod(self.shape.nav)

            ec.setDetectorConfig('ntrigger', 1)
            ec.setDetectorConfig('nimages', 1)
            ec.setDetectorConfig('trigger_mode', self._trigger_mode)
            if self._trigger_mode in ('exte', 'inte'):
                ec.setDetectorConfig('ntrigger', nimages)
            elif self._trigger_mode in ('exts', 'ints'):
                ec.setDetectorConfig('nimages', nimages)

            result = ec.sendDetectorCommand('arm')
            sequence_id = result['sequence id']
            # arm result is something like {'sequence id': 18}

            try:
                self._acq_state = AcquisitionParams(
                    sequence_id=sequence_id,
                    nimages=nimages,
                    trigger_mode=self._trigger_mode,
                )
                # this triggers, either via API or via HW trigger (in which case we
                # don't need to do anything in the trigger function):
                self.trigger()
                yield
            finally:
                self._acq_state = None
        finally:
            self.close()

    def check_valid(self):
        pass

    def need_decode(self, read_dtype, roi, corrections):
        return True  # FIXME: we just do this to get a large tile size

    def adjust_tileshape(self, tileshape, roi):
        depth = 24
        return (depth, *self.meta.shape.sig)
        # return Shape((self._end_idx - self._start_idx, 256, 256), sig_dims=2)

    def get_max_io_size(self):
        # return 12*256*256*8
        # FIXME magic numbers?
        return 24*np.prod(self.meta.shape.sig)*8

    def get_base_shape(self, roi):
        return (1, 1, self.meta.shape.sig[-1])

    @property
    def acquisition_state(self):
        return self._acq_state

    def get_receiver(self):
        # FIXME:
        return ZeroMQReceiver(self._socket, params=self._acq_state)

    def get_partitions(self):
        # FIXME: only works for inline executor or similar, as we are using a zeromq socket
        # which is not safe to be passed to other threads
        num_frames = np.prod(self._nav_shape, dtype=np.uint64)
        num_partitions = int(num_frames // self._frames_per_partition)

        slices = BasePartition.make_slices(self.shape, num_partitions)

        receiver = self.get_receiver()

        for part_slice, start, stop in slices:
            yield DectrisLivePartition(
                start_idx=start,
                end_idx=stop,
                meta=self._meta,
                partition_slice=part_slice,
                receiver=receiver,
            )


class DectrisLivePartition(Partition):
    def __init__(
        self, start_idx, end_idx, partition_slice,
        meta, receiver: Receiver,
    ):
        super().__init__(meta=meta, partition_slice=partition_slice, io_backend=None, decoder=None)
        self._start_idx = start_idx
        self._end_idx = end_idx
        self._receiver = receiver

    def shape_for_roi(self, roi):
        return self.slice.adjust_for_roi(roi).shape

    @property
    def shape(self):
        return self.slice.shape

    @property
    def dtype(self):
        return self.meta.raw_dtype

    def set_corrections(self, corrections):
        self._corrections = corrections

    def _preprocess(self, tile_data, tile_slice):
        if self._corrections is None:
            return
        self._corrections.apply(tile_data, tile_slice)

    def _get_tiles_fullframe(self, tiling_scheme: TilingScheme, dest_dtype="float32", roi=None):
        assert len(tiling_scheme) == 1
        logger.debug("reading up to frame idx %d for this partition", self._end_idx)
        to_read = self._end_idx - self._start_idx
        depth = tiling_scheme.depth
        buf = np.zeros((depth,) + tiling_scheme[0].shape, dtype=dest_dtype)
        buf_idx = 0
        tile_start = self._start_idx
        self._receiver.start()
        while to_read > 0:
            # 1) put frame into tile buffer (including dtype conversion if needed)
            assert buf_idx < depth,\
                    f"buf_idx should be in bounds of buf! ({buf_idx} < ({depth} == {buf.shape[0]}))"
            try:
                frame = next(self._receiver)
                buf[buf_idx] = frame
                buf_idx += 1
                to_read -= 1

                # if buf is full, or the partition is done, yield the tile
                tile_done = buf_idx == depth
                partition_done = to_read == 0
            except StopIteration:
                assert to_read == 0, f"we were still expecting to read {to_read} frames more!"
                tile_done = True
                partition_done = True

            if tile_done or partition_done:
                frames_in_tile = buf_idx
                tile_buf = buf[:frames_in_tile]
                if tile_buf.shape[0] == 0:
                    assert to_read == 0
                    continue  # we are done and the buffer is empty

                tile_shape = Shape(
                    (frames_in_tile,) + tuple(tiling_scheme[0].shape),
                    sig_dims=2
                )
                tile_slice = Slice(
                    origin=(tile_start,) + (0, 0),
                    shape=tile_shape,
                )
                # print(f"yielding tile for {tile_slice}")
                self._preprocess(tile_buf, tile_slice)
                yield DataTile(
                    tile_buf,
                    tile_slice=tile_slice,
                    scheme_idx=0,
                )
                tile_start += frames_in_tile
                buf_idx = 0
        logger.debug("LivePartition.get_tiles: end of method")

    def get_tiles(self, tiling_scheme, dest_dtype="float32", roi=None):
        yield from self._get_tiles_fullframe(tiling_scheme, dest_dtype, roi)

    def __repr__(self):
        return f"<DectrisLivePartition {self._start_idx}:{self._end_idx}>"
