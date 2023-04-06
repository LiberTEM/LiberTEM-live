import logging
import socket
import time
from typing import (
    Generator, Optional, NamedTuple, Dict, Tuple,
)

import numpy as np

from libertem_live.detectors.base.acquisition import AcquisitionTimeout
from .decoders import (
    decode_multi_u1,
    decode_multi_u2,
    decode_multi_r1,
    decode_multi_r6,
    decode_multi_r12,
    decode_quad_r1,
    decode_quad_r6,
    decode_quad_r12,
)


logger = logging.getLogger(__name__)


class AcquisitionHeader(NamedTuple):
    frames_in_acquisition: int
    frames_per_trigger: int
    raw_keys: Dict[str, str]

    @classmethod
    def from_raw(cls, header: bytes) -> "AcquisitionHeader":
        result: Dict[str, str] = {}
        for line in header.decode("latin1").split('\n'):
            try:
                if line.startswith("HDR") or line.startswith("End\t"):
                    continue
                k, v = line.split("\t", 1)
                k = k.rstrip(':')
                v = v.rstrip("\r")
                v = v.rstrip("\n")
            except ValueError:
                logger.warn("error while parsing line %r", line)
                raise
            result[k] = v
        return AcquisitionHeader(
            raw_keys=result,
            frames_in_acquisition=int(result['Frames in Acquisition (Number)']),
            frames_per_trigger=int(result['Frames per Trigger (Number)']),
        )


def get_np_dtype(dtype, bit_depth) -> np.dtype:
    dtype = dtype.lower()
    num_bits = int(dtype[1:])
    if dtype[0] == "u":
        num_bytes = num_bits // 8
        return np.dtype(">u%d" % num_bytes)
    elif dtype[0] == "r":
        if bit_depth == 1:
            return np.dtype("uint64")
        elif bit_depth == 6:
            return np.dtype("uint8")
        elif bit_depth in (12, 24):
            # 24bit raw is two 12bit images after another:
            return np.dtype("uint16")
        else:
            raise NotImplementedError(f"unknown bit depth: {bit_depth}")
    else:
        raise NotImplementedError(f"unknown dtype prefix: {dtype[0]}")


class FrameHeader(NamedTuple):
    header_size_bytes: int
    dtype: np.dtype
    mib_dtype: str   # rXX or uXX
    mib_kind: str  # 'r' or 'u'
    bits_per_pixel: int
    image_size: Tuple[int, int]
    image_size_eff: Tuple[int, int]
    image_size_bytes: int
    sequence_first_image: int
    num_chips: int

    @classmethod
    def from_raw(cls, raw_data: bytes) -> "FrameHeader":
        # FIXME: like in the MIB reader, but no 'filesize' and 'num_images'
        # keys (they can be deduced from a .mib file, but don't make sense
        # in the networked case)
        header = raw_data.decode('ascii', errors='ignore')
        parts = header.split(",")
        header_size_bytes = int(parts[2])
        parts = [p
                for p in header[:header_size_bytes].split(",")
                if '\x00' not in p]
        dtype = parts[6].lower()
        mib_kind = dtype[0]
        image_size = (int(parts[5]), int(parts[4]))
        # FIXME: There can either be threshold values for all chips, or maybe
        # also none. For now, we just make use of the fact that the bit depth
        # is supposed to be the last value.
        bits_per_pixel_raw = int(parts[-1])
        if mib_kind == "u":
            bytes_per_pixel = int(parts[6][1:]) // 8
            image_size_bytes = image_size[0] * image_size[1] * bytes_per_pixel
        elif mib_kind == "r":
            size_factor = {
                1: 1/8,
                6: 1,
                12: 2,
                24: 4,
            }[bits_per_pixel_raw]
            if bits_per_pixel_raw == 24:
                image_size = (image_size[0], image_size[1] // 2)
            image_size_bytes = int(image_size[0] * image_size[1] * size_factor)
        else:
            raise ValueError("unknown kind: %s" % mib_kind)

        num_chips = int(parts[3])
        if num_chips == 4 and mib_kind == "r":
            image_size_eff = (512, 512)
            assert np.prod(image_size_eff) == np.prod(image_size)
        else:
            image_size_eff = image_size

        return FrameHeader(
            header_size_bytes=header_size_bytes,
            dtype=get_np_dtype(parts[6], bits_per_pixel_raw),
            mib_dtype=dtype,
            mib_kind=mib_kind,
            bits_per_pixel=bits_per_pixel_raw,
            image_size=image_size,
            image_size_eff=image_size_eff,
            image_size_bytes=image_size_bytes,
            sequence_first_image=int(parts[1]),
            num_chips=num_chips,
        )


def validate_get_sig_shape(frame_hdr, sig_shape=None):
    image_size = frame_hdr.image_size_eff
    if image_size is None and sig_shape is None:
        raise ValueError(
                'Frame header "image_size" not present and sig shape '
                'not given, cannot determine sig shape'
            )
    elif sig_shape is None:
        return image_size
    else:
        if image_size != sig_shape:
            raise ValueError(
                f'Mismatch between sig_shape {sig_shape} setting and '
                f'received "image_size" header {image_size}.'
            )
        return sig_shape


class MerlinRawFrames:
    def __init__(
        self,
        buffer,
        start_idx: int,
        end_idx: int,
        first_frame_header: FrameHeader,
    ):
        self._buffer = buffer
        self._start_idx = start_idx
        self._end_idx = end_idx
        self._first_frame_header = first_frame_header

    @property
    def num_frames(self):
        return self._end_idx - self._start_idx

    @property
    def start_idx(self):
        return self._start_idx

    @property
    def end_idx(self):
        return self._end_idx

    @property
    def first_frame_header(self):
        return self._first_frame_header

    @property
    def buffer(self):
        return self._buffer

    def decode(self, out_flat: np.ndarray):
        fh = self._first_frame_header
        header_size = int(fh.header_size_bytes) + 15
        itemsize = fh.dtype.itemsize
        bits_pp = fh.bits_per_pixel
        num_rows = fh.image_size[0]
        if fh.mib_kind == 'u':
            # binary:
            if itemsize == 1:
                fn = decode_multi_u1
            elif itemsize == 2:
                fn = decode_multi_u2
            else:
                raise RuntimeError("itemsize %d currently not supported" % itemsize)
        else:
            # raw binary:
            if fh.num_chips == 4:
                if bits_pp == 1:
                    fn = decode_quad_r1
                elif bits_pp == 6:
                    fn = decode_quad_r6
                elif bits_pp == 12:
                    fn = decode_quad_r12
                else:
                    raise RuntimeError(
                        "can't handle quad raw binary %d bits per pixel yet" % bits_pp
                    )
            elif fh.num_chips == 1:
                if bits_pp == 1:
                    fn = decode_multi_r1
                elif bits_pp == 6:
                    fn = decode_multi_r6
                elif bits_pp == 12:
                    fn = decode_multi_r12
                else:
                    raise RuntimeError("can't handle raw binary %d bits per pixel yet" % bits_pp)
            else:
                raise RuntimeError(f"Can't handle num_chips={fh.num_chips}")
        num_frames = self.num_frames
        compat_shape = (num_frames, num_rows, -1)
        input_arr = np.frombuffer(self._buffer, dtype=np.uint8).reshape(
            (num_frames, -1)
        )[:, header_size:].reshape(compat_shape)
        out = out_flat[:num_frames].reshape(compat_shape)
        fn(input_arr, out, header_size, num_frames)


class MerlinDecodedFrames:
    def __init__(
        self,
        buffer,
        start_idx: int,
        end_idx: int,
        first_frame_header: FrameHeader,
    ):
        self._buffer = buffer
        self._start_idx = start_idx
        self._end_idx = end_idx
        self._first_frame_header = first_frame_header

    @property
    def num_frames(self):
        return self._end_idx - self._start_idx

    @property
    def start(self):
        return self._start_idx

    @property
    def stop(self):
        return self._end_idx

    @property
    def first_frame_header(self):
        return self._first_frame_header

    @property
    def buf(self):
        return self._buffer

    @classmethod
    def from_raw(cls, raw_frames: MerlinRawFrames, out: np.ndarray) -> "MerlinDecodedFrames":
        """
        Decode `raw_frames` into `out` and return a `MerlinDecodedFrames`
        object referencing `out`.
        """
        out_flat = out.reshape((out.shape[0], -1))
        raw_frames.decode(
            out_flat=out_flat
        )
        chunk = out[:raw_frames.num_frames]
        return cls(
            buffer=chunk,
            start_idx=raw_frames.start_idx,
            end_idx=raw_frames.end_idx,
            first_frame_header=raw_frames.first_frame_header,
        )


class MerlinRawSocket:
    """
    Read packets of frames from the merlin data socket and
    pass them on in undecoded form
    """
    def __init__(self, host: str = '127.0.0.1', port: int = 6342, timeout: float = 1.0):
        self._host = host
        self._port = port
        self._timeout = timeout
        self._socket = None
        self._is_connected = False
        self._frame_counter = 0

    def connect(self):
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.connect((self._host, self._port))
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.settimeout(self._timeout)
        self._is_connected = True
        return self

    def is_connected(self):
        return self._is_connected

    def read_unbuffered(self, length, cancel_timeout=None):
        """
        read exactly length bytes from the socket

        note: efficiency to be evaluated
        """
        if not self.is_connected():
            raise RuntimeError("can't read without connection")
        assert self._socket is not None
        total_bytes_read = 0
        buf = bytearray(16*1024*1024)
        assert length < len(buf)
        view = memoryview(buf)
        start_time = time.time()
        while total_bytes_read < length:
            try:
                bytes_read = self._socket.recv_into(
                    view[total_bytes_read:],
                    length - total_bytes_read
                )
                if bytes_read == 0:
                    raise EOFError("EOF")
                total_bytes_read += bytes_read
            except socket.timeout:
                pass
            if cancel_timeout is not None and time.time() - start_time > cancel_timeout:
                raise AcquisitionTimeout(f"Timeout after reading {total_bytes_read} bytes.")
        return buf[:total_bytes_read]

    def read_into(self, out):
        """
        read exactly len(out) bytes from the socket
        """
        if not self.is_connected():
            raise RuntimeError("can't read without connection")
        assert self._socket is not None
        length = len(out)
        total_bytes_read = 0
        view = memoryview(out)
        while total_bytes_read < length:
            try:
                bytes_read = self._socket.recv_into(
                    view[total_bytes_read:],
                    length - total_bytes_read
                )
            except socket.timeout:
                return total_bytes_read
            if bytes_read == 0:  # EOF
                return total_bytes_read
            total_bytes_read += bytes_read
        return total_bytes_read

    def read_mpx_length(self, cancel_timeout=None):
        # structure: MPX,<ten digits>,<header>
        hdr = self.read_unbuffered(15, cancel_timeout=cancel_timeout)
        # logger.debug("MPX prefix: %r", hdr)
        assert hdr.startswith(b'MPX,'), "Should start with MPX, first bytes are %r" % hdr[:16]
        parts = hdr.split(b',')
        length = int(parts[1])
        # we already consumed the comma, which seems to be part of the
        # length calculation, that's why we substract 1 here:
        return length - 1

    def read_acquisition_header(self, cancel_timeout: Optional[float] = None) -> AcquisitionHeader:
        # assumption: when we connect, the connection is idle
        # so the first thing we will get is the acquisition header.
        # we read it in an inefficient way, but the header is small,
        # so this should be ok:
        length = self.read_mpx_length(cancel_timeout=cancel_timeout)
        header = self.read_unbuffered(length)
        header = AcquisitionHeader.from_raw(header)
        self._acquisition_header = header
        return header

    def peek_frame_header(self) -> FrameHeader:
        # first, peek only the MPX header part:
        assert self._socket is not None
        while True:
            try:
                buf = self._socket.recv(15, socket.MSG_PEEK)
                assert len(buf) == 15
                break
            except socket.timeout:
                pass
        assert len(buf) == 15
        parts = buf.split(b',')
        length = int(parts[1])

        # now, peek enough to read the frame header:
        buf = b''
        peek_length = min(15 + length, 4096)
        while len(buf) < peek_length:
            # need to repeat, as the first peek can fail to give the full length message:
            buf = self._socket.recv(peek_length, socket.MSG_PEEK)
        frame_header = FrameHeader.from_raw(buf[15:])
        return frame_header

    def close(self):
        if self._is_connected:
            assert self._socket is not None
            self._socket.close()
            self._socket = None
            self._is_connected = False

    def __enter__(self):
        self.connect()

    def __exit__(self, type, value, traceback):
        self.close()

    def drain(self):
        """
        read data from the data socket until we hit the timeout; returns
        the number of bytes drained
        """
        assert self._socket is not None
        bytes_read = 0
        # read from the socket until we hit the timeout:
        old_timeout = self._socket.gettimeout()
        self._socket.settimeout(0.1)
        try:
            while True:
                data = self._socket.recv(4096)
                bytes_read += len(data)
        except socket.timeout:
            return bytes_read
        finally:
            self._socket.settimeout(old_timeout)


class MerlinFrameStream:
    """
    This class takes over reading from the `MerlinRawSocket` once the
    `AcquisitionHeader` has been read.

    It first peeks the first frame header, and then reads multiple frames
    in stacks of mostly fixed size.
    """
    def __init__(
        self,
        raw_socket: MerlinRawSocket,
        acquisition_header: AcquisitionHeader,
        first_frame_header: FrameHeader,
    ):
        self._raw_socket = raw_socket
        self._acquisition_header = acquisition_header
        self._first_frame_header = first_frame_header
        self._frame_counter = 0

    @classmethod
    def from_frame_header(
        cls,
        raw_socket: MerlinRawSocket,
        acquisition_header: AcquisitionHeader
    ) -> "MerlinFrameStream":
        first_frame_header = raw_socket.peek_frame_header()
        return cls(
            raw_socket=raw_socket,
            acquisition_header=acquisition_header,
            first_frame_header=first_frame_header,
        )

    def read_multi_frames(self, input_buffer, num_frames=32, read_upto_frame=None):
        """
        Returns `False` on timeout, `True` in case we are done and don't need to
        read any more frames (according to `read_upto_frame`), or a
        `MerlinRawFrames` object

        On EOF or timeout, can read less than `num_frames`. In that case, `buffer` is sliced
        to only contain decoded data. `out` will not be fully overwritten in this case
        and can contain garbage at the end.

        `read_upto_frame` can be used to only read up to a total number of frames.
        Once this number is reached, we behave the same way as if we had reached EOF.
        """
        assert self._first_frame_header is not None
        header_size = int(self._first_frame_header.header_size_bytes) + 15
        bytes_per_frame = header_size + self._first_frame_header.image_size_bytes

        input_buffer = memoryview(input_buffer)
        if read_upto_frame is not None:
            num_frames = min(num_frames, read_upto_frame - self._frame_counter)
            input_buffer = input_buffer[:num_frames * bytes_per_frame]
            if num_frames == 0:
                return True  # we are done.
        bytes_read = self._raw_socket.read_into(input_buffer)
        frames_read = bytes_read // bytes_per_frame
        if bytes_read % bytes_per_frame != 0:
            raise EOFError(
                "input data stream truncated (%d,%d)" % (
                    bytes_read, bytes_per_frame
                )
            )

        start_idx = self._frame_counter

        self._frame_counter += frames_read
        if bytes_read == 0:
            # timeout or EOF without any data read:
            return False

        end_idx = start_idx + frames_read

        return MerlinRawFrames(
            input_buffer[:bytes_read],
            start_idx,
            end_idx,
            self._first_frame_header,
        )

    def get_input_buffer(self, num_frames: int) -> bytearray:
        assert self._first_frame_header is not None
        header_size = int(self._first_frame_header.header_size_bytes) + 15
        image_size = int(self._first_frame_header.image_size_bytes)
        read_size = num_frames*(header_size + image_size)
        input_bytes = bytearray(read_size)
        return input_bytes

    def get_first_frame_header(self) -> FrameHeader:
        return self._first_frame_header


class MerlinDataSource:
    def __init__(
        self,
        host,
        port,
        pool_size=2,
        sig_shape=None,
        timeout: Optional[float] = None
    ):
        self._sig_shape = sig_shape
        self.socket = MerlinRawSocket(
            host=host,
            port=port,
            timeout=timeout,
        )

    def __enter__(self):
        self.socket.__enter__()

    def __exit__(self, *args, **kwargs):
        self.socket.__exit__(*args, **kwargs)

    def inline_stream(self, read_dtype=np.float32, chunk_size=10, num_frames=None):
        chunks = self._read_and_decode(
            read_dtype=read_dtype,
            chunk_size=chunk_size,
            num_frames=num_frames,
        )
        for chunk in chunks:
            yield chunk.buf

    def _read_and_decode(
        self, read_dtype=np.float32, chunk_size=10, num_frames=None
    ) -> Generator[MerlinDecodedFrames, None, None]:
        acq_header = self.socket.read_acquisition_header()
        stream = MerlinFrameStream.from_frame_header(
            raw_socket=self.socket,
            acquisition_header=acq_header,
        )

        logger.info(acq_header)

        frame_hdr = stream.get_first_frame_header()
        logger.info(frame_hdr)

        sig_shape = validate_get_sig_shape(frame_hdr, self._sig_shape)
        out = np.zeros((chunk_size,) + sig_shape, dtype=read_dtype)
        input_buffer = stream.get_input_buffer(num_frames=chunk_size)

        while True:
            res = stream.read_multi_frames(
                num_frames=chunk_size,
                input_buffer=input_buffer,
                read_upto_frame=num_frames,
            )
            if not res:
                break
            if res is True:
                break
            yield MerlinDecodedFrames.from_raw(res, out)

    def stream(self, num_frames=None, chunk_size=11, read_dtype=np.float32):
        """
        Examples
        --------

        >>> source = MerlinDataSource(
        ...     host='127.0.0.1',
        ...     port=MERLIN_DATA_PORT,
        ... )
        >>> result = np.zeros((256, 256), dtype=np.float32)
        >>> with source:
        ...     for chunk in source.stream(num_frames=128*128, chunk_size=16):
        ...         result += chunk.buf.sum(axis=0)
        """
        chunks = self._read_and_decode(
            read_dtype=read_dtype,
            chunk_size=chunk_size,
            num_frames=num_frames,
        )
        yield from chunks
