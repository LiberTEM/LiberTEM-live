import logging
import socket
import threading
import contextlib
import queue
import time

import numba
import numpy as np

from libertem.io.dataset.base.decode import byteswap_2_decode

from libertem_live.detectors.base.acquisition import AcquisitionTimeout

logger = logging.getLogger(__name__)


def get_np_dtype(dtype, bit_depth):
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
            raise NotImplementedError("unknown bit depth: %s" % bit_depth)


def _parse_frame_header(raw_data):
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

    return {
        'header_size_bytes': header_size_bytes,
        'dtype': get_np_dtype(parts[6], bits_per_pixel_raw),
        'mib_dtype': dtype,
        'mib_kind': mib_kind,
        'bits_per_pixel': bits_per_pixel_raw,
        'image_size': image_size,
        'image_size_bytes': image_size_bytes,
        'sequence_first_image': int(parts[1]),
    }


@numba.njit(nogil=True, cache=True, parallel=True)
def decode_multi_u2(input_bytes, out, header_size_bytes, num_frames):
    """
    Decode multiple >u2 frames
    """
    out = out.reshape((num_frames, -1))
    frame_size_bytes = 256 * 256 * 2
    for i in numba.prange(num_frames):
        start_offset = header_size_bytes*(i+1)+i*frame_size_bytes
        end_offset = start_offset + frame_size_bytes
        in_for_frame = input_bytes[start_offset:end_offset]
        byteswap_2_decode(in_for_frame, out[i])


@numba.njit(nogil=True, cache=True, parallel=True)
def decode_multi_u1(input_bytes, out, header_size_bytes, num_frames):
    """
    Decode multiple u1 frames
    """
    out = out.reshape((num_frames, -1))
    frame_size_bytes = 256 * 256
    for i in numba.prange(num_frames):
        start_offset = header_size_bytes*(i+1)+i*frame_size_bytes
        end_offset = start_offset + frame_size_bytes
        in_for_frame = input_bytes[start_offset:end_offset]
        out[i] = in_for_frame
        # for j in range(end_offset - start_offset):
        #     out[i, j] = in_for_frame[j]


@numba.jit(inline='always', cache=True)
def decode_r1_swap(inp, out, idx):
    """
    RAW 1bit format: each bit is actually saved as a single bit. 64 bits
    need to be unpacked together.
    """
    for stripe in range(inp.shape[0] // 8):
        for byte in range(8):
            inp_byte = inp[(stripe + 1) * 8 - (byte + 1)]
            for bitpos in range(8):
                out[idx, 64 * stripe + 8 * byte + bitpos] = (inp_byte >> bitpos) & 1


@numba.jit(nogil=True, cache=True, parallel=True)
def decode_multi_r1(input_bytes, out, header_size_bytes, num_frames):
    out = out.reshape((num_frames, -1))
    frame_size_bytes = 256 * 256 // 8
    for i in numba.prange(num_frames):
        start_offset = header_size_bytes*(i+1)+i*frame_size_bytes
        end_offset = start_offset + frame_size_bytes
        in_for_frame = input_bytes[start_offset:end_offset]
        decode_r1_swap(in_for_frame, out, i)


@numba.njit(inline='always', cache=True)
def decode_r6_swap(inp, out, idx):
    """
    RAW 6bit format: the pixels need to be re-ordered in groups of 8. `inp`
    should have dtype uint8.
    """
    for i in range(out.shape[1]):
        col = i % 8
        pos = i // 8
        out_pos = (pos + 1) * 8 - col - 1
        out[idx, out_pos] = inp[i]


@numba.jit(nogil=True, cache=True, parallel=True)
def decode_multi_r6(input_bytes, out, header_size_bytes, num_frames):
    out = out.reshape((num_frames, -1))
    frame_size_bytes = 256 * 256
    for i in numba.prange(num_frames):
        start_offset = header_size_bytes*(i+1)+i*frame_size_bytes
        end_offset = start_offset + frame_size_bytes
        in_for_frame = input_bytes[start_offset:end_offset]
        decode_r6_swap(in_for_frame, out, i)


@numba.njit(inline='always', cache=True)
def decode_r12_swap(inp, out, idx):
    """
    RAW 12bit format: the pixels need to be re-ordered in groups of 4. `inp`
    should be an uint8 view on big endian 12bit data (">u2")
    """
    for i in range(out.shape[1]):
        col = i % 4
        pos = i // 4
        out_pos = (pos + 1) * 4 - col - 1
        out[idx, out_pos] = (inp[i * 2] << 8) + (inp[i * 2 + 1] << 0)


@numba.jit(nogil=True, cache=True, parallel=True)
def decode_multi_r12(input_bytes, out, header_size_bytes, num_frames):
    out = out.reshape((num_frames, -1))
    frame_size_bytes = 2 * 256 * 256
    for i in numba.prange(num_frames):
        start_offset = header_size_bytes*(i+1)+i*frame_size_bytes
        end_offset = start_offset + frame_size_bytes
        in_for_frame = input_bytes[start_offset:end_offset]
        decode_r12_swap(in_for_frame, out, i)


@numba.njit(inline='always', cache=True)
def decode_r24_swap(inp, out, idx):
    """
    RAW 24bit format: a single 24bit consists of two frames that are encoded
    like the RAW 12bit format, the first contains the most significant bits.

    So after a frame header, there are (512, 256) >u2 values, which then
    need to be shuffled like in `decode_r12_swap`.

    This decoder function only works together with mib_r24_get_read_ranges
    which generates twice as many read ranges than normally.
    """
    for i in range(out.shape[1]):
        col = i % 4
        pos = i // 4
        out_pos = (pos + 1) * 4 - col - 1
        out_val = np.uint32((inp[i * 2] << 8) + (inp[i * 2 + 1] << 0))
        if idx % 2 == 0:  # from first frame: most significant bits
            out_val = out_val << 12
        out[idx // 2, out_pos] += out_val


class MerlinDataSocket:
    def __init__(self, host='127.0.0.1', port=6342, timeout=1.0):
        self._host = host
        self._port = port
        self._timeout = timeout
        self._socket = None
        self._acquisition_header = None
        self._is_connected = False
        self._read_lock = threading.Lock()
        self._frame_counter = 0

    def connect(self):
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.connect((self._host, self._port))
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.settimeout(self._timeout)
        self._is_connected = True
        self._frame_counter = 0

    def is_connected(self):
        return self._is_connected

    def read_unbuffered(self, length, cancel_timeout=None):
        """
        read exactly length bytes from the socket

        note: efficiency to be evaluated
        """
        if not self.is_connected():
            raise RuntimeError("can't read without connection")
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

    def _read_acquisition_header(self, cancel_timeout=None):
        # assumption: when we connect, the connection is idle
        # so the first thing we will get is the acquisition header.
        # we read it in an inefficient way, but the header is small,
        # so this should be ok:
        length = self.read_mpx_length(cancel_timeout=cancel_timeout)
        header = self.read_unbuffered(length)
        header = self._parse_acq_header(header)
        self._acquisition_header = header
        return header

    def get_acquisition_header(self):
        return self._acquisition_header

    def read_headers(self, cancel_timeout=None):
        """
        Read acquisition header, and peek first frame header

        The acquisition header is consumed, the first frame header will
        be kept in the socket queue, and can be read regularly afterwards.
        """
        self._acquisition_header = self._read_acquisition_header(cancel_timeout=cancel_timeout)
        self._first_frame_header = self._peek_frame_header()
        self._frame_counter = self._first_frame_header['sequence_first_image'] - 1
        logger.info(f"got headers; frame offset = {self._frame_counter}")
        return self._acquisition_header, self._first_frame_header

    def get_input_buffer(self, num_frames):
        header_size = int(self._first_frame_header['header_size_bytes']) + 15
        image_size = int(self._first_frame_header['image_size_bytes'])
        read_size = num_frames*(header_size + image_size)
        input_bytes = bytearray(read_size)
        return input_bytes

    def get_out_buffer(self, num_frames, dtype=np.float32):
        return np.zeros((num_frames, 256, 256), dtype=dtype)

    def read_multi_frames(self, out, input_buffer, num_frames=32, read_upto_frame=None, timeout=-1):
        """
        Returns `False` on timeout, `True` in case we are done and don't need to
        read any more frames (according to `read_upto_frame`), or a tuple
        `(buffer, frame_idx_start, frame_idx_end)`

        On EOF or timeout, can read less than `num_frames`. In that case, `buffer` is sliced
        to only contain decoded data. `out` will not be fully overwritten in this case
        and can contain garbage at the end.

        `read_upto_frame` can be used to only read up to a total number of frames.
        Once this number is reached, we behave the same way as if we had reached EOF.
        """
        out_flat = out.reshape((num_frames, -1))
        header_size = int(self._first_frame_header['header_size_bytes']) + 15
        bytes_per_frame = header_size + self._first_frame_header['image_size_bytes']

        lock_success = self._read_lock.acquire(timeout=timeout)
        if not lock_success:
            return False
        try:
            input_buffer = memoryview(input_buffer)
            if read_upto_frame is not None:
                num_frames = min(num_frames, read_upto_frame - self._frame_counter)
                input_buffer = input_buffer[:num_frames * bytes_per_frame]
                if num_frames == 0:
                    return True  # we are done.
            bytes_read = self.read_into(input_buffer)
            frames_read = bytes_read // bytes_per_frame
            if bytes_read % bytes_per_frame != 0:
                raise EOFError(
                    "input data stream truncated (%d,%d)" % (
                        bytes_read, bytes_per_frame
                    )
                )
            self._frame_counter += frames_read
            # save frame_counter while we hold the lock:
            frame_counter = self._frame_counter
            if bytes_read == 0:
                # timeout or EOF without any data read:
                return False
        finally:
            self._read_lock.release()
        self.decode(input_buffer[:bytes_read], out_flat[:frames_read], header_size, frames_read)
        return (out[:frames_read], frame_counter - frames_read, frame_counter)

    def decode(self, input_buffer, out_flat, header_size, num_frames):
        fh = self._first_frame_header
        itemsize = fh['dtype'].itemsize
        bits_pp = fh['bits_per_pixel']
        if fh['mib_kind'] == 'u':
            # binary:
            if itemsize == 1:
                fn = decode_multi_u1
            elif itemsize == 2:
                fn = decode_multi_u2
            else:
                raise Exception("itemsize %d currently not supported" % itemsize)
        else:
            # raw binary:
            if bits_pp == 1:
                fn = decode_multi_r1
            elif bits_pp == 6:
                fn = decode_multi_r6
            elif bits_pp == 12:
                fn = decode_multi_r12
            else:
                raise Exception("can't handle %d bits per pixel yet" % bits_pp)
        input_arr = np.frombuffer(input_buffer, dtype=np.uint8)
        fn(input_arr, out_flat[:num_frames], header_size, num_frames)

    def _peek_frame_header(self):
        # first, peek only the MPX header part:
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
        frame_header = _parse_frame_header(buf[15:])
        return frame_header

    def _parse_acq_header(self, header):
        result = {}
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
        return result

    def close(self):
        self._socket.close()
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
        bytes_read = 0
        # read from the socket until we hit the timeout:
        while True:
            try:
                data = self._socket.recv(4096)
                bytes_read += len(data)
            except socket.timeout:
                return bytes_read


class ResultWrap:
    def __init__(self, start, stop, buf):
        self._consumed = threading.Event()
        self.start = start
        self.stop = stop
        self.buf = buf[:stop - start]
        assert buf.shape[0] == stop - start

    def is_released(self, timeout):
        # called from `ReaderThread` - after data is consumed,
        # we are free to reuse our buffer
        return self._consumed.wait(timeout)

    def release(self):
        self._consumed.set()


# FIXME: use ErrThreadMixin and maybe_raise in the pool
class ReaderThread(threading.Thread):
    def __init__(self, backend, out_queue, chunk_size, default_timeout=0.2, read_dtype=np.float32,
                 read_upto_frame=None, *args, **kwargs):
        super().__init__(name='ReaderThread', *args, **kwargs)
        self._stop_event = threading.Event()
        self._eof_event = threading.Event()
        self._backend = backend
        self._chunk_size = chunk_size
        self._out_queue = out_queue
        self._default_timeout = default_timeout
        self._read_dtype = read_dtype
        self._read_upto_frame = read_upto_frame

    def stop(self):
        self._stop_event.set()

    def is_stopped(self):
        return self._stop_event.is_set()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args, **kwargs):
        self.stop()

    def run(self):
        try:
            out = self._backend.get_out_buffer(self._chunk_size, dtype=self._read_dtype)
            input_buffer = self._backend.get_input_buffer(num_frames=self._chunk_size)
            should_exit = False
            while not should_exit:
                if self.is_stopped():
                    break
                res = self._backend.read_multi_frames(
                    out=out,
                    num_frames=self._chunk_size,
                    input_buffer=input_buffer,
                    timeout=self._default_timeout,
                    read_upto_frame=self._read_upto_frame,
                )
                if res is False:
                    continue  # timeout, give main thread the chance to stop us
                if res is True:
                    break  # we are done
                res_buffer, res_start, res_stop = res

                # EOF, no frames left on the socket:
                if res_stop - res_start == 0:
                    break

                wrapped = ResultWrap(res_start, res_stop, res_buffer)

                # retry until there is space in the output queue, or the thread is stopped:
                while True:
                    try:
                        self._out_queue.put(wrapped, timeout=self._default_timeout)
                        break
                    except queue.Full:
                        if self.is_stopped():
                            should_exit = True
                            break

                # retry until value is consumed
                while not wrapped.is_released(timeout=self._default_timeout):
                    if self.is_stopped():  # break out of outer loop
                        should_exit = True
                        break
        finally:
            self.stop()  # make sure the stopped flag is set in any case


class ReaderPoolImpl:
    def __init__(self, backend, pool_size, chunk_size, read_upto_frame):
        self._pool_size = pool_size
        self._backend = backend
        self._chunk_size = chunk_size
        self._out_queue = queue.Queue()  # TODO: possibly limit size?
        self._threads = None
        self._read_upto_frame = read_upto_frame

    def __enter__(self):
        self._threads = []
        for i in range(self._pool_size):
            t = ReaderThread(
                backend=self._backend,
                chunk_size=self._chunk_size,
                out_queue=self._out_queue,
                read_upto_frame=self._read_upto_frame,
            )
            t.start()
            self._threads.append(t)
        return self

    def __exit__(self, *args, **kwargs):
        logger.debug("ReaderPoolImpl.__exit__: stopping threads")
        for t in self._threads:  # TODO: handle errors on stopping/joining? re-throw exceptions?
            t.stop()
            logger.debug("ReaderPoolImpl: stop signal set")
            t.join()
            logger.debug("ReaderPoolImpl: thread joined")
        logger.debug("ReaderPoolImpl.__exit__: threads stopped")

    @contextlib.contextmanager
    def get_result(self):
        while True:
            try:
                res = self._out_queue.get(timeout=0.2)
                yield res
                res.release()
                return
            except queue.Empty:
                if self.should_stop():
                    yield None
                    return

    def should_stop(self):
        return any(
            t.is_stopped()
            for t in self._threads
        )


class ReaderPool:
    def __init__(self, backend, pool_size):
        self._backend = backend
        self._pool_size = pool_size

    def get_impl(self, chunk_size=10, read_upto_frame=None):
        """
        Returns a new `ReaderPoolImpl`, which will read up to
        the frame index given in `read_upto_frame`, or all frames
        in case it is `None`.
        """
        return ReaderPoolImpl(
            backend=self._backend,
            pool_size=self._pool_size,
            chunk_size=chunk_size,
            read_upto_frame=read_upto_frame,
        )


class MerlinDataSource:
    def __init__(self, host, port, pool_size=2):
        self.socket = MerlinDataSocket(host=host, port=port)
        self.pool = ReaderPool(backend=self.socket, pool_size=pool_size)

    def __enter__(self):
        self.socket.__enter__()

    def __exit__(self, *args, **kwargs):
        self.socket.__exit__(*args, **kwargs)

    def inline_stream(self, read_dtype=np.float32, chunk_size=10):
        self.socket.read_headers()
        hdr = self.socket.get_acquisition_header()
        logger.info(hdr)

        out = self.socket.get_out_buffer(chunk_size, dtype=read_dtype)
        input_buffer = self.socket.get_input_buffer(num_frames=chunk_size)

        while True:
            res = self.socket.read_multi_frames(
                out=out,
                num_frames=chunk_size,
                input_buffer=input_buffer
            )
            if not res:
                break
            if res is True:
                break
            buf, frame_idx_start, frame_idx_end = res
            if frame_idx_end - frame_idx_start == 0:
                break
            yield buf

    def stream(self, num_frames=None, chunk_size=11):
        """
        Examples
        --------

        >>> source = MerlinDataSource(...)  # doctest: +SKIP
        >>> with source:  # doctest: +SKIP
        ...     for _ in source.stream():  # doctest: +SKIP
        ...         ...
        """
        self.socket.read_headers()
        hdr = self.socket.get_acquisition_header()
        logger.info(hdr)
        pool = self.pool.get_impl(
            read_upto_frame=num_frames,
            chunk_size=chunk_size,
        )
        with pool:
            while True:
                with pool.get_result() as res_wrapped:
                    if pool.should_stop():
                        break
                    yield res_wrapped
