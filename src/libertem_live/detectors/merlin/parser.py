import time
import queue
import threading 

from concurrent.futures import ThreadPoolExecutor

import numba

import numpy as np
from libertem.io.dataset.mib import MIBDecoder
from libertem.common.buffers import BufferPool
from .data import EOFError


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


class ReadThread(threading.Thread):
    def __init__(self, backend, out_queue, *args, **kwargs):
        self._backend = backend
        self._out_queue = out_queue
        self._stop_event = threading.Event()
        self._have_header = threading.Event()
        self._cached_frame_header = None
        super().__init__(name='ReadThread', *args, **kwargs)

    def stop(self):
        print("stopping read thread")
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
            while True:
                if self.is_stopped():
                    print("is_stopped, breaking")
                    break
                try:
                    length = self._backend.read_mpx_length()
                    frame_data = self._backend.read(length)
                except EOFError:
                    break
                if self._cached_frame_header is None:
                    self._cached_frame_header = _parse_frame_header(frame_data)
                    self._have_header.set()
                self._out_queue.put(
                    frame_data
                )
        finally:
            print("at the finally of run()")
            self.stop()
        print("at the end of run()")


class MIBParser:
    def __init__(self, backend, read_dtype, workers=16):
        self._backend = backend
        self._read_dtype = read_dtype
        self._cached_frame_header = None
        self._decode = None
        self._pool = BufferPool()
        self._queue_raw = queue.Queue()
        self._read_thread = None
        self._workers = workers

    def get_decode(self):
        # FIXME: decode doesn't work unchanged
        if self._decode is None:
            frame_hdr = self._cached_frame_header
            decoder = MIBDecoder(
                kind=frame_hdr['mib_kind'],
                dtype=None,
                bit_depth=frame_hdr['bits_per_pixel']
            )
            self._decode = decoder.get_decode(
                native_dtype=frame_hdr['dtype'],
                read_dtype=self._read_dtype
            )
        return self._decode

    def _get_frame_raw(self):
        length = self._backend.read_mpx_length()
        frame_data = self._backend.read(length)
        if self._cached_frame_header is None:
            self._cached_frame_header = _parse_frame_header(frame_data)
        return frame_data

    def get_frame(self):
        frame_data = self._get_frame_raw()
        frame_hdr = self._cached_frame_header
        # decode = self.get_decode()
        # FIXME: raw decoding? how to put detector into raw mode?
        header_size = frame_hdr['header_size_bytes']
        frame_only = frame_data[header_size:]
        if frame_hdr['mib_kind'] == 'u':
            frame = np.frombuffer(
                frame_data[header_size:],
                dtype=frame_hdr['dtype']
            ).reshape(256, 256)
            if frame.dtype != np.dtype(self._read_dtype):
                frame = frame.astype(self._read_dtype)
        elif frame_hdr['mib_kind'] == 'r':
            # FIXME: get rid of allocation?
            # with self._pool.empty((256, 256), dtype=self._read_dtype) as frame:
            frame = np.empty((256, 256), dtype=self._read_dtype)
            # FIXME: other raw formats
            decode_r6_swap(inp=frame_only, out=frame.reshape((-1,)))
        return frame

    def get_many(self, num_frames, out):
        """
        read `num_frames` into `out`
        """
        t0 = time.time()
        if self._read_thread is None:
            self.start()
        out_flat = out.reshape((num_frames, -1))
        frames_done = 0  # done means decoded to the out buffer
        # read first frame here, for header
        if self._cached_frame_header is None:
            self._read_thread._have_header.wait()
            self._cached_frame_header = self._read_thread._cached_frame_header
        frame_hdr = self._cached_frame_header
        header_size = frame_hdr['header_size_bytes']
        # print("preamble: %.3f" % (time.time() - t0,))
        if frame_hdr['mib_kind'] == 'u':
            decode_fn = get_decode_np(
                dtype=frame_hdr['dtype'],
            )
        else:
            bpp = frame_hdr['bits_per_pixel']
            if bpp == 6:
                decode_fn = decode_r6_swap
            elif bpp == 1:
                decode_fn = decode_r1_swap
            else:
                raise ValueError("unknown bit depth: %d" % bpp)
        # overlap decoding and receiving:
        try:
            with ThreadPoolExecutor(max_workers=self._workers) as tp:
                while frames_done < num_frames:
                    # get a chunk of raw frames and decode them:
                    chunk_size = min(num_frames - frames_done, 128)
                    frames = []
                    idx = 0
                    t0 = time.time()
                    while len(frames) < chunk_size:
                        try:
                            frames.append(
                                (idx, self._queue_raw.get(timeout=1))
                            )
                            idx += 1
                            # print("reading from queue... (%d/%d)" % (len(frames), chunk_size))
                        except queue.Empty as e:
                            print("queue is empty, waiting... (%d/%d)" % (len(frames), chunk_size))
                            if self._read_thread.is_stopped():
                                raise e
                    # print("get frames: %.3f" % (time.time() - t0,))
                    # decode directly into the out buffer
                    t0 = time.time()
                    for result in tp.map(
                        lambda frame_and_idx: decode_fn(
                            frame_and_idx[1][header_size:],
                            out=out[frames_done + frame_and_idx[0]]
                        ),
                        frames,
                    ):
                        pass
                    t1 = time.time()
                    # print("decode: %.3f" % (t1 - t0,))
                    frames_done += chunk_size
        finally:
            pass
        # print("get_many done: %d" % frames_done)

    def stop(self):
        self._read_thread.stop()
        self._read_thread.join()    

    def start(self):
        self._read_thread = ReadThread(
            self._backend, self._queue_raw,
        )
        self._read_thread.start()
    
    def warmup(self):
        # some more input data than needed:
        # frame_only = np.empty((256*256), dtype=np.uint64).tobytes()
        frame_only = np.empty((256*256), dtype=np.uint8).tobytes()
        with self._pool.empty((256, 256), dtype=self._read_dtype) as frame:
            decode_r6_swap(inp=frame_only, out=frame.reshape((-1,)))


def get_decode_np(dtype):
    def decode_np(inp, out):
        out[:] = np.frombuffer(
            inp,
            dtype=dtype,
        ).reshape((256, 256))
    return decode_np


@numba.njit(inline='always', nogil=True)
def decode_r6_swap(inp, out):
    """
    RAW 6bit format: the pixels need to be re-ordered in groups of 8. `inp`
    should have dtype uint8 (or bytes).
    """
    for i in range(out.shape[0]):
        col = i % 8
        pos = i // 8
        out_pos = (pos + 1) * 8 - col - 1
        out[out_pos] = inp[i]


@numba.njit(inline='always', nogil=True)
def decode_r1_swap(inp, out):
    """
    RAW 1bit format: each bit is actually saved as a single bit. 64 bits
    need to be unpacked together.
    """
    for stripe in range(len(inp) // 8):
        for byte in range(8):
            inp_byte = inp[(stripe + 1) * 8 - (byte + 1)]
            for bitpos in range(8):
                out[64 * stripe + 8 * byte + bitpos] = (inp_byte >> bitpos) & 1
