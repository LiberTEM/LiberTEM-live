import multiprocessing as mp
import os
import time
import socket
import threading
import contextlib
import logging
import enum
from collections import namedtuple

import numpy as np
import numba
import zmq

from libertem.io.dataset.k2is import DataBlock, SHUTTER_ACTIVE_MASK
from libertem.common.buffers import bytes_aligned, zeros_aligned
from libertem_live.utils.net import mcast_socket
from libertem_live.detectors.k2is.decode import (
    decode_bulk_uint12_le, PacketHeader, decode_header_into, decode_uint12_le, make_PacketHeader
)
from libertem.io.dataset.base.tiling import TilingScheme, DataTile
from libertem.common import Shape, Slice
from libertem.io.dataset.base import Partition, DataSetMeta
from libertem.udf.base import UDFRunner
from libertem.executor.base import Environment
from ..common import ErrThreadMixin, send_serialized, recv_serialized
from .state import (
    EventReplicaClient,
    ProcessingDoneEvent,
    ProcessingState,
    CamConnectedEvent, CamDisconnectedEvent,
)

GROUP = '225.1.1.1'

logger = logging.getLogger(__name__)


def warmup():
    packet_1 = make_packet(0, 0, *block_xy(0))
    packet_2 = make_packet(1, 0, *block_xy(0))
    warmup_buf_inp = bytes_aligned(0x5758*2)
    warmup_buf_inp[:] = packet_1 + packet_2
    warmup_buf_inp = np.array(warmup_buf_inp, dtype=np.uint8)

    header = make_PacketHeader()
    for dtype in [np.uint16]:  # , np.float32, np.float64]:
        bufs = [
            make_tile((1, 1860, 256), 0, dtype=dtype)
        ]
        decoder_state = make_DecoderState(32)
        decoder_state.first_frame_id[0] = 0
        decoder_state.recent_frame_id[0] = 0
        decoder_state.end_after_idx[0] = 1
        decoder_state.end_dataset_after_idx[0] = 2
        process_packets(
            bufs_inout=numba.typed.List(bufs),
            header_inout=header,
            decoder_state=decoder_state,
            packets=warmup_buf_inp,
            num_packets=2
        )


class FakeEnvironment(Environment):
    @contextlib.contextmanager
    def enter(self):
        yield


class SyncState:
    """
    Shared state for synchronization between processes
    """

    def __init__(self, num_processes):
        self.num_processes = num_processes
        self.first_frame_id = mp.Value('I', 0)
        self.sync_msg_seen = mp.Value('i', 0)
        self.sync_done = mp.Event()
        self.initial_sync = mp.Value('i', 0)
        self.reset_barrier = mp.Barrier(num_processes)

    def set_initial_sync(self):
        """
        Call this function to indicate that data from a sector
        has been received:
        """
        with self.initial_sync.get_lock():
            self.initial_sync.value += 1
            return self.initial_sync.value

    def set_first_frame_id(self, frame_id):
        """
        Each process should set the `frame_id` of the first full frame they observe
        """
        with self.first_frame_id.get_lock(), self.sync_msg_seen.get_lock():
            if self.sync_done.is_set():
                raise ValueError("sync already done")
            self.first_frame_id.value = max(self.first_frame_id.value, frame_id)
            self.sync_msg_seen.value += 1
            if self.sync_msg_seen.value == self.num_processes:
                self.sync_done.set()

    def get_first_frame_id(self):
        if not self.sync_done.is_set():
            raise RuntimeError("get_first_frame_id called with incomplete sync procedure")
        return self.first_frame_id.value

    def reset(self):
        # NOTE: blocking here, as timeout breaks the barrier state
        # this is the only place we are blocking for a while
        logger.info("SyncState.reset before barrier")
        ret = self.reset_barrier.wait(timeout=10)
        logger.info(f"SyncState.reset after barrier {ret}")
        if ret == 0:
            logger.info("SyncState.reset ret==0")
            with self.first_frame_id.get_lock(),\
                    self.sync_msg_seen.get_lock(),\
                    self.initial_sync.get_lock():
                self.first_frame_id.value = 0
                self.sync_msg_seen.value = 0
                self.initial_sync.value = 0
                self.sync_done.clear()
                self.reset_barrier.reset()


# FIXME later...
ts = TilingScheme.make_for_shape(
    tileshape=Shape((1, 930, 16), sig_dims=2),
    dataset_shape=Shape((1, 2*930, 16*8*16), sig_dims=2)
)


class PlaceholderPartition(Partition):
    def __init__(
        self, meta: DataSetMeta, partition_slice: Slice, tiles, start_frame: int, num_frames: int,
    ):
        self._tiles = tiles
        self._start_frame = start_frame
        self._num_frames = num_frames
        super().__init__(
            meta=meta,
            partition_slice=partition_slice,
            io_backend=None,
        )

    def get_tiles(self, tiling_scheme, dest_dtype=np.float32, roi=None):
        assert roi is None

        # FIXME: how stop after processing `num_frames`?
        for tile in self._tiles:
            yield tile

    def need_decode(self, read_dtype, roi, corrections):
        return True

    def get_base_shape(self, roi):
        return (930, 16)

    def adjust_tileshape(self, tileshape, roi):
        return tileshape  # FIXME

    def set_corrections(self, corrections):
        self._corrections = corrections


class SectorState(enum.IntEnum):
    # the UDP sockets are listening for multicast packets:
    INIT = 1

    # we received the first UDP packet, so we consider outselves "connected":
    CONNECTED = 2

    # we are ready once we know what UDFs we should run, and other parameters:
    READY = 3

    # the user has indicated that processing should start; we should start
    # synchronization to a common `first_frame_id` with the shutter active flag set:
    PRIMED = 4

    # we are in the inner processing loop; in continuous mode, we stay in this
    # state until processining is cancelled by the user. Otherwise, we transition
    # to the DONE state once the pre-defined `nav_shape` is fully processed.
    PROCESSING = 5

    # we are not in continuous mode, and the full `nav_shape` is processed.
    # we need to dispatch an event to the main state management indicating that
    # this sector is done:
    DONE = 6


class EndOfStreamMarker:
    def __init__(self, idx: int):
        self.idx = idx


class DisconnectedError(Exception):
    pass


Carry = namedtuple('Carry', ['data', 'packet_count'])


# To be safe we allocate twice the space of a block so that we can completely
# carry one buffer into another if there is a cascade of spills
def make_carry(num_packets=256):
    packet_size = 0x5758
    return Carry(
        data=np.empty(packet_size*num_packets, dtype=np.uint8),
        packet_count=np.zeros(1, dtype=int)
    )


def reset_carry(carry_inout: Carry):
    carry_inout.packet_count[0] = 0


@numba.njit(cache=True)
def carryover(carry_inout, packet):
    packet_size = 0x5758
    assert len(packet) == packet_size
    start = carry_inout.packet_count[0] * packet_size
    stop = start + packet_size
    assert stop <= carry_inout.data.shape[0]
    carry_inout.data[start:stop] = packet
    carry_inout.packet_count[0] += 1


# namedtuple is compatible with Numba
# Do some "fake object orientation"
Tile = namedtuple('Tile', ['frame_offset', 'data', 'damage', 'unique_count', 'expected_count'])


def make_tile(tileshape, frame_offset, dtype=np.uint16) -> Tile:
    frame_count = tileshape[0]
    t = Tile(
        frame_offset=np.full(1, frame_offset, dtype=int),
        data=zeros_aligned(tileshape, dtype=dtype),
        damage=np.zeros((frame_count, 32), dtype=bool),
        unique_count=np.zeros(1, dtype=int),
        expected_count=32*frame_count,
    )
    return t


def is_complete(t: Tile):
    return t.unique_count == t.expected_count


@numba.njit(inline='always', cache=True)
def block_idx(x, y):
    '''
    Transform the pixel_x_start, pixel_y_start of a block to
    a block index in 0:32 following the native sequence of K2IS packets

    x in [240, 224, ..., 16, 0]
    y in [0, 930]
    '''
    offset = 0 if y == 0 else 16
    return 15 - x//16 + offset


@numba.njit(inline='always', cache=True)
def block_xy(block_idx: int):
    '''
    Derive pixel_x_start, pixel_y_start from block index
    '''
    y_mult, xx = divmod(block_idx, 16)
    return (240 - xx * 16, y_mult*930)


def make_packet(frame, block_count, x, y):
    PACKET_SIZE = 0x5758
    header = np.zeros(1, dtype=DataBlock.header_dtype)[0]
    header['sync'] = 0xFFFF0055
    header['version'] = 1
    header['flags'] = SHUTTER_ACTIVE_MASK
    header['block_count'] = block_count
    header['width'] = 256
    header['height'] = 1860
    header['frame_id'] = frame
    header['pixel_x_start'] = x
    header['pixel_y_start'] = y
    header['pixel_x_end'] = x + 15
    header['pixel_y_end'] = y + 929
    header['block_size'] = PACKET_SIZE
    payload = bytearray(PACKET_SIZE - header.nbytes)
    # Encode the block count as uint12 payload in the first 1 1/2 bytes
    tag = np.int16(block_count)
    b1 = tag & 0xFF
    b2 = (tag & 0x0F00) >> 8
    payload[0] = b1
    payload[1] = b2

    return header.tobytes() + payload


@numba.njit(cache=True)
def merge_packet(tile_inout: Tile, header: PacketHeader, packet: bytes, offset: int):
    inp_part_data = packet[40:]
    stride_y = tile_inout.data.shape[2]
    index = block_idx(header.pixel_x_start[0], header.pixel_y_start[0])
    frame_idx = header.frame_id[0] - tile_inout.frame_offset[0] - offset
    if tile_inout.damage[frame_idx, index]:
        # print("skipping duplicate package")
        return

    if frame_idx >= tile_inout.data.shape[0]:
        # print(header.frame_id, frame_idx)
        raise RuntimeError("frame_idx is out of bounds")

    block_offset = header.pixel_y_start[0] * stride_y + header.pixel_x_start[0]

    out_z = tile_inout.data[frame_idx].reshape((-1,))

    # decode_uint12_le(inp=inp_part_data, out=out_part)
    # we are inlining the decoding here to write directly
    # to the right position inside the output array:o

    # inp is uint8, so the outer loop needs to jump 24 bytes each time.
    for row in range(len(inp_part_data) // 3 // 8):
        # row is the output row index of a single block,
        # so the beginning of the row in output coordinates:
        out_pos = block_offset + row * stride_y
        in_row_offset = row * 3 * 8

        # processing for a single row:
        # for each j, we process bytes of input into two output numbers
        # -> we consume 8*3 = 24 bytes and generate 8*2=16 numbers
        decode_uint12_le(
            inp=inp_part_data[in_row_offset:in_row_offset+24],
            out=out_z[out_pos:out_pos+16]
        )
    tile_inout.damage[frame_idx, index] = True
    tile_inout.unique_count[0] += 1


def erase_missing(tile_inout: Tile):
    '''
    Overwrite undamaged blocks with zeros
    '''
    if not is_complete(tile_inout):
        for frame_idx in range(tile_inout.damage.shape[0]):
            # 0..32
            for i in range(tile_inout.damage.shape[1]):
                if not tile_inout.damage[frame_idx, i]:
                    x, y = block_xy(i)
                    tile_inout.data[frame_idx, x:x+16, y:y+930] = 0


def recycle(tile_inout: Tile, frame_offset):
    '''
    Reset the tile to a new frame offset and unset unique_count and damage

    The contents are not zeroed out since new data will likely overwrite them.
    Instead, :func:`erase_missing` can selectively overwrite undamaged blocks in
    case of missing packages. This will hopefuilly be rare.
    '''
    tile_inout.frame_offset[0] = frame_offset
    tile_inout.damage[:] = False
    tile_inout.unique_count[0] = 0


# To make sure we have a uniform int return type
# The regular zero or positive return values indicate a buffer index
TARGET_STRAGGLER = -1
TARGET_NEXT_TILE = -2
TARGET_FORERUNNER = -3
TARGET_PARTITION_CARRY = -4
TARGET_DATASET_CARRY = -5


@numba.njit(inline='always', cache=True)
def find_target(bufs, header, frame_offset, end_after_idx, end_dataset_after_idx):
    frame_idx = header.frame_id[0] - frame_offset
    # FIXME or >=?
    assert end_dataset_after_idx >= end_after_idx
    if frame_idx >= end_after_idx:
        # FIXME or >=?
        if frame_idx >= end_dataset_after_idx:
            # print("dataset carry")
            return TARGET_DATASET_CARRY
        else:
            # print("partition carry", frame_offset, header.frame_id[0], frame_idx, end_after_idx)
            return TARGET_PARTITION_CARRY
    for i, buf in enumerate(bufs):
        if frame_idx >= buf.frame_offset[0] and frame_idx < buf.frame_offset[0] + buf.data.shape[0]:
            # print("found target", i, frame_offset, header.frame_id[0], frame_idx, end_after_idx)
            return i
        # before first tile or between tiles
        if frame_idx < buf.frame_offset[0]:
            # print("straggler")
            return TARGET_STRAGGLER
    # Block would be within next consecutive tile
    if frame_idx < bufs[-1].frame_offset + 2*buf.data.shape[0]:
        # print("next tile")
        return TARGET_NEXT_TILE
    else:
        # print("forerunner")
        # Block lies beyond the next tile
        # needs to make a jump
        return TARGET_FORERUNNER


DecoderState = namedtuple(
    'DecoderState',
    [
        'dataset_carry', 'partition_carry', 'tile_carry',
        'first_frame_id',
        'recent_frame_id',
        'end_after_idx',
        'end_dataset_after_idx'
    ]
)


def make_DecoderState(num_packets):
    carry_size = 2*num_packets
    return DecoderState(
        dataset_carry=make_carry(carry_size),
        partition_carry=make_carry(carry_size),
        tile_carry=make_carry(carry_size),
        first_frame_id=np.full(1, -1, dtype=int),
        recent_frame_id=np.full(1, -1, dtype=int),
        end_after_idx=np.full(1, -1, dtype=int),
        end_dataset_after_idx=np.full(1, -1, dtype=int),
    )

@numba.njit(cache=True)
def process_packets(bufs_inout, header_inout, decoder_state: DecoderState, packets, num_packets):
    packet_size = 0x5758
    c_tiles = False
    c_partition = False
    c_dataset = False
    for i in range(num_packets):
        packet = packets[i*packet_size:(i+1)*packet_size]
        decode_header_into(header_inout, packet)
        target = find_target(
            bufs=bufs_inout,
            header=header_inout,
            frame_offset=decoder_state.first_frame_id[0],
            end_after_idx=decoder_state.end_after_idx[0],
            end_dataset_after_idx=decoder_state.end_dataset_after_idx[0]
        )
        packet = packets[i*packet_size:(i+1)*packet_size]
        if target >= 0:  # happy case
            merge_packet(bufs_inout[target], header_inout, packet, offset=decoder_state.first_frame_id[0])
        elif target == TARGET_STRAGGLER:
            # skip stragglers for now
            continue
        elif target == TARGET_NEXT_TILE:
            carryover(decoder_state.tile_carry, packet)
            c_tiles = True
        elif target == TARGET_PARTITION_CARRY:
            carryover(decoder_state.partition_carry, packet)
            c_partition = True
        elif target == TARGET_DATASET_CARRY:
            carryover(decoder_state.dataset_carry, packet)
            c_dataset = True
            pass
        decoder_state.recent_frame_id[0] = max(
            decoder_state.recent_frame_id[0],
            header_inout.frame_id[0]
        )  # most recent frame id
    # print(c_tiles, c_partition, c_dataset)
    return c_tiles, c_partition, c_dataset


class MsgReaderThread(ErrThreadMixin, threading.Thread):
    def __init__(
        self, idx, port, affinity_set, sync_state,
        local_addr='0.0.0.0', iface='enp193s0f0', timeout=5, pdb=False,
        profile=False,
        *args, **kwargs
    ):
        # TODO: separate this class... way too much state!
        self.idx = idx
        self.port = port
        self.affinity_set = affinity_set
        self.iface = iface
        self.local_addr = local_addr
        self.timeout = timeout
        self.sync_state = sync_state
        self.sync_timeout = 1  # TODO: make this a parameter?
        self.replica = EventReplicaClient()
        self.decoder_state = make_DecoderState(128)
        self.sync_barrier = mp.Barrier(8)  # FIXME: hardcoded number of participants
        self.sector_state = SectorState.INIT
        self.packet_counter = 0
        self.pdb = pdb
        self.profile = profile

        super().__init__(*args, **kwargs)

    def run(self):
        profiler = None
        if self.profile:
            from line_profiler import LineProfiler
            profiler = LineProfiler()
            profiler.add_function(self.read_loop)
            profiler.add_function(self.get_tiles)
            profiler.add_function(self.main_loop)
            profiler.add_function(UDFRunner.run_for_partition)
            profiler.add_function(UDFRunner._run_udfs)
            profiler.add_function(UDFRunner._run_tile)
            try:
                profiler.runcall(self._real_run)
            finally:
                import io
                out = io.StringIO()
                profiler.print_stats(stream=out)
                out.seek(0)
                with open("/tmp/profile-%d.txt" % os.getpid(), "w") as f:
                    print("writing profile")
                    f.write(out.read())
        else:
            self._real_run()

    def _real_run(self):
        self.result_socket = ResultSource()
        logger.info(f"thread {threading.get_native_id()}")
        os.sched_setaffinity(0, self.affinity_set)
        warmup()
        try:
            # FIXME: benchmark number of threads
            env = Environment(threads_per_worker=1)
            with mcast_socket(self.port, GROUP, self.local_addr, self.iface) as s,\
                    env.enter(), self.result_socket:
                logger.info(f"listening on {self.local_addr}:{self.port}/{GROUP} on {self.iface}")
                while True:
                    if self.is_stopped():
                        break
                    try:
                        self.main_loop(s)
                    except DisconnectedError:
                        self.reset()
                        self.replica.dispatch(CamDisconnectedEvent())
                        self.replica.do_events()
                        continue
        except Exception as e:
            return self.error(e)
        self.stop()

    def reset(self):
        """
        Completely reset our state. This is called when we run into a timeout,
        or other recoverable errors.
        """
        # FIXME: maybe extract the main loop into another class, which we
        # completely re-create when handling recoverable errors!
        self.sector_state = SectorState.INIT
        self.buffered_tile = None
        # reset synchronization, to make sure we re-sync when data starts to
        # come in again
        self.sync_state.reset()

    def update_first_frame_id(self, new_frame_id):
        self.first_frame_id = new_frame_id

    def read_first_packet(self, s):
        """
        Read and discard first UDP packet we received - this is intended
        to simplify the reading logic in the `read_loop*` methods, in that
        they can assume that we are actively receiving data from the camera.

        We can discard the packet, because it is unlikely to be the first
        packet of the first frame anyways (and we will sync afterwards, discarding
        ~32 more packets)

        After this function has run, either the thread is stopping, or we have received
        data on the UDP socket.

        Parameters
        ----------
        s : socket.socket
            The UDP multicast socket
        """
        buf = bytes_aligned(0x5758)
        while True:
            if self.is_stopped():
                return
            try:
                p = s.recvmsg_into([buf])
                assert p[0] == 0x5758
                self.packet_counter += 1
                self.sector_state = SectorState.CONNECTED
                return
            except socket.timeout:
                continue

    def read_loop(self, s):
        # NOTE: non-IS data is truncated - we only read the first 0x5758 bytes of the message
        buf = bytes_aligned(0x5758)
        s.settimeout(self.timeout)
        while True:
            if self.is_stopped():
                return
            try:
                p = s.recvmsg_into([buf])
                assert p[0] == 0x5758
                self.packet_counter += 1
            except socket.timeout:
                # On timeout, disconnect. Handled in `_real_run`,
                # transitioning to INIT state etc.
                logger.warn("read_loop timeout while connected, resetting")
                raise DisconnectedError()

            yield (buf, p[1])

    def read_loop_bulk(self, s, num_packets=128):
        """
        Read `num_packets` at once. Yields tuples `(buffer, ancdata)`, where `ancdata`
        is UDP-specific anciliary data (and currently unused)
        """
        # NOTE: non-IS data is truncated - we only read the first 0x5758 bytes of the message
        packet_size = 0x5758
        buf = bytes_aligned(packet_size * num_packets)
        buf_arr = np.frombuffer(buf)
        s.settimeout(self.timeout)
        idx = 0

        # first, sync up to `self.first_frame_id`:
        buf_part = buf[idx*packet_size:(idx + 1) * packet_size]
        while True:
            # p is (nbytes, ancdata, msg_flags, address)
            p = s.recvmsg_into([buf_part])
            h = np.frombuffer(buf_part, dtype=DataBlock.header_dtype, count=1, offset=0)
            # TODO: wraparound here? frame_id set to 0 here?
            if int(h['frame_id']) >= self.decoder_state.first_frame_id[0]:
                idx = 1
                break

        while True:
            if self.is_stopped():
                return
            try:
                buf_part = buf[idx*packet_size:(idx + 1) * packet_size]
                p = s.recvmsg_into([buf_part])
                assert p[0] == packet_size
                idx += 1
                self.packet_counter += 1
                if idx == num_packets:
                    yield (buf, p[1])
                    idx = 0
                    buf_arr[:] = 0
            except socket.timeout:
                logger.warn("read_loop timeout while connected, resetting")
                raise DisconnectedError()

    def sync(self, read_iter):
        """
        Initial synchronization of all sectors
        """
        first_frame_id = -1

        # get the frame ids for the first 32 blocks, to discard possibly incomplete
        # frames at the beginning:
        for i in range(32):
            p = next(read_iter)
            h = np.frombuffer(p[0], dtype=DataBlock.header_dtype, count=1, offset=0)
            first_frame_id = max(first_frame_id, int(h['frame_id']))

        if self.sync_state.set_initial_sync() >= 8:
            self.replica.dispatch(CamConnectedEvent())

        # discard packets until the shutter is active:
        while h['flags'] & SHUTTER_ACTIVE_MASK != 1:
            p = next(read_iter)
            h = np.frombuffer(p[0], dtype=DataBlock.header_dtype, count=1, offset=0)
            # not a max() here, because the frame_id can wrap back to 0 when the
            # acquisition starts and the SHUTTER_ACTIVE flag is set!
            # first_frame_id = max(first_frame_id, int(h['frame_id']))
            first_frame_id = int(h['frame_id'])

        # we send the highest frame_id we have seen:
        return self.sync_to_frame_id(first_frame_id)

    def sync_to_frame_id(self, frame_id: int) -> int:
        """
        Synchronize with the specified :code:`frame_id`. The largest :code:`frame_id`
        sent by any sector will be returned.

        Parameters
        ----------
        frame_id : int
            The :code:`frame_id` that should be sent to the synchonization process

        Returns
        -------
        int
            The resulting :code:`frame_id` that won the "election".

        Raises
        ------
        RuntimeError
            If the election fails to resolve in the given timeout
        """
        # send the frame_id to the synchronizer
        # (increment by 1, because we have read at least one block of this frame)
        self.sync_state.set_first_frame_id(frame_id + 1)

        # ... and wait, until all processes have sent their frame_id:
        if not self.sync_state.sync_done.wait(timeout=self.sync_timeout):
            raise RuntimeError("timed out waiting for sync")
        self.packet_counter = 0
        return self.sync_state.get_first_frame_id()

    @property
    def x_offset(self):
        return self.idx * 256

    def get_tiles(
            self, read_iter, start_frame, end_after_idx, end_dataset_after_idx, num_packets=128):
        self.decoder_state.end_dataset_after_idx[0] = end_dataset_after_idx
        self.decoder_state.end_after_idx[0] = end_after_idx
        assert self.decoder_state.tile_carry.packet_count == 0

        assert num_packets % 32 == 0
        num_frames = num_packets // 32
        tileshape = Shape((num_frames, 1860, 256), sig_dims=2)

        bufs = []  # numba.typed.List() unfortunately slow in regular Python
        buf_start = start_frame
        for count in range(3):
            if buf_start < end_after_idx:
                bufs.append(make_tile(tileshape, buf_start))
            buf_start += num_frames

        header = make_PacketHeader()

        x_offset = self.x_offset
        sig_origin = (0, x_offset)

        origin_to_idx = {}
        # ts is a module-level variable
        for idx, slice_ in ts.slices:
            origin_to_idx[slice_.origin] = idx

        logger.info(
            "get_tiles: end_after_idx=%d, num_packets=%d, bufs[0].data.shape=%s",
            end_after_idx, num_packets, bufs[0].data.shape,
        )

        def yield_and_rotate(new_start_idx=None):
            '''
            We finalize the first buffer,
            yield it if it was touched at all, reset it, and set it as the last buffer
            in case we still have work to do
            '''
            if not bufs:
                return
            buf = bufs.pop(0)
            frame_idx = buf.frame_offset[0]
            end_frame = min(frame_idx + buf.data.shape[0], end_after_idx)
            real_tileshape = Shape((end_frame-frame_idx, ) + tileshape[1:], sig_dims=2)
            if new_start_idx is None:
                if bufs:
                    last = bufs[-1]
                    new_start_idx = last.frame_offset[0] + last.data.shape[0]
                else:
                    new_start_idx = end_after_idx
            # tile has been touched, otherwise we just skip it
            if buf.unique_count[0]:
                tile_slice = Slice(
                    origin=(frame_idx,) + sig_origin,
                    shape=real_tileshape,
                )
                scheme_idx = origin_to_idx[sig_origin]
                erase_missing(buf)
                dt = DataTile(
                    # slice out the part that contains data that still belongs to this partition
                    buf.data[:end_frame-frame_idx],
                    tile_slice=tile_slice,
                    scheme_idx=scheme_idx,
                )
                yield dt
            if new_start_idx < end_after_idx:
                recycle(buf, new_start_idx)
                # print("recycle", new_start_idx, end_after_idx)
                bufs.append(buf)

        def wrapup():
            # end_after_idx + 1 makes sure we are not rotating the buffer, but emptying it out
            yield from yield_and_rotate(end_after_idx + 1)
            yield from yield_and_rotate(end_after_idx + 1)
            yield from yield_and_rotate(end_after_idx + 1)

        def deal_with_tile_carry():
            # print("tile carry")
            # We should not come into carry in case we are already finished with the partition
            assert bufs
            # first rotate the buffers to make room for the carried packets
            if self.decoder_state.recent_frame_id[0] < bufs[-1].frame_offset + 2*num_frames:
                yield from yield_and_rotate(None)
            else:
                yield from yield_and_rotate(self.decoder_state.recent_frame_id[0] - num_frames)
            # We should have a buffer at the end that can fit the last carried package
            # and process whatever is left over.
            # This should not re-carry anything for that reason, but rather drop stragglers
            process_packets(
                bufs_inout=numba.typed.List(bufs),
                header_inout=header,
                decoder_state=self.decoder_state,
                packets=self.decoder_state.tile_carry.data,
                num_packets=self.decoder_state.tile_carry.packet_count[0]
            )
            reset_carry(self.decoder_state.tile_carry)
            # Make sure we have a fresh buffer at the end
            # to resume normal operation and not go into endless carry
            yield from rotate_if_necessary()

        def rotate_if_necessary():
            # Happy case: dispatch complete tiles
            while bufs and is_complete(bufs[0]):
                yield from yield_and_rotate(None)
            # Not at the end of the partition
            # and last tile touched
            if len(bufs) == 3 and bufs[-1].unique_count[0]:
                yield from yield_and_rotate(None)

        # First deal with partition carry.
        # We make a copy since we MIGHT carry right back
        # The dataset carry buffer SHOULD have extra space
        tmp_data = self.decoder_state.partition_carry.data.copy()
        # int, by value
        tmp_count = self.decoder_state.partition_carry.packet_count[0]
        # We can carry right into the next partition if necessary
        # Therefore empty out before processing
        reset_carry(self.decoder_state.partition_carry)
        # print("partition carry", tmp_count)
        c_tiles, c_partition, c_dataset = process_packets(
            bufs_inout=numba.typed.List(bufs),
            header_inout=header,
            decoder_state=self.decoder_state,
            packets=tmp_data,
            num_packets=tmp_count
        )

        # Wrap up in case we are already in the next partition or epoch
        if c_partition or c_dataset:
            yield from wrapup()
            return

        if c_tiles:
            yield from deal_with_tile_carry()
        else:
            yield from rotate_if_necessary()

        # Then deal with dataset carry.
        # We make a copy since we MIGHT carry right back
        # The dataset and partition carry buffer SHOULD have extra space enough
        tmp_data = self.decoder_state.dataset_carry.data.copy()
        # int, by value
        tmp_count = self.decoder_state.dataset_carry.packet_count[0]
        reset_carry(self.decoder_state.dataset_carry)
        # print("dataset carry", tmp_count)
        c_tiles, c_partition, c_dataset = process_packets(
            bufs_inout=numba.typed.List(bufs),
            header_inout=header,
            decoder_state=self.decoder_state,
            packets=tmp_data,
            num_packets=tmp_count
        )

        # Wrap up in case we are already in the next partition or dataset
        if c_partition or c_dataset:
            yield from wrapup()
            return

        if c_tiles:
            yield from deal_with_tile_carry()
        else:
            yield from rotate_if_necessary()

        # initialization for loop
        c_tiles = False

        # we might already be finished, see above
        while bufs:
            p = next(read_iter)
            packets, anc = p
            packets = np.array(packets, dtype=np.uint8)
            # print("loop process")
            c_tiles, c_partition, c_dataset = process_packets(
                bufs_inout=numba.typed.List(bufs),
                header_inout=header,
                decoder_state=self.decoder_state,
                packets=packets,
                num_packets=num_packets,
            )
            # wrap up
            if c_partition or c_dataset:
                # print("loop wrapup")
                yield from wrapup()
                return

            if c_tiles:
                yield from deal_with_tile_carry()
            else:
                yield from rotate_if_necessary()
            # FIXME detect wrap-around of frame ID
    # print("end tiles")

    @property
    def nav_shape(self):
        return self.replica.store.state.nav_shape

    def should_process(self):
        if self.replica.store.state is None:
            return False
        if not self.is_connected:
            return False
        return self.replica.store.state.processing is ProcessingState.RUNNING

    def main_loop(self, s):
        self.replica.do_events()

        logger.info("MsgReaderThread: waiting for CONNECTED state")

        self.read_first_packet(s)

        logger.info("MsgReaderThread: CONNECTED, waiting for PRIMED state")

        if self.is_stopped():
            self.sector_state = SectorState.INIT
            return

        # get a first read iterator (non-bulk) for synchronization:
        read_iter_sync = self.read_loop(s)

        while self.sector_state != SectorState.PRIMED:
            # all "busy" loops need to check stopped flag:
            if self.is_stopped():
                self.sector_state = SectorState.INIT
                return

            # we are connected but aren't running a UDF yet, so we need to drain
            # the socket until we are told to run:
            _ = next(read_iter_sync)
            self.replica.do_events()

            # global state transitioned to RUNNING:
            if self.replica.store.state.processing == ProcessingState.RUNNING:
                self.sector_state = SectorState.PRIMED

        logger.info("MsgReaderThread: in PRIMED state, syncing")
        self.decoder_state.first_frame_id[0] = self.sync(read_iter_sync)
        self.decoder_state.recent_frame_id[0] = self.decoder_state.first_frame_id[0]
        logger.info(f"synced to {self.decoder_state.first_frame_id[0]}")

        logger.info("MsgReaderThread: in PROCESSING state")
        self.sector_state = SectorState.PROCESSING

        # NOTE: we could make this dynamic, but that's for another day;
        # as currently the UDF needs to know how many frames there
        # are per partition. we could keep this rather small and
        # have another thread `merge` all the intermediate results,
        # and then sample _that_ intermediate result to the main process
        frames_per_partition = 100

        frame_counter = 0
        epoch = 0
        tiling_scheme = None

        read_iter = self.read_loop_bulk(s)

        while True:
            num_frames = np.prod(self.nav_shape, dtype=np.int64)
            frames_in_partition = min(frames_per_partition, num_frames - frame_counter)

            if self.is_stopped():
                return  # process should shut down

            if self.replica.store.state.processing != ProcessingState.RUNNING:
                # processing should be cancelled, reset our state:
                self.reset()
                return

            if frames_in_partition <= 0:
                # if not in continuous mode, this thread is done with the dataset:
                if not self.replica.store.state.continuous:
                    # FIXME: control flow to "READY" state, instead of basically INIT:
                    self.replica.dispatch(ProcessingDoneEvent(idx=self.idx))
                    logger.info(f"sending EndOfStreamMarker for {self.idx}")
                    self.result_socket.send(EndOfStreamMarker(idx=self.idx))
                    self.sector_state = SectorState.DONE
                    self.replica.do_events()
                    self.sync_state.reset()
                    # wait until other processes are done:
                    while self.replica.store.state.processing != ProcessingState.READY:
                        if self.is_stopped():
                            return
                        self.replica.do_events()
                    return
                frame_counter = 0
                # FIXME: epoch handling here is a bit hacky,
                # this should be fixed to keep `first_frame_id` and the
                # "start of the current epoch" frame id separate.
                epoch += 1
                self.decoder_state.first_frame_id[0] = self.decoder_state.recent_frame_id[0]
                continue

            tiles = self.get_tiles(
                read_iter,
                start_frame=frame_counter,
                end_after_idx=frame_counter + frames_in_partition - 1,
                end_dataset_after_idx=num_frames
            )

            meta = DataSetMeta(
                shape=Shape((num_frames, 1860, 2048), sig_dims=2),
                image_count=num_frames,
                raw_dtype=np.uint16,
            )

            partition_slice = Slice(
                origin=(frame_counter, 0, self.x_offset),
                shape=Shape((frames_in_partition, 1860, 256), sig_dims=2),
            )

            partition = PlaceholderPartition(
                meta=meta,
                partition_slice=partition_slice,
                tiles=tiles,
                start_frame=frame_counter,
                num_frames=frames_in_partition,
            )

            # udfs = [
            #     udf.copy()
            #     for udf in self.replica.store.state.udfs
            # ]
            udfs = self.replica.store.state.udfs
            runner = UDFRunner(udfs)

            logger.debug(f"before run_for_partition {frame_counter}")
            result = runner.run_for_partition(
                partition=partition,
                corrections=None,
                roi=None,
                env=FakeEnvironment(threads_per_worker=1),
                tiling_scheme=tiling_scheme,
                pdb_port=4444 + self.idx if self.pdb else None,
            )
            tiling_scheme = result.tiling_scheme
            logger.info("sending some result for %r (sector %d)", partition_slice, self.idx)
            self.send_result(partition_slice, result, epoch, self.packet_counter)
            frame_counter += frames_in_partition

    def send_result(self, partition_slice, result, epoch, packet_counter):
        self.result_socket.send((partition_slice, result, epoch, packet_counter))


class ResultSink:
    def __init__(self, conn="tcp://*:7205", timeout=0.1):
        self.conn = conn
        self.sink_socket = None
        self.poller = zmq.Poller()
        self.timeout = timeout

    def __enter__(self):
        zctx = zmq.Context.instance()
        self.sink_socket = zctx.socket(zmq.PULL)
        self.sink_socket.bind(self.conn)
        self.poller.register(self.sink_socket, zmq.POLLIN)

    def __exit__(self, *args, **kwargs):
        self.sink_socket.close()
        self.poller.unregister(self.sink_socket)

    def poll(self, timeout=None):
        if timeout is None:
            timeout = self.timeout
        poll_events = dict(self.poller.poll(timeout))
        if self.sink_socket in poll_events:
            res = recv_serialized(self.sink_socket)
            return res


class ResultSource:
    def __init__(self, conn="tcp://127.0.0.1:7205"):
        self.conn = conn
        self.result_socket = None

    def __enter__(self):
        zctx = zmq.Context.instance()
        self.result_socket = zctx.socket(zmq.PUSH)
        self.result_socket.connect(self.conn)

    def __exit__(self, *args, **kwargs):
        self.result_socket.close()

    def send(self, result_obj):
        send_serialized(self.result_socket, result_obj)


def get_settings_for_sector(idx, use_veth):
    settings = {
        'idx': idx,  # zero-based index of sector
        'local_addr': '225.1.1.1',
        'port': 2001 + idx,
        'affinity_set': {8 + idx},
        'iface': 'enp193s0f0' if idx < 4 else 'enp193s0f1',
    }
    if use_veth:
        settings.update({
            'iface': 'veth2',
        })
    return settings


class K2ListenerProcess(mp.Process):
    def __init__(self, idx, sync_state, enable_tracing, pdb, profile, use_veth, *args, **kwargs):
        self.idx = idx
        self.sync_state = sync_state
        self.enable_tracing = enable_tracing
        self.pdb = pdb
        self.profile = profile
        self.use_veth = use_veth
        self._stop_event = mp.Event()
        super().__init__(*args, **kwargs)

    def stop(self):
        self._stop_event.set()

    def is_stopped(self):
        return self._stop_event.is_set()

    def run(self):
        import gc
        gc.disable()
        if self.enable_tracing:
            import pytracing
            self._trace_f = open("/tmp/cam-server-%d.json" % os.getpid(), "wb")
            tp = pytracing.TraceProfiler(output=self._trace_f)
            try:
                with tp.traced():
                    self.main()
            finally:
                self._trace_f.close()
        else:
            self.main()
        gc.enable()

    def main(self):
        warmup()

        settings = get_settings_for_sector(self.idx, use_veth=self.use_veth)
        settings.update({
            'sync_state': self.sync_state,
            'pdb': self.pdb,
            'profile': self.profile,
        })
        t = MsgReaderThread(**settings)

        try:
            t.start()
            logger.info(f"MySubProcess {self.idx} started")
            while not (t.is_stopped() or self.is_stopped()):
                t.maybe_raise()
                time.sleep(0.3)
            logger.info(f"MySubProcess {self.idx} stopped")
            t.maybe_raise()
        finally:
            t.stop()
            t.join()
            self.stop()
        logger.info(f"MySubProcess {self.idx} end of run()")
