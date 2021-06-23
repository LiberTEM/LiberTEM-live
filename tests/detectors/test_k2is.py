import numpy as np
from numpy.core.numeric import full
import pytest

from libertem_live.detectors.k2is.proto import (
    MsgReaderThread, block_idx, block_xy,
    make_DecoderState, make_packet, warmup
)
from libertem.io.dataset.k2is import DataBlock, SHUTTER_ACTIVE_MASK


PACKET_SIZE = 0x5758
MAX_PER_TILE = 4

START = 2


def blockstream(start_frame=0):
    frame = start_frame
    block = start_frame*32
    while True:
        for i in range(32):
            yield (frame, block, i)
            block += 1
        frame += 1


def validate_frame_slice(data, frame_id, damage=None):
    '''
    Check that the block_count that was baked into the payload by make_packet()
    ended up where it was supposed to.
    '''
    if damage is None:
        damage = np.ones(32, dtype=bool)
    for i in range(32):
        x, y = block_xy(i)
        tag = data[y, x]
        target = 32*frame_id + i
        if damage[i]:
            # 12 bit maximum
            assert tag == target % 2**12
        else:
            # Confirm that the data was erased
            assert np.all(data[y:y+930, x:x+16] == 0)


class MockThread:
    timeout = 1
    # first sector
    x_offset = 0

    def __init__(self):
        self.decoder_state = make_DecoderState(128)
        self.packet_counter = 0
        self.decoder_state.recent_frame_id[0] = -1
        self.decoder_state.first_frame_id[0] = 2

    def is_stopped(self):
        return False


class MockSocket:
    def __init__(self, packet_generator):
        self._generator = packet_generator

    def recvmsg_into(self, buffers):
        # The calling code always receives into a single buffer for a single packet
        assert len(buffers) == 1
        buffer = buffers[0]
        assert len(buffer) == PACKET_SIZE

        (frame, block, packet) = next(self._generator)

        x, y = block_xy(packet)

        data = make_packet(frame, block, x, y)
        nbytes = PACKET_SIZE
        ancdata = None
        msg_flags = None
        address = None
        buffer[:] = data
        # Know where we are in the virtual dataset

        return (nbytes, ancdata, msg_flags, address)

    def settimeout(self, timeout):
        return


@pytest.mark.parametrize(
    ('per_partition', 'num_partitions', 'carry_partition', 'carry_dataset'), [
        (1, 3, [2*32, 1*32, 0*32], [1*32, 1*32, 1*32]),
        (4, 3, [0*32, 0*32, 0*32], [0*32, 0*32, 0*32]),
        (5, 3, [3*32, 2*32, 0*32], [0*32, 0*32, 1*32]),
        (40, 3, [0*32, 0*32, 0*32], [0*32, 0*32, 0*32]),
        (41, 3, [3*32, 2*32, 0*32], [0*32, 0*32, 1*32]),
        (5, 1, [0*32], [3*32]),
    ]
)
def test_gettiles(per_partition, num_partitions, carry_partition, carry_dataset):
    thread = MockThread()
    socket = MockSocket(blockstream())
    packets = MsgReaderThread.read_loop_bulk(thread, socket, num_packets=128)
    frame_id = thread.decoder_state.first_frame_id[0]
    end_dataset_after_idx = per_partition * num_partitions
    full_tiles, tail = divmod(per_partition, MAX_PER_TILE)
    # Extra tile for the tail
    n_tiles = full_tiles + (1 if tail else 0)
    # Full tiles plus a tile
    frames_in_tile = (MAX_PER_TILE, ) * full_tiles + ((tail, ) if tail else tuple())
    for repeat in range(num_partitions):
        start = frame_id - thread.decoder_state.first_frame_id[0]
        print("start", start)
        tiles = MsgReaderThread.get_tiles(
            thread,
            packets,
            start_frame=start,
            end_after_idx=start+per_partition,
            end_dataset_after_idx=end_dataset_after_idx
        )

        # Since the buffer of a tile is reused,
        # one should consume and check the tiles as they come in
        # instead of making a list of them

        for i, t in enumerate(tiles):
            assert t.shape[0] == frames_in_tile[i]
            f_i_t = t.shape[0]
            print("frames in tile:", f_i_t)
            for frame in range(f_i_t):
                print("frame ID, frame: ", frame_id, frame)
                validate_frame_slice(t[frame], frame_id)
                frame_id += 1
        # Check the state and number of tiles after
        # all tiles in the partition tile generator are exhausted
        assert thread.decoder_state.partition_carry.packet_count == carry_partition[repeat]
        assert thread.decoder_state.dataset_carry.packet_count == carry_dataset[repeat]
        assert i == n_tiles - 1

    # Do two other datasets with one partition of 1 frames right after the first
    # to confirm that dataset carry works, including cascading carry

    for i in range(2):
        start = end_dataset_after_idx + i
        print("start following dataset", start)
        tiles = MsgReaderThread.get_tiles(
            thread,
            packets,
            start_frame=start,
            end_after_idx=start + 1,
            end_dataset_after_idx=start + 1
        )

        for i, t in enumerate(tiles):
            assert t.shape[0] == 1
            f_i_t = t.shape[0]
            print("frames in tile:", f_i_t)
            for frame in range(f_i_t):
                print("frame ID, frame: ", frame_id, frame)
                validate_frame_slice(t[frame], frame_id)
                frame_id += 1


def test_warmup():
    warmup()


def test_sequence():
    offsets = []
    # [240, 224, 208, ..., 16, 0]
    for x in range(240, -16, -16):
        offsets.append((x, 0))
    for x in range(240, -16, -16):
        offsets.append((x, 930))
    for i in range(32):
        x, y = block_xy(i)
        assert offsets[i] == (x, y)
        assert block_idx(x, y) == i
