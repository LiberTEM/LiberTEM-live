import os

os.environ['NUMBA_FULL_TRACEBACKS'] = '1'

import random

import numpy as np
import pytest

from libertem_live.detectors.k2is.proto import (
    MsgReaderThread, block_idx, block_xy,
    make_DecoderState, make_packet, warmup
)


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


def wrap(packages, index, step):
    while True:
        (frame, block, i) = next(packages)
        if frame < index:
            yield (frame, block, i)
        else:
            yield(frame - step, block, i)


def scramble(packages, window=16):
    when = 0
    buffer = {}
    for i in range(window//2):
        try:
            buffer[when] = next(packages)
            when += 1
        except StopIteration:
            break
    while buffer:
        keys = list(buffer.keys())
        min_key = min(keys)
        # make sure we only scramble by window size:
        # yield items that are too old
        if when - min_key > window:
            key = min_key
        else:
            key = random.choice(keys)
        yield buffer.pop(key)
        try:
            buffer[when] = next(packages)
            when += 1
        except StopIteration:
            continue


def random_dup(packages, p=0.1):
    for pack in packages:
        yield pack
        if random.random() <= p:
            yield pack


def random_drop(packages, bin, p=0.1):
    for pack in packages:
        if random.random() > p:
            yield pack
        else:
            # print("drop", pack)
            bin.append(pack)


def block_drop(packages, offset, count, bin):
    counter = 0
    for pack in packages:
        if counter < offset or counter >= (offset + count):
            yield pack
        else:
            bin.append(pack)
        counter += 1


def validate_frame_slice(data, frame_id, bin=[], stragglers=[]):
    '''
    Check that the block_count that was baked into the payload by make_packet()
    ended up where it was supposed to. If it is in the bin, make sure it is
    zeroed out correctly
    '''
    def make_key(frame_id, i):
        return (frame_id, frame_id*32 + i, i)

    straggle_keys = []
    for h in stragglers:
        i = block_idx(h.pixel_x_start[0], h.pixel_y_start[0])
        straggle_keys.append(make_key(h.raw_frame_id[0], i))
    for i in range(32):
        x, y = block_xy(i)
        tag = data[y, x]
        target = 32*frame_id + i
        key = make_key(frame_id, i)
        if key in bin or key in straggle_keys:
            assert np.all(data[y:y+930, x:x+16] == 0)
        else:
            try:
                # 12 bit maximum
                assert tag == target % 2**12
            except AssertionError:
                print(key, bin, straggle_keys)
                raise


class MockThread:
    timeout = 1
    # first sector
    x_offset = 0

    def __init__(self):
        self.decoder_state = make_DecoderState(128)
        self.packet_counter = 0
        self.decoder_state.recent_frame_id[0] = -1
        self.decoder_state.first_frame_id[0] = 2
        self.record_stragglers = True
        self.stragglers = []

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
@pytest.mark.parametrize(
    ('do_scramble', 'do_drop', 'do_dup', 'do_wrap'), [
        (0, True, True, True),
        (MAX_PER_TILE*32//8, True, True, True),
        # Too much scrambling can upset wrap detection
        (MAX_PER_TILE*32//2, True, True, False),
        # This will cause many stragglers
        (MAX_PER_TILE*32*2, False, False, False),
    ]
)
def test_gettiles(
        per_partition, num_partitions, carry_partition, carry_dataset,
        do_scramble, do_drop, do_dup, do_wrap):
    to_validate = []
    step_index = 97
    step_amount = 89
    if do_wrap and per_partition*num_partitions <= step_index:
        pytest.skip("Skipping wrapping since too few frames")
    if do_wrap and do_scramble > MAX_PER_TILE*32//2:
        pytest.skip("Skipping wrapping since too much scrambling to detect jump reliably")
    gen = blockstream()
    bin = []
    if do_scramble:
        gen = scramble(gen, window=do_scramble)
    if do_drop:
        gen = random_drop(gen, bin)
    if do_wrap:
        gen = wrap(gen, step_index, step_amount)
    if do_dup:
        gen = random_dup(gen)

    thread = MockThread()
    socket = MockSocket(gen)
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
        # one has to take a copy for later validation
        # Th evalidation has to happen at the end since stragglers
        # and dropped packets can only be distinguished there.

        for i, t in enumerate(tiles):
            assert t.shape[0] == frames_in_tile[i]
            f_i_t = t.shape[0]
            print("frames in tile:", f_i_t)
            for frame in range(f_i_t):
                print("frame ID, frame: ", frame_id, frame)
                to_validate.append(dict(data=t[frame].copy(), frame_id=frame_id))
                frame_id += 1
        # Test the state of partition and dataset carry after
        # all tiles in the partition tile generator are exhausted.
        # This is skipped if we do random scramble, drop or dup since that disturbs the
        # relation between packet count and carry state, so we can't predict
        # it anymore
        if not do_scramble and not do_drop and not do_dup and not do_wrap:
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
                to_validate.append(dict(data=t[frame].copy(), frame_id=frame_id))
                frame_id += 1

    # Stragglers can only be caused by scrambling
    if not do_scramble:
        assert not thread.stragglers

    # Consume more frames to drain all stragglers from the source
    start = end_dataset_after_idx + 2
    n_frames = max(32, do_scramble//32*2)
    tiles = MsgReaderThread.get_tiles(
        thread,
        packets,
        start_frame=start,
        end_after_idx=start + n_frames,
        end_dataset_after_idx=start + n_frames,
    )
    for i, t in enumerate(tiles):
        pass

    for item in to_validate:
        validate_frame_slice(bin=bin, stragglers=thread.stragglers, **item)


def test_gettiles_blockdrop():
    bin = []
    gen = block_drop(blockstream(), 69, 153, bin)
    # gen = blockstream()
    thread = MockThread()
    socket = MockSocket(gen)
    packets = MsgReaderThread.read_loop_bulk(thread, socket, num_packets=128)
    per_partition = 63
    num_partitions = 3
    end_dataset_after_idx = per_partition * num_partitions
    for repeat in range(num_partitions):
        start = repeat * per_partition
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

        for t in tiles:
            f_i_t = t.shape[0]
            print("frames in tile:", f_i_t)
            (frame_id, _, _) = t.tile_slice.origin
            frame_id += thread.decoder_state.first_frame_id[0]
            for frame in range(f_i_t):
                print("frame ID, frame: ", frame_id, frame)
                validate_frame_slice(t[frame], frame_id, bin=bin)
                frame_id += 1
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

        for t in tiles:
            assert t.shape[0] == 1
            f_i_t = t.shape[0]
            print("frames in tile:", f_i_t)
            (frame_id, _, _) = t.tile_slice.origin
            frame_id += thread.decoder_state.first_frame_id[0]
            for frame in range(f_i_t):
                print("frame ID, frame: ", frame_id, frame)
                validate_frame_slice(t[frame], frame_id, bin=bin)
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
