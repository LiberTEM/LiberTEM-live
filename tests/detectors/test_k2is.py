import numpy as np

from libertem_live.detectors.k2is.proto import (
    MsgReaderThread, block_idx, block_xy,
    make_carry, make_DecoderState, make_packet, warmup
)
from libertem.io.dataset.k2is import DataBlock, SHUTTER_ACTIVE_MASK


PACKET_SIZE = 0x5758

START = 2


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
    def __init__(self):
        self.block_count = 0
        self.offsets = []
        # [240, 224, 208, ..., 16, 0]
        for x in range(240, -16, -16):
            self.offsets.append((x, 0))
        for x in range(240, -16, -16):
            self.offsets.append((x, 930))

    def recvmsg_into(self, buffers):
        frame, packet = divmod(self.block_count, 32)
        # The calling code always receives into a single buffer for a single packet
        assert len(buffers) == 1
        buffer = buffers[0]
        assert len(buffer) == PACKET_SIZE

        x, y = self.offsets[packet]

        data = make_packet(frame, self.block_count, x, y)
        nbytes = PACKET_SIZE
        ancdata = None
        msg_flags = None
        address = None
        buffer[:] = data
        # Know where we are in the virtual dataset
        self.block_count += 1

        return (nbytes, ancdata, msg_flags, address)

    def settimeout(self, timeout):
        return


def test_gettiles():
    thread = MockThread()
    socket = MockSocket()
    packets = MsgReaderThread.read_loop_bulk(thread, socket, num_packets=128)
    tile_id = 0
    frame_id = thread.decoder_state.first_frame_id[0]
    per_partition = 17
    num_partitions = 3
    end_dataset_after_idx = per_partition * num_partitions
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

        for t in tiles:
            frames_in_tile = t.shape[0]
            print("frames_in_tile:", frames_in_tile)
            for frame in range(frames_in_tile):
                # Check that the block_count that was baked into the payload by MockSocket
                # ended up where it was supposed to
                for y in (0, 930):
                    for x in range(0, 256, 16):
                        tag = t[frame, y, x]
                        offset = 0 if y == 0 else 16
                        # unwind the sequence from MockSocket
                        target = 15 - x//16 + offset + 32*frame_id
                        # print(tile_id, frame_id, y, x, tag, target)
                        assert tag == target
                frame_id += 1
            tile_id += 1


def test_warmup():
    warmup()


def test_sequence():
    socket = MockSocket()
    for i in range(32):
        x, y = block_xy(i)
        assert socket.offsets[i] == (x, y)
        assert block_idx(x, y) == i
