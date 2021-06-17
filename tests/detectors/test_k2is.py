import numpy as np

from libertem_live.detectors.k2is.proto import MsgReaderThread, block_idx, block_xy, make_carry
from libertem.io.dataset.k2is import DataBlock, SHUTTER_ACTIVE_MASK


PACKET_SIZE = 0x5758

START = 2


class MockThread:
    buffered_tile = None
    first_frame_id = START
    timeout = 1
    # first sector
    x_offset = 0

    def __init__(self):
        self.packet_counter = 0
        self.recent_frame_id = 0
        self.partition_carry = make_carry(num_packets=256)
        self.dataset_carry = make_carry(num_packets=256)

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

        header = np.zeros(1, dtype=DataBlock.header_dtype)[0]
        header['sync'] = 0xFFFF0055
        header['version'] = 1
        header['flags'] = SHUTTER_ACTIVE_MASK
        header['block_count'] = self.block_count
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
        tag = np.int16(self.block_count)
        b1 = tag & 0xFF
        b2 = (tag & 0x0F00) >> 8
        payload[0] = b1
        payload[1] = b2

        nbytes = PACKET_SIZE
        ancdata = None
        msg_flags = None
        address = None
        buffer[:] = header.tobytes() + payload
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
    frame_id = thread.first_frame_id
    end_dataset_after_idx = 24
    for repeat in range(3):
        start = frame_id - thread.first_frame_id
        print("start", start)
        tiles = MsgReaderThread.get_tiles(
            thread,
            packets,
            start_frame=start,
            end_after_idx=start+8,
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


def test_sequence():
    socket = MockSocket()
    for i in range(32):
        x, y = block_xy(i)
        assert socket.offsets[i] == (x, y)
        assert block_idx(x, y) == i
