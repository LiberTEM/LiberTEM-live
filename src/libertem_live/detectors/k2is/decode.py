import numba
import numpy as np

from libertem.io.dataset.base.decode import byteswap_2_decode, byteswap_4_decode


@numba.njit(inline='always', nogil=True, cache=True)
def decode_uint12_le(inp, out):
    """
    Decode bytes from bytestring ``inp`` as 12 bit into ``out``

    Based partially on https://stackoverflow.com/a/45070947/540644
    """
    # assert np.mod(len(inp), 3) == 0
    # assert len(out) >= len(inp) * 2 / 3

    for i in range(len(inp) // 3):
        fst_uint8 = np.uint16(inp[i * 3])
        mid_uint8 = np.uint16(inp[i * 3 + 1])
        lst_uint8 = np.uint16(inp[i * 3 + 2])

        a = fst_uint8 | (mid_uint8 & 0x0F) << 8
        b = (mid_uint8 & 0xF0) >> 4 | lst_uint8 << 4
        out[i * 2] = a
        out[i * 2 + 1] = b


@numba.njit(inline='always', nogil=True, cache=True)
def decode_bulk_uint12_le(inp, out, num_packets, meta_out=None):
    """
    `inp` should be a byte array or memory view containing `num_packets` of
    raw data received via UDP

    `out` should be of shape (num_packets, -1)
    """

    packet_size = 0x5758

    if meta_out is None:
        meta_out = np.zeros((num_packets, 3), dtype=np.uint32)

    assert num_packets % 32 == 0, "num_packets must be a multiple of 32"
    num_frames = num_packets // 32

    out_3d = out.reshape((num_frames, 1860, 256))

    # the offset between two rows in the output (in indices, not bytes)
    stride_y = out_3d.shape[2]

    first_frame_id = np.zeros((1,), dtype=np.uint32)
    byteswap_4_decode(inp=inp[24:28], out=first_frame_id)
    first_frame_id = first_frame_id[0]

    for p in range(num_packets):
        # assert inp_part[:4] == b'\xff\xff\x00\x55'
        inp_part = inp[p * packet_size:(p + 1) * packet_size]
        inp_part_data = inp_part[40:]

        # assert inp_part[0] == 0xFF
        # assert inp_part[1] == 0xFF
        # assert inp_part[2] == 0x00
        # assert inp_part[3] == 0x55

        byteswap_4_decode(inp=inp_part[24:28], out=meta_out[p, 0:1])  # frame_id
        byteswap_2_decode(inp=inp_part[30:32], out=meta_out[p, 1:2])  # pixel_y_start
        byteswap_2_decode(inp=inp_part[28:30], out=meta_out[p, 2:3])  # pixel_x_start

        frame_id = meta_out[p, 0]
        pixel_y_start = meta_out[p, 1]
        pixel_x_start = meta_out[p, 2]

        # starting offset of the current block:
        block_offset = pixel_y_start * stride_y + pixel_x_start

        frame_idx = frame_id - first_frame_id

        if frame_idx >= out_3d.shape[0]:
            print(set(meta_out[..., 0]))
            raise RuntimeError("frame_idx is out of bounds")

        out_z = out_3d[frame_idx].reshape((-1,))

        # decode_uint12_le(inp=inp_part_data, out=out_part)
        # we are inlining the decoding here to write directly
        # to the right position inside the output array:o

        # inp is uint8, so the outer loop needs to jump 24 bytes each time.
        for row in range(len(inp_part_data) // 3 // 8):
            # row is the output row index of a single block,
            # so the beginning of the row in output coordinates:
            out_pos = block_offset + row * stride_y
            in_row_offset = row * 3 * 8

            # loop for a single row:
            # for each j, we process bytes of input into two output numbers
            # -> we consume 8*3 = 24 bytes and generate 8*2=16 numbers
            # for col in range(8):
            #     triplet_offset = in_row_offset + col * 3
            #     fst_uint8 = np.uint16(inp_part_data[triplet_offset])
            #     mid_uint8 = np.uint16(inp_part_data[triplet_offset + 1])
            #     lst_uint8 = np.uint16(inp_part_data[triplet_offset + 2])

            #     a = fst_uint8 | (mid_uint8 & 0x0F) << 8
            #     b = (mid_uint8 & 0xF0) >> 4 | lst_uint8 << 4

            #     out_z[2 * col + out_pos] = a
            #     out_z[2 * col + out_pos + 1] = b

            # processing for a single row:
            # for each j, we process bytes of input into two output numbers
            # -> we consume 8*3 = 24 bytes and generate 8*2=16 numbers
            decode_uint12_le(
                inp=inp_part_data[in_row_offset:in_row_offset+24],
                out=out_z[out_pos:out_pos+16]
            )

    return meta_out
