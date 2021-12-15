import numba
import numpy as np

# These encoders takes 2D input/output data - this means we can use
# strides to do slicing and reversing. 2D input data means one output
# row (of bytes) corresponds to one input row (of pixels).


@numba.njit(cache=True)
def encode_u1(inp, out):
    for y in range(out.shape[0]):
        out[y] = inp[y]


@numba.jit(cache=True)
def encode_u2(inp, out):
    for y in range(out.shape[0]):
        row_out = out[y]
        row_in = inp[y]
        for i in range(row_in.shape[0]):
            in_value = row_in[i]
            row_out[i * 2] = (0xFF00 & in_value) >> 8
            row_out[i * 2 + 1] = 0xFF & in_value


@numba.njit(cache=True)
def encode_r1(inp, out):
    for y in range(out.shape[0]):
        row_out = out[y]
        row_in = inp[y]
        for stripe in range(row_out.shape[0] // 8):
            for byte in range(8):
                out_byte = 0
                for bitpos in range(8):
                    value = row_in[64 * stripe + 8 * byte + bitpos] & 1
                    out_byte |= (value << bitpos)
                row_out[(stripe + 1) * 8 - (byte + 1)] = out_byte


@numba.njit(cache=True)
def encode_r6(inp, out):
    for y in range(out.shape[0]):
        row_out = out[y]
        row_in = inp[y]
        for i in range(row_out.shape[0]):
            col = i % 8
            pos = i // 8
            in_pos = (pos + 1) * 8 - col - 1
            row_out[i] = row_in[in_pos]


@numba.njit(cache=True)
def encode_r12(inp, out):
    for y in range(out.shape[0]):
        row_out = out[y]
        row_in = inp[y]
        for i in range(row_in.shape[0]):
            col = i % 4
            pos = i // 4
            in_pos = (pos + 1) * 4 - col - 1
            in_value = row_in[in_pos]
            row_out[i * 2] = (0xFF00 & in_value) >> 8
            row_out[i * 2 + 1] = 0xFF & in_value


def encode_quad(encode, data, bits_per_pixel, with_headers=False):
    """
    Parameters
    ==========

    with_headers : bool
        Will insert some random data between the frames, not real headers.
    """
    shape = data.shape
    num_frames = shape[0]
    # typically the header size for quad data, but doesn't really matter,
    # as we don't generate a "true" header, but just random padding at
    # the beginning of each frame.
    header_bytes = 768
    assert len(shape) == 3  # decoding multiple frames at once
    enc_bytes_per_frame = shape[1] * shape[2] // 8 * bits_per_pixel
    x_shape_px = 256 // 8 * bits_per_pixel

    encoded = np.zeros(
        data.size // 8 * bits_per_pixel + shape[0] * header_bytes,
        dtype=np.uint8
    )
    encoded = encoded.reshape((-1, enc_bytes_per_frame + header_bytes))

    # encoders only do one frame per call:
    for i in range(shape[0]):
        encoded[i, :header_bytes] = np.random.randint(0, 0x100, header_bytes)

        # reshape destination buffer to allow convenient row-based assignment:
        # dest = [4 | 3 | 2 | 1]
        dest = encoded[i, header_bytes:].reshape((256, -1))
        assert dest.shape == (256, 4 * x_shape_px)

        src = data[i]
        src_half = src.shape[0] // 2, src.shape[1] // 2

        q1 = src[:src_half[0], :src_half[1]]
        encode(inp=q1, out=dest[:, 3 * x_shape_px:])

        q2 = src[:src_half[0], src_half[1]:]
        encode(inp=q2, out=dest[:, 2 * x_shape_px:3 * x_shape_px])

        # q3/q4 flipped in y direction
        q3 = src[src_half[0]:, :src_half[1]][::-1, ::-1]
        encode(inp=q3, out=dest[:, 1 * x_shape_px:2 * x_shape_px])
        q4 = src[src_half[0]:, src_half[1]:][::-1, ::-1]
        encode(inp=q4, out=dest[:, 0 * x_shape_px:1 * x_shape_px])

    if with_headers:
        return encoded
    else:
        encoded_data = encoded.reshape((num_frames, -1,))[:, header_bytes:].reshape(
            (num_frames, data.shape[1], -1)
        )
        return encoded_data
