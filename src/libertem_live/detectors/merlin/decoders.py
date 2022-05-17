import numba


@numba.jit(cache=True)
def decode_u2(inp, out):
    for y in range(out.shape[0]):
        row_out = out[y]
        row_in = inp[y]
        for i in range(row_in.shape[0] // 2):
            o0 = row_in[i * 2 + 0] << 8
            o1 = row_in[i * 2 + 1] << 0
            row_out[i] = o0 | o1


@numba.njit(nogil=True, cache=True, parallel=False)
def decode_multi_u2(input_bytes, out, header_size_bytes, num_frames):
    """
    Decode multiple >u2 frames
    """
    for frame in numba.prange(num_frames):
        in_for_frame = input_bytes[frame]
        out_for_frame = out[frame]
        decode_u2(in_for_frame, out_for_frame)


@numba.njit(nogil=True, cache=True, parallel=False)
def decode_multi_u1(input_bytes, out, header_size_bytes, num_frames):
    """
    Decode multiple u1 frames
    """
    for frame in numba.prange(num_frames):
        in_for_frame = input_bytes[frame]
        out_for_frame = out[frame]
        out_for_frame[:] = in_for_frame


@numba.jit(cache=True)
def decode_r1(inp, out):
    """
    Decode a number of 1bit raw encoded rows of input data.
    `inp` and `out` should be of shape `(num_rows, -1)`.
    This function works with non-contiguous input and output arrays.
    """
    for y in range(out.shape[0]):
        row_out = out[y]
        row_in = inp[y]

        for stripe in range(inp.shape[1] // 8):
            for byte in range(8):
                inp_byte = row_in[(stripe + 1) * 8 - (byte + 1)]
                for bitpos in range(8):
                    outpos = 64 * stripe + 8 * byte + bitpos
                    row_out[outpos] = (inp_byte >> bitpos) & 1


@numba.jit(nogil=True, cache=True, parallel=False)
def decode_multi_r1(input_bytes, out, header_size_bytes, num_frames):
    """
    input_bytes: shape (num_frames, num_rows, -1), with headers already cut off.
    """
    for frame in numba.prange(num_frames):
        in_for_frame = input_bytes[frame]
        out_for_frame = out[frame]
        decode_r1(in_for_frame, out_for_frame)


@numba.njit(nogil=True, parallel=False, cache=True)
def decode_quad_r1(input_bytes, out, header_size_bytes, num_frames):
    bpp = 1  # bits per pixel
    x_size_half_px = 256  # assumption - might need to be a parameter later
    x_size_half = x_size_half_px * bpp // 8
    y_size_half_px = 256

    out = out.reshape((num_frames, 512, 512))

    assert input_bytes.shape == (num_frames, 256, 1024 // 8 * bpp)

    for i in numba.prange(num_frames):
        out_for_frame = out[i]
        in_for_frame = input_bytes[i]

        # read the strided data, per input quadrant, one row at a time:
        # quadrant 1:
        out_for_q = out_for_frame[:y_size_half_px, :x_size_half_px]
        in_for_q = in_for_frame[:, 3 * x_size_half:]
        decode_r1(in_for_q, out_for_q)

        # quadrant 2:
        out_for_q = out_for_frame[:y_size_half_px, x_size_half_px:]
        in_for_q = in_for_frame[:, 2 * x_size_half:3 * x_size_half]
        decode_r1(in_for_q, out_for_q)

        # quadrant 3:
        out_for_q = out_for_frame[y_size_half_px:, :x_size_half_px][::-1, ::-1]
        in_for_q = in_for_frame[:, 1 * x_size_half:2 * x_size_half]
        decode_r1(in_for_q, out_for_q)

        # quadrant 4:
        out_for_q = out_for_frame[y_size_half_px:, x_size_half_px:][::-1, ::-1]
        in_for_q = in_for_frame[:, 0 * x_size_half:1 * x_size_half]
        decode_r1(in_for_q, out_for_q)


@numba.njit(cache=True)
def decode_r6(inp, out):
    """
    RAW 6bit format: the pixels need to be re-ordered in groups of 8. `inp`
    should have dtype uint8.
    """
    for y in range(out.shape[0]):
        row_out = out[y]
        row_in = inp[y]

        for i in range(out.shape[1]):
            col = i % 8
            pos = i // 8
            out_pos = (pos + 1) * 8 - col - 1
            row_out[out_pos] = row_in[i]


@numba.jit(nogil=True, cache=True, parallel=False)
def decode_multi_r6(input_bytes, out, header_size_bytes, num_frames):
    """
    input_bytes: shape (num_frames, num_rows, -1), with headers already cut off.
    """
    for frame in numba.prange(num_frames):
        in_for_frame = input_bytes[frame]
        out_for_frame = out[frame]
        decode_r6(in_for_frame, out_for_frame)


@numba.njit(nogil=True, cache=True, parallel=False)
def decode_quad_r6(input_bytes, out, header_size_bytes, num_frames):
    bpp = 8  # bits per pixel
    x_size_half_px = 256  # assumption - might need to be a parameter later
    x_size_half = x_size_half_px * bpp // 8
    y_size_half_px = 256

    out = out.reshape((num_frames, 512, 512))

    assert input_bytes.shape == (num_frames, 256, 1024 // 8 * bpp)

    for i in numba.prange(num_frames):
        out_for_frame = out[i]
        in_for_frame = input_bytes[i]

        # read the strided data, per input quadrant, one row at a time:
        # quadrant 1:
        out_for_q = out_for_frame[:y_size_half_px, :x_size_half_px]
        in_for_q = in_for_frame[:, 3 * x_size_half:]
        decode_r6(in_for_q, out_for_q)

        # quadrant 2:
        out_for_q = out_for_frame[:y_size_half_px, x_size_half_px:]
        in_for_q = in_for_frame[:, 2 * x_size_half:3 * x_size_half]
        decode_r6(in_for_q, out_for_q)

        # quadrant 3:
        out_for_q = out_for_frame[y_size_half_px:, :x_size_half_px][::-1, ::-1]
        in_for_q = in_for_frame[:, 1 * x_size_half:2 * x_size_half]
        decode_r6(in_for_q, out_for_q)

        # quadrant 4:
        out_for_q = out_for_frame[y_size_half_px:, x_size_half_px:][::-1, ::-1]
        in_for_q = in_for_frame[:, 0 * x_size_half:1 * x_size_half]
        decode_r6(in_for_q, out_for_q)


@numba.njit(cache=True)
def decode_r12(inp, out):
    """
    RAW 12bit format: the pixels need to be re-ordered in groups of 4. `inp`
    should be an uint8 view on big endian 12bit data (">u2")

    `inp` and `out` should be of shape `(num_rows, -1)`.
    This function works with non-contiguous input and output arrays.
    """
    for y in range(out.shape[0]):
        row_out = out[y]
        row_in = inp[y]

        for i in range(out.shape[1]):
            col = i % 4
            pos = i // 4
            out_pos = (pos + 1) * 4 - col - 1
            row_out[out_pos] = (row_in[i * 2] << 8) + (row_in[i * 2 + 1] << 0)


@numba.jit(nogil=True, cache=True, parallel=False)
def decode_multi_r12(input_bytes, out, header_size_bytes, num_frames):
    """
    input_bytes: shape (num_frames, num_rows, -1), with headers already cut off.
    """
    for frame in numba.prange(num_frames):
        in_for_frame = input_bytes[frame]
        out_for_frame = out[frame]
        decode_r12(in_for_frame, out_for_frame)


@numba.njit(nogil=True, cache=True, parallel=False)
def decode_quad_r12(input_bytes, out, header_size_bytes, num_frames):
    bpp = 16  # bits per pixel
    x_size_half_px = 256  # assumption - might need to be a parameter later
    x_size_half = x_size_half_px * bpp // 8
    y_size_half_px = 256

    out = out.reshape((num_frames, 512, 512))

    assert input_bytes.shape == (num_frames, 256, 1024 // 8 * bpp)

    for i in numba.prange(num_frames):
        out_for_frame = out[i]
        in_for_frame = input_bytes[i]

        # read the strided data, per input quadrant, one row at a time:
        # quadrant 1:
        out_for_q = out_for_frame[:y_size_half_px, :x_size_half_px]
        in_for_q = in_for_frame[:, 3 * x_size_half:]
        decode_r12(in_for_q, out_for_q)

        # quadrant 2:
        out_for_q = out_for_frame[:y_size_half_px, x_size_half_px:]
        in_for_q = in_for_frame[:, 2 * x_size_half:3 * x_size_half]
        decode_r12(in_for_q, out_for_q)

        # quadrant 3:
        out_for_q = out_for_frame[y_size_half_px:, :x_size_half_px][::-1, ::-1]
        in_for_q = in_for_frame[:, 1 * x_size_half:2 * x_size_half]
        decode_r12(in_for_q, out_for_q)

        # quadrant 4:
        out_for_q = out_for_frame[y_size_half_px:, x_size_half_px:][::-1, ::-1]
        in_for_q = in_for_frame[:, 0 * x_size_half:1 * x_size_half]
        decode_r12(in_for_q, out_for_q)
