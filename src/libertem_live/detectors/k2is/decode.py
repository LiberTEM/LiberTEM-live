import numba
import numpy as np


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
