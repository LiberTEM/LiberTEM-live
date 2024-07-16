import numpy as np
from numpy.testing import assert_allclose
import pytest

from libertem_live.detectors.merlin.decoders import (
    decode_r1,
    decode_r6,
    decode_r12,
    decode_u2,

    decode_multi_u1,
    decode_multi_u2,
    decode_multi_r1,
    decode_multi_r6,
    decode_multi_r12,
    decode_quad_r1,
    decode_quad_r6,
    decode_quad_r12,
)
from libertem_live.detectors.merlin.utils import (
    encode_u1,
    encode_u2,
    encode_r1,
    encode_r6,
    encode_r12,
    encode_quad,
)
from libertem_qd_mpx import QdDecoder


def encode_roundtrip(encode, decode, bits_per_pixel, shape=(512, 512)):
    max_value = (1 << bits_per_pixel) - 1
    data = np.random.randint(0, max_value + 1, shape)
    encoded = np.zeros(data.size // 8 * bits_per_pixel, dtype=np.uint8)
    encoded = encoded.reshape((shape[0], -1))
    encode(inp=data, out=encoded)
    decoded = np.zeros_like(data)
    decode(inp=encoded, out=decoded)
    return data, decoded


@pytest.mark.with_numba
@pytest.mark.parametrize(
    'encode,decode,bits_per_pixel', [
        (encode_u2, decode_u2, 16),
        (encode_r1, decode_r1, 1),
        (encode_r6, decode_r6, 8),
        (encode_r12, decode_r12, 16),
    ],
)
def test_encode_roundtrip(encode, decode, bits_per_pixel):
    data, decoded = encode_roundtrip(encode, decode, bits_per_pixel, shape=(256, 256))
    assert_allclose(data, decoded)


def encode_roundtrip_multi(encode, decode, bits_per_pixel, shape):
    # typically the header size for quad data, but doesn't really matter,
    # as we don't generate a "true" header, but just random padding at
    # the beginning of each frame.
    header_bytes = 768
    assert len(shape) == 3  # decoding multiple frames at once
    max_value = (1 << bits_per_pixel) - 1
    data = np.random.randint(0, max_value + 1, shape)
    enc_bytes_per_frame = np.prod(shape[1:]) // 8 * bits_per_pixel
    encoded = np.zeros(
        data.size // 8 * bits_per_pixel + shape[0] * header_bytes,
        dtype=np.uint8
    )
    encoded = encoded.reshape((-1, enc_bytes_per_frame + header_bytes))
    encoded_data = encoded[:, header_bytes:].reshape((shape[0], shape[1], -1))

    # encoders only do one frame per call:
    for i in range(shape[0]):
        encoded[i, :header_bytes] = np.random.randint(0, 0x100, header_bytes)
        encode(inp=data[i], out=encoded_data[i])

    decoded = np.zeros_like(data)
    decode(
        input_bytes=encoded_data,
        out=decoded,
        header_size_bytes=header_bytes,
        num_frames=shape[0]
    )
    return data, decoded


@pytest.mark.with_numba
@pytest.mark.parametrize(
    'encode,decode,bits_per_pixel', [
        (encode_r1, decode_multi_r1, 1),
        (encode_r6, decode_multi_r6, 8),
        (encode_r12, decode_multi_r12, 16),

        (encode_u1, decode_multi_u1, 8),
        (encode_u2, decode_multi_u2, 16),
    ],
)
def test_encode_roundtrip_multi(encode, decode, bits_per_pixel):
    shape = (3, 256, 256)
    data, decoded = encode_roundtrip_multi(encode, decode, bits_per_pixel, shape)
    assert_allclose(data, decoded)


def encode_roundtrip_quad(encode, decode, bits_per_pixel, num_frames=2, input_data=None):
    shape = (num_frames, 512, 512)
    header_bytes = 768
    max_value = (1 << bits_per_pixel) - 1
    if input_data is None:
        data = np.random.randint(0, max_value + 1, shape)
        # make sure min/max values are indeed hit:
        data.reshape((-1,))[0] = max_value
        data.reshape((-1,))[-1] = 0
        assert np.max(data) == max_value
        assert np.min(data) == 0
    else:
        data = input_data
    encoded_data = encode_quad(encode, data, bits_per_pixel, with_headers=False)
    decoded = np.zeros_like(data)
    decode(
        input_bytes=encoded_data.reshape((num_frames, 256, -1)),
        out=decoded,
        header_size_bytes=header_bytes,
        num_frames=shape[0]
    )
    return data, decoded


@pytest.mark.with_numba
@pytest.mark.parametrize(
    'encode,decode,bits_per_pixel', [
        (encode_r1, decode_quad_r1, 1),
        (encode_r6, decode_quad_r6, 8),
        (encode_r12, decode_quad_r12, 16),
    ],
)
def test_encode_roundtrip_quad(encode, decode, bits_per_pixel):
    data, decoded = encode_roundtrip_quad(encode, decode, bits_per_pixel, num_frames=3)
    assert_allclose(data, decoded)


@pytest.mark.parametrize(
    'counter_depth,bits_per_pixel,encode_fn,mpx_dtype', [
        (1, 1, encode_r1, 'R64'),
        (6, 8, encode_r6, 'R64'),
        (12, 16, encode_r12, 'R64'),

        (1, 8, encode_u1, 'U01'),
        (12, 16, encode_u2, 'U16'),
    ]
)
@pytest.mark.parametrize(
    'decode_fn,dest_dtype', [
        ('decode_to_u64', np.uint64),
        ('decode_to_f32', np.float32),
        ('decode_to_f64', np.float64),
    ]
)
def test_rs_decoders(
    counter_depth, bits_per_pixel, encode_fn, mpx_dtype,
    decode_fn, dest_dtype,
):
    # minimal shape for debuggability (but not too simple to hide bugs!)
    shape = (1, 128)

    decoder = QdDecoder(layout="1x1", dtype=mpx_dtype, counter_depth=counter_depth)

    max_value = (1 << counter_depth) - 1
    data = np.random.randint(0, max_value + 1, shape).astype(np.uint64)
    encoded = np.zeros(data.size // 8 * bits_per_pixel, dtype=np.uint8)
    encoded = encoded.reshape((shape[0], -1))
    encode_fn(inp=data, out=encoded)

    decoded = np.zeros_like(data, dtype=dest_dtype).reshape((-1,))

    # print(encoded.dtype, decoded.dtype)
    decode_fn = getattr(decoder, decode_fn)
    decode_fn(input=encoded.reshape(-1), output=decoded)

    if dest_dtype == np.uint64:
        print("  data", "decoded")
        for i in range(8):
            print(hex(data[0, i]), hex(decoded.reshape(shape)[0, i]))

    assert_allclose(data, decoded.reshape(shape))


@pytest.mark.parametrize(
    'counter_depth,bits_per_pixel,encode_fn,mpx_dtype', [
        (1, 1, encode_r1, 'R64'),
        (6, 8, encode_r6, 'R64'),
        (12, 16, encode_r12, 'R64'),
    ]
)
@pytest.mark.parametrize(
    'decode_fn,dest_dtype', [
        ('decode_to_u64', np.uint64),
        ('decode_to_f32', np.float32),
        ('decode_to_f64', np.float64),
    ]
)
def test_rs_decoders_quad_raw(
    counter_depth, bits_per_pixel, encode_fn, mpx_dtype,
    decode_fn, dest_dtype,
):
    shape = (2, 512, 512)

    decoder = QdDecoder(layout="2x2", dtype=mpx_dtype, counter_depth=counter_depth)

    max_value = (1 << counter_depth) - 1
    data = np.random.randint(0, max_value + 1, shape).astype(np.uint64)
    print("calling encode_quad", encode_fn, data, bits_per_pixel, False)
    encoded = encode_quad(
        encode=encode_fn,
        data=data,
        bits_per_pixel=bits_per_pixel,
        with_headers=False
    ).reshape((shape[0], -1))

    decoded = np.zeros_like(data, dtype=dest_dtype)

    assert encoded.size == data.size // 8 * bits_per_pixel

    # print(encoded.dtype, decoded.dtype)
    decode_fn = getattr(decoder, decode_fn)
    for i in range(shape[0]):
        decode_fn(input=encoded[i].reshape((-1,)), output=decoded[i].reshape((-1,)))

    print("encoded beginning:", encoded.reshape((-1,))[:32])

    if dest_dtype == np.uint64:
        print("  data", "decoded")
        for i in range(8):
            print(i, hex(data[0, 0, i]), hex(decoded.reshape(shape)[0, 0, i]))

    # for each frame?
    for i in range(shape[0]):
        assert_allclose(decoded[i].reshape(shape[1:]), desired=data[i])

    assert_allclose(decoded.reshape(shape), desired=data)


@pytest.mark.parametrize(
    'counter_depth,bits_per_pixel,encode_fn,mpx_dtype', [
        (1, 8, encode_u1, 'U01'),
        (12, 16, encode_u2, 'U16'),
    ]
)
@pytest.mark.parametrize(
    'decode_fn,dest_dtype', [
        ('decode_to_u64', np.uint64),
        ('decode_to_f32', np.float32),
        ('decode_to_f64', np.float64),
    ]
)
def test_rs_decoders_quad_uint(
    counter_depth, bits_per_pixel, encode_fn, mpx_dtype,
    decode_fn, dest_dtype,
):
    shape = (512, 512)

    decoder = QdDecoder(layout="2x2", dtype=mpx_dtype, counter_depth=counter_depth)

    max_value = (1 << counter_depth) - 1
    data = np.random.randint(0, max_value + 1, shape).astype(np.uint64)

    # quad in uint mode is just like "normal" uints:
    encoded = np.zeros(data.size // 8 * bits_per_pixel, dtype=np.uint8)
    encoded = encoded.reshape((shape[0], -1))
    encode_fn(inp=data, out=encoded)

    decoded = np.zeros_like(data, dtype=dest_dtype).reshape((-1,))

    # print(encoded.dtype, decoded.dtype)
    decode_fn = getattr(decoder, decode_fn)
    decode_fn(input=encoded.reshape((-1,)), output=decoded)

    print("encoded beginning:", encoded.reshape((-1,))[:32])

    if dest_dtype == np.uint64:
        print("  data", "decoded")
        for i in range(8):
            print(i, hex(data[0, i]), hex(decoded.reshape(shape)[0, i]))

    assert_allclose(decoded.reshape(shape), desired=data)
