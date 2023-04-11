from typing import Tuple

import pytest

from libertem_live.api import Hooks
from libertem_live.detectors.base.acquisition import (
    determine_nav_shape,
)
from libertem_live.detectors.base.connection import (
    PendingAcquisition,
)


class MockPendingAq(PendingAcquisition):
    def __init__(self, nimages: int):
        self._nimages = nimages

    @property
    def nimages(self) -> int:
        return self._nimages


def test_square_shape_fail():
    mock_pending_aq = MockPendingAq(
        nimages=128*128+1,
    )
    with pytest.raises(RuntimeError) as m:
        determine_nav_shape(
            hooks=Hooks(),
            pending_aq=mock_pending_aq,
            controller=None,
            shape_hint=None,
        )
    m.match(r"^Can't handle non-square scans.*")


def test_square_shape_success():
    mock_pending_aq = MockPendingAq(
        nimages=128*128,
    )
    assert determine_nav_shape(
        hooks=Hooks(),
        pending_aq=mock_pending_aq,
        controller=None,
        shape_hint=None,
    ) == (128, 128)


def test_invalid_placeholder_shape_zeros():
    mock_pending_aq = MockPendingAq(
        nimages=128*128,
    )
    with pytest.raises(ValueError) as m:
        determine_nav_shape(
            hooks=Hooks(),
            pending_aq=mock_pending_aq,
            controller=None,
            shape_hint=(0, 1, 2, -1),
        )
    m.match(r'^shape cannot contain zeros$')


def test_invalid_placeholder_shape_divisible():
    mock_pending_aq = MockPendingAq(
        nimages=128*128,
    )
    with pytest.raises(ValueError) as m:
        determine_nav_shape(
            hooks=Hooks(),
            pending_aq=mock_pending_aq,
            controller=None,
            shape_hint=(-1, 17),
        )
    m.match(r'^number of images \(16384\) must be divisible')  # etc...


def test_invalid_placeholder_too_many():
    mock_pending_aq = MockPendingAq(
        nimages=128*128,
    )
    with pytest.raises(ValueError) as m:
        determine_nav_shape(
            hooks=Hooks(),
            pending_aq=mock_pending_aq,
            controller=None,
            shape_hint=(-1, -1, -1),
        )
    m.match(
        r'^shape can only contain up to two placeholders \(-1\); shape is \(-1, -1, -1\)$'
    )


@pytest.mark.parametrize(
    ['nimages', 'shape_hint', 'expected'],
    [
        # nice and round:
        ((128*128), (-1, -1), (128, 128)),
        ((128*128), (-1, 128), (128, 128)),
        ((128*128), (64, -1), (64, 256)),
        ((128*128), (-1, 64), (256, 64)),
        ((128*128), (-1, 64, 64), (4, 64, 64)),
        ((128*128), (4, -1, -1), (4, 64, 64)),

        # ugly:
        ((17*29), (-1, 17), (29, 17)),
        (1, (-1, -1), (1, 1)),
        (1, (-1, -1, 1), (1, 1, 1)),
    ]
)
def test_shapes_with_placeholders(
    nimages: int, shape_hint: Tuple[int, ...], expected: Tuple[int, ...],
):
    mock_pending_aq = MockPendingAq(
        nimages=nimages,
    )
    assert determine_nav_shape(
        hooks=Hooks(),
        pending_aq=mock_pending_aq,
        controller=None,
        shape_hint=shape_hint,
    ) == expected
