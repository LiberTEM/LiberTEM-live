from typing import TYPE_CHECKING, Optional, Tuple
from typing_extensions import Protocol
from contextlib import contextmanager
import logging
import math

from libertem.common import Shape
from libertem.common.math import prod
from libertem_live.hooks import Hooks, DetermineNavShapeEnv

if TYPE_CHECKING:
    from .connection import DetectorConnection, PendingAcquisition
    from .controller import AcquisitionController
    from libertem.common.executor import JobExecutor

logger = logging.getLogger(__name__)


class AcquisitionTimeout(Exception):
    pass


def _shape_with_placeholders(
    shape_hint: Tuple[int, ...],
    nimages: int
) -> Tuple[int, ...]:
    fixed = tuple(val for val in shape_hint if val > -1)
    if 0 in fixed:
        raise ValueError("shape cannot contain zeros")
    fixed_prod = prod(fixed)

    num_placeholders = len(shape_hint) - len(fixed)

    total_to_be_distributed, zero_rest = divmod(nimages, fixed_prod)

    if zero_rest != 0:
        raise ValueError(
            f"number of images ({nimages}) must be divisible by the fixed "
            f"parts of the shape ({fixed_prod}), but has rest {zero_rest}"
        )

    if num_placeholders == 1:
        replacement = total_to_be_distributed
    elif num_placeholders == 2:
        replacement = int(math.sqrt(total_to_be_distributed))
    else:
        raise ValueError(
            f"shape can only contain up to two placeholders (-1); shape is {shape_hint}"
        )

    result = []
    for s in shape_hint:
        if s == -1:
            result.append(replacement)
        else:
            result.append(s)

    assert prod(result) == nimages

    return tuple(result)


def determine_nav_shape(
    hooks: "Hooks",
    pending_aq: "PendingAcquisition",
    controller: Optional["AcquisitionController"],
    shape_hint: Optional[Tuple[int, ...]],
) -> Tuple[int, ...]:
    nimages = pending_aq.nimages
    # Order of operations to determine `nav_shape`:
    # - 1) If a concrete `nav_shape` is given as `shape_hint`, use that
    #   (this method is not called in that case)
    # - 2) Call `Hooks.on_determine_nav_shape` and use that if possible
    # - 3) If a `nav_shape` with placeholders, i.e. `-1` entries, is given,
    #   use the number of images to fill these placeholders
    # - 4) If no `nav_shape` is give, ask the pending acquisition or controller
    # - 5) If the controller doesn't know, try to make a 2D square
    # - 6) If all above fails, raise an Exception

    # case 2: use the hook results
    hook_result = hooks.on_determine_nav_shape(DetermineNavShapeEnv(
        nimages=nimages,
        shape_hint=shape_hint,
    ))
    if hook_result is not None:
        if -1 in hook_result:
            raise ValueError(
                f"Result from `Hooks.on_determine_nav_shape` should "
                f"be a tuple of integers, without placeholders "
                f"(got {hook_result})"
            )
        if prod(hook_result) != nimages:
            raise ValueError(
                f"Result from `Hooks.on_determine_nav_shape` ({hook_result}) is not "
                f"compatible with number of images ({nimages})"
            )
        return hook_result

    # case 3: placeholders
    if shape_hint is not None and -1 in shape_hint:
        return _shape_with_placeholders(
            shape_hint=shape_hint,
            nimages=nimages,
        )

    # case 4.0: ask the `PendingAcquisition`:
    if pending_aq.nav_shape is not None:
        return pending_aq.nav_shape

    # case 4.1: ask the controller, if we have one
    if controller is not None:
        try:
            new_shape = controller.determine_nav_shape(
                nimages=nimages,
            )
            if new_shape is not None:
                return new_shape
        except NotImplementedError:
            pass

    # case 5: try to make a square shape
    side = int(math.sqrt(nimages))
    if side * side != nimages:
        # case 6: can't make a square shape, raise Exception
        raise RuntimeError(
            "Can't handle non-square scans by default, please override"
            " `Hooks.determine_nav_shape` or pass in a concrete nav_shape"
        )
    return (side, side)


class AcquisitionMixin:
    def __init__(
        self,
        *,
        conn: "DetectorConnection",
        frames_per_partition: int,
        nav_shape: Optional[Tuple[int, ...]] = None,
        controller: Optional["AcquisitionController"] = None,
        pending_aq: Optional["PendingAcquisition"] = None,
        hooks: Optional["Hooks"] = None,
    ):
        if hooks is None:
            hooks = Hooks()
        self._conn = conn
        self._controller = controller
        self._pending_aq = pending_aq
        self._hooks = hooks

        if nav_shape is None or -1 in nav_shape:
            if pending_aq is None:
                raise RuntimeError(
                    "In active mode, please pass the full `nav_shape`"
                )
            nav_shape = determine_nav_shape(
                hooks=hooks,
                controller=controller,
                shape_hint=nav_shape,
                pending_aq=pending_aq,
            )
            logger.info(f"determined nav_shape: {nav_shape}")

        self._nav_shape = nav_shape
        frames_per_partition = min(frames_per_partition, prod(nav_shape))
        self._frames_per_partition = frames_per_partition

        super().__init__()

    @contextmanager
    def acquire(self):
        raise NotImplementedError()


class AcquisitionProtocol(Protocol):
    """
    Methods and attributed that are guaranteed to be available on
    Acquisition objects.
    """
    # NOTE: this protocol is needed as mypy doesn't support an obvious way to
    # "intersect" two types, i.e. the symmetric operation to Union that
    # gives you access to properties of a set of types (in this case, `AcquisitionMixin`
    # and `DataSet` would be the appropriate types)

    def __init__(
        self,
        *,
        conn: "DetectorConnection",
        nav_shape: Optional[Tuple[int, ...]] = None,
        frames_per_partition: Optional[int] = None,
        controller: Optional["AcquisitionController"] = None,
        pending_aq: Optional["PendingAcquisition"] = None,
        hooks: Optional["Hooks"] = None,
    ):
        ...

    @property
    def shape(self) -> Shape:
        """
        The shape of the acquisition, includes both navigation and signal
        dimensions.
        """
        ...

    def initialize(self, executor: "JobExecutor") -> "AcquisitionProtocol":
        ""
        ...
