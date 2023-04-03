from typing import TYPE_CHECKING, Optional, Tuple, Protocol
from contextlib import contextmanager
import logging

from libertem.common import Shape
from libertem_live.hooks import Hooks

if TYPE_CHECKING:
    from .connection import DetectorConnection, PendingAcquisition
    from .controller import AcquisitionController
    from libertem.common.executor import JobExecutor

logger = logging.getLogger(__name__)


class AcquisitionTimeout(Exception):
    pass


class AcquisitionMixin:
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
        if hooks is None:
            hooks = Hooks()
        self._conn = conn
        self._nav_shape = nav_shape
        self._frames_per_partition = frames_per_partition
        self._controller = controller
        self._pending_aq = pending_aq
        self._hooks = hooks
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
