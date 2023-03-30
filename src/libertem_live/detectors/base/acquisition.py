from typing import TYPE_CHECKING, Optional, Tuple
from contextlib import contextmanager
import logging

if TYPE_CHECKING:
    from .connection import DetectorConnection, PendingAcquisition
    from .controller import AcquisitionController
    from libertem_live.hooks import Hooks

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
        self._conn = conn
        self._hooks = hooks
        self._nav_shape = nav_shape
        self._frames_per_partition = frames_per_partition
        self._controller = controller
        self._pending_aq = pending_aq
        self._hooks = hooks
        super().__init__()

    @contextmanager
    def acquire(self):
        raise NotImplementedError()
