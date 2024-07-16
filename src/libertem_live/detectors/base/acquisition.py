from typing import TYPE_CHECKING, Optional
from typing_extensions import Protocol
import logging
import math
import time

from libertem.common import Shape, Slice
from libertem.common.math import prod
from libertem.common.executor import (
    TaskCommHandler, TaskProtocol, WorkerQueue, JobCancelledError,
)
from libertem.io.dataset.base import (
    DataTile, TilingScheme,
)
from libertem_live.hooks import Hooks, DetermineNavShapeEnv
from sparseconverter import ArrayBackend
from opentelemetry import trace
import numpy as np

if TYPE_CHECKING:
    from .connection import DetectorConnection, PendingAcquisition
    from .controller import AcquisitionController
    from libertem.common.executor import JobExecutor
    from libertem.corrections.corrset import CorrectionSet

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class AcquisitionTimeout(Exception):
    pass


def _shape_with_placeholders(
    shape_hint: tuple[int, ...],
    nimages: int
) -> tuple[int, ...]:
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
    shape_hint: Optional[tuple[int, ...]],
) -> tuple[int, ...]:
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
        nav_shape: Optional[tuple[int, ...]] = None,
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

    def start_acquisition(self):
        raise NotImplementedError()

    def end_acquisition(self):
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
        nav_shape: Optional[tuple[int, ...]] = None,
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


class GenericCommHandler(TaskCommHandler):
    def __init__(self, conn):
        self._conn = conn

    def get_conn_impl(self):
        raise NotImplementedError()

    def handle_task(self, task: "TaskProtocol", queue: "WorkerQueue"):
        conn = self.get_conn_impl()
        with tracer.start_as_current_span("GenericCommHandler.handle_task") as span:
            span.set_attribute(
                "socket_path",
                conn.get_socket_path(),
            )
            queue.put({
                "type": "BEGIN_TASK",
                "socket": conn.get_socket_path(),
            })
            put_time = 0.0
            recv_time = 0.0
            # send the data for this task to the given worker
            partition = task.get_partition()
            slice_ = partition.slice
            start_idx = slice_.origin[0]
            end_idx = slice_.origin[0] + slice_.shape[0]
            span.set_attributes({
                "libertem.partition.start_idx": start_idx,
                "libertem.partition.end_idx": end_idx,
            })
            chunk_size = 128
            current_idx = start_idx
            while current_idx < end_idx:
                current_chunk_size = min(chunk_size, end_idx - current_idx)

                t0 = time.perf_counter()
                frame_stack = conn.get_next_stack(
                    max_size=current_chunk_size
                )
                if frame_stack is None:
                    if current_idx != end_idx:
                        queue.put({
                            "type": "END_PARTITION",
                        })
                        raise JobCancelledError("premature end of frame iterator")
                assert len(frame_stack) <= current_chunk_size, \
                    f"{len(frame_stack)} <= {current_chunk_size}"
                t1 = time.perf_counter()
                recv_time += t1 - t0
                if len(frame_stack) == 0:
                    if current_idx != end_idx:
                        raise JobCancelledError("premature end of frame iterator")
                    break

                t0 = time.perf_counter()
                serialized = frame_stack.serialize()
                queue.put({
                    "type": "FRAMES",
                }, payload=serialized)
                t1 = time.perf_counter()
                put_time += t1 - t0

                current_idx += len(frame_stack)
            queue.put({
                "type": "END_PARTITION",
            })
            span.set_attributes({
                "total_put_time": put_time,
                "total_recv_time": recv_time,
            })


class GetFrames:
    CAM_CLIENT_CLS = None
    FRAME_STACK_CLS = None

    def __init__(self, request_queue, dtype, sig_shape):
        self._request_queue = request_queue
        self._dtype = dtype
        self._sig_shape = sig_shape
        self._cam_client = None
        self._last_stack = None
        self._last_stack_pos = None
        self._buf = None
        self._end_seen = False

    def __enter__(self):
        with self._request_queue.get() as msg:
            header, _ = msg
            header_type = header["type"]
            assert header_type == "BEGIN_TASK", f"expected BEGIN_TASK, got {header_type}"
            socket_path = header["socket"]
            self._cam_client = self.CAM_CLIENT_CLS(socket_path)
        return self

    def __exit__(self, *args, **kwargs):
        if self._cam_client is not None:
            if self._last_stack is not None:
                self._cam_client.done(self._last_stack)
                self._last_stack = None
                self._last_stack_pos = None
            self._cam_client.close()
            self._cam_client = None
            self._buf = None

    def _snack(self, frame_stack, depth: int, start_idx: int):
        """
        Return at most `depth` frames from `frame_stack` as decoded data, as a view
        into a fixed array/buffer. Take note of left overs, which will be snacked upon
        on the next call.
        """
        if self._buf is None or self._buf.shape[0] < depth:
            self._buf = np.zeros((depth,) + self._sig_shape, self._dtype)

        end_idx = min(start_idx + depth, len(frame_stack))
        num_frames = end_idx - start_idx
        view = self._buf[0:num_frames, ...]
        self._cam_client.decode_range_into_buffer(
            frame_stack,
            view,
            start_idx,
            end_idx
        )

        if end_idx < len(frame_stack):
            # we are not done, store our position:
            self._last_stack = frame_stack
            self._last_stack_pos = end_idx  # end_idx is the new start_idx
        else:
            self._last_stack = None
            self._last_stack_pos = None
            self._cam_client.done(frame_stack)
        return view

    def get_partition_tile(self, depth: int) -> np.ndarray:
        """
        Like `next_tile`, but always reads _all_ FRAMES messages that belong
        to the current partition, and decodes them directly into a
        larger buffer.
        """
        assert self._cam_client is not None, "should be connected"

        buf = np.zeros((depth,) + self._sig_shape, self._dtype)
        start_idx = 0
        while True:
            with self._request_queue.get() as msg:
                header, payload = msg
                header_type = header["type"]
                if header_type == "FRAMES":
                    frame_stack = self.FRAME_STACK_CLS.deserialize(payload)
                    num_frames = len(frame_stack)
                    view = buf[start_idx:start_idx+num_frames, ...]
                    self._cam_client.decode_range_into_buffer(
                        frame_stack,
                        view,
                        0,
                        len(frame_stack)
                    )
                    self._cam_client.done(frame_stack)
                    start_idx += num_frames
                elif header_type == "END_PARTITION":
                    self._end_seen = True
                    return buf
                else:
                    raise RuntimeError(
                        f"invalid header type {header['type']}; FRAME or END_PARTITION expected;"
                        f" (header={header})"
                    )

    def next_tile(self, depth: int) -> Optional[np.ndarray]:
        """
        Consume a FRAMES messages from the request queue, and return the next
        tile, or return None if we get an END_PARTITION message (which we also
        consume). The tile will contain less than or equal to `depth` frames.
        """

        assert self._cam_client is not None, "should be connected"

        if self._last_stack is not None:
            return self._snack(self._last_stack, depth, self._last_stack_pos)

        while True:
            with self._request_queue.get() as msg:
                header, payload = msg
                header_type = header["type"]
                if header_type == "FRAMES":
                    frame_stack = self.FRAME_STACK_CLS.deserialize(payload)
                    return self._snack(frame_stack, depth, 0)
                elif header_type == "END_PARTITION":
                    self._end_seen = True
                    if self._last_stack is not None:
                        return self._snack(self._last_stack, depth, self._last_stack_pos)
                    return None
                else:
                    raise RuntimeError(
                        f"invalid header type {header['type']}; FRAME or END_PARTITION expected;"
                        f" (header={header})"
                    )

    def expect_end(self):
        if self._end_seen:
            return
        with self._request_queue.get() as msg:
            header, _ = msg
            header_type = header["type"]
            if header_type != "END_PARTITION":
                raise RuntimeError(
                    f"invalid header type {header['type']}; END_PARTITION expected;"
                    f" (header={header})"
                )
            self._end_seen = True

    def get_tiles(
        self,
        to_read: int,
        start_idx: int,
        tiling_scheme: "TilingScheme",
        corrections: 'Optional["CorrectionSet"]' = None,
        roi=None,
        array_backend: 'Optional["ArrayBackend"]' = None,
    ):
        tile_start = start_idx
        depth = tiling_scheme.depth
        frames_in_tile = 0
        while to_read > 0:
            if tiling_scheme.intent == "partition":
                raw_tile = self.get_partition_tile(depth=to_read)
            else:
                raw_tile = self.next_tile(depth)
                if raw_tile is None:
                    raise RuntimeError(
                        f"we were still expecting to read {to_read} frames more!"
                    )

            frames_in_tile = raw_tile.shape[0]
            to_read -= frames_in_tile

            if tiling_scheme.intent == "partition":
                assert to_read == 0

            if raw_tile.shape[0] == 0:
                assert to_read == 0
                continue  # we are done and the buffer is empty

            tile_shape = Shape(
                (frames_in_tile,) + tuple(tiling_scheme[0].shape),
                sig_dims=2
            )
            tile_slice = Slice(
                origin=(tile_start,) + (0, 0),
                shape=tile_shape,
            )
            if corrections is not None:
                corrections.apply(raw_tile, tile_slice)
            yield DataTile(
                raw_tile,
                tile_slice=tile_slice,
                scheme_idx=0,
            )
            tile_start += frames_in_tile
        self.expect_end()
        logger.debug("GetFrames.get_tiles: end of method")
