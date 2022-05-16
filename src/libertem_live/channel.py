import pickle
import math
import contextlib
import multiprocessing as mp
from multiprocessing import shared_memory
from multiprocessing.managers import SharedMemoryManager
from queue import Empty
from typing import Callable, NamedTuple, Optional, Tuple

import numpy as np


class PoolAllocation(NamedTuple):
    shm_name: str
    handle: int  # internal handle, could be an offset
    full_size: int  # full size of the allocation, in bytes (including padding)
    req_size: int  # requested allocation size


class PoolShmClient:
    def __init__(self):
        self._shm_cache = {}

    def get(self, allocation: PoolAllocation) -> memoryview:
        offset = allocation.handle
        name = allocation.shm_name
        if name in self._shm_cache:
            shm = self._shm_cache[name]
        else:
            self._shm_cache[name] = shm = shared_memory.SharedMemory(name=name, create=False)
        return shm.buf[offset:offset+allocation.req_size]


class PoolShmAllocator:
    ALIGN_TO = 4096

    def __init__(self, item_size, size_num_items, create=True, name=None):
        """
        Bump plus free list allocator. Can allocate objects of a fixed size.
        Memory above the `_used` offset is free, allocated blocks from the
        `_free` list are free, everything else is in use.

        Allocation and recycling should happen on the same process.
        """
        self._item_size = item_size
        size = item_size * size_num_items
        size = self.ALIGN_TO * math.ceil(size / self.ALIGN_TO)
        self._size = size
        self._shm = shared_memory.SharedMemory(create=create, name=name, size=size)
        self._free = []  # list of byte offsets into `_shm`
        self._used = 0  # byte offset into `_shm`; everything above is free memory

    def allocate(self, req_size: int) -> PoolAllocation:
        if req_size > self._item_size:
            raise RuntimeError(f"allocation request for size {req_size} cannot be serviced")
        # 1) check free list, get an item from there
        if len(self._free) > 0:
            offset = self._free.pop()
        # 2) if free list is empty, bump up `_used` and return a "new" block of
        #    memory
        elif self._used + self._item_size < self._size:
            offset = self._used
            self._used += self._item_size
        else:
            raise RuntimeError("pool shm is out of memory")
        return PoolAllocation(
            shm_name=self._shm.name,
            handle=offset,
            full_size=self._item_size,
            req_size=req_size,
        )

    def get(self, allocation: PoolAllocation) -> memoryview:
        offset = allocation.handle
        assert allocation.shm_name == self._shm.name
        return self._shm.buf[offset:offset+allocation.req_size]

    def recycle(self, allocation: PoolAllocation):
        self._free.append(allocation.handle)

    def shutdown(self):
        self._shm.unlink()

    @property
    def name(self):
        return self._shm.name


class ShmQueue:
    def __init__(self):
        self.q = mp.Queue()
        self.release_q = mp.Queue()
        self._psa = None
        self._psc = None

    def put(self, header, payload: Optional[memoryview] = None):
        """
        Send the (header, payload) tuple via this channel - copying the
        `payload` to a shared memory segment while sending `header` plainly
        via a queue. The header should be `pickle`able.
        """
        # FIXME: zero-copy version of this, which could be used to
        # receive/decode/...  directly into a shared memory segment. Not
        # important now, as zmq doesn't have a `recv_into` anyways...
        if payload is not None:
            payload_shm = self._copy_to_shm(payload)
        else:
            payload_shm = None
        self.q.put((pickle.dumps(header), 'bytes', payload_shm))

    def _copy_to_shm(self, src_buffer: memoryview) -> str:
        """
        Copy the `buffer` to shared memory and return its name
        """
        if self._psa is None:
            # FIXME: config item size, pool size
            self._psa = PoolShmAllocator(item_size=512*512*4*2, size_num_items=4096)
        size = src_buffer.nbytes
        try:
            alloc_handle = self.release_q.get_nowait()
        except Empty:
            alloc_handle = self._psa.allocate(src_buffer.nbytes)
        payload_shm = self._psa.get(alloc_handle)
        payload_arr = np.frombuffer(src_buffer, dtype=np.uint8)
        payload_arr_shm = np.frombuffer(payload_shm[:size], dtype=np.uint8)
        payload_arr_shm[:] = payload_arr
        return alloc_handle

    def _get_named_shm(self, name: str) -> shared_memory.SharedMemory:
        return shared_memory.SharedMemory(name=name, create=False)

    @contextlib.contextmanager
    def get(self, timeout: Optional[float] = None):
        """
        Receive a message. Memory of the payload will be cleaned up after the
        context manager scope, so don't keep references outside of it!

        Parameters
        ----------
        timeout
            Timeout in seconds,
        """
        if self._psc is None:
            self._psc = PoolShmClient()
        while True:
            try:
                header, typ, payload_handle = self.q.get(timeout=timeout)
            except Empty:
                continue
            try:
                if payload_handle is not None:
                    payload_buf = self._psc.get(payload_handle)
                    payload_memview = memoryview(payload_buf)
                else:
                    payload_buf = None
                    payload_memview = None
                if typ == "bytes":
                    yield (pickle.loads(header), payload_memview)
            finally:
                if payload_memview is not None:
                    payload_memview.release()
                if payload_buf is not None:
                    self.release_q.put(payload_handle)
            break

    def empty(self):
        return self.q.empty()


class ChannelManager:
    def __init__(self):
        self._smm = SharedMemoryManager()
        self._smm.start()

    def shutdown(self):
        self._smm.shutdown()


class WorkerQueues(NamedTuple):
    request: ShmQueue
    response: ShmQueue


class WorkerPool:
    def __init__(self, processes: int, worker_fn: Callable):
        self._cm = ChannelManager()
        self._workers: Tuple[WorkerQueues, mp.Process] = []
        self._worker_fn = worker_fn
        self._processes = processes
        self._start_workers()

    @property
    def size(self):
        return self._processes

    def _start_workers(self):
        for i in range(self._processes):
            queues = self._make_worker_queues()
            p = mp.Process(target=self._worker_fn, args=(queues,))
            p.start()
            self._workers.append((queues, p))

    def all_worker_queues(self):
        for (qs, _) in self._workers:
            yield qs

    def join_all(self):
        for (_, p) in self._workers:
            p.join()

    def get_worker_queues(self, idx) -> WorkerQueues:
        return self._workers[idx][0]

    def poll_all(self) -> Optional[WorkerQueues]:
        # FIXME: is there a way to just poll all queues together?
        for (qs, p) in self._workers:
            if not qs.response.empty():
                return qs

    def _make_worker_queues(self):
        return WorkerQueues(
            request=ShmQueue(),
            response=ShmQueue(),
        )
