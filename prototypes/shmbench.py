import time
from multiprocessing import shared_memory, Process, Queue, managers, Event
import numpy as np


def receiver(rx, e, start_e):
    start_e.set()
    t0 = time.time()
    seen = {}
    nbytes = 0
    while True:
        res = rx.get()
        if res is None:
            t1 = time.time()
            delta = t1 - t0
            print(
                f"receiver is shutting down, was alive for {delta}s, {nbytes/delta/1024/1024/1024}GiB/s"
            )
            return
        else:
            if res in seen:
                obj = seen[res]
            else:
                obj = seen[res] = shared_memory.SharedMemory(create=False, name=res)
            np.frombuffer(obj.buf, dtype=np.uint8).sum()
            nbytes += obj.buf.nbytes
        if e.is_set():
            print(q.qsize())
            e.clear()
        # else: do nothing?


if __name__ == "__main__":
    zeros = np.zeros((512, 512, 512), dtype=np.uint8).reshape((-1,))
    shm_zeros = shared_memory.SharedMemory(create=True, size=zeros.nbytes)
    shm_zeros_arr = np.frombuffer(shm_zeros.buf, dtype=np.uint8)
    shm_zeros_arr[:] = zeros

    q = Queue()
    e = Event()
    start_e = Event()

    p = Process(target=receiver, args=(q, e, start_e))
    p.start()

    start_e.wait()

    t0 = time.time()
    for i in range(10*8):
        q.put(shm_zeros.name)
    t1 = time.time()
    print(t1-t0)

    e.set()

    print("shutting down")
    q.put(None)

    print(f"waiting for receiver to die, qsize={q.qsize()}")
    p.join()

    print("freeing shared memory")
    shm_zeros.unlink()
