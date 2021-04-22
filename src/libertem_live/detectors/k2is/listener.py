import os
import time
import socket
import struct
import contextlib
import threading

import click
import numpy as np
import numba
import hexdump

from libertem.io.dataset.k2is import DataBlock
from libertem.common.buffers import bytes_aligned, zeros_aligned
from libertem_live.utils.net import mcast_socket
from libertem_live.detectors.k2is.decode import decode_uint12_le
from libertem_live.detectors.common import StoppableThreadMixin


GROUP = '225.1.1.1'


class MsgReaderThread(StoppableThreadMixin, threading.Thread):
    def __init__(self, idx, port, affinity_set, local_addr='0.0.0.0', iface='enp193s0f0', timeout=0.1, *args, **kwargs):
        self.idx = idx
        self.port = port
        self.affinity_set = affinity_set
        self.iface = iface
        self.local_addr = local_addr
        self.timeout = timeout
        self.e = threading.Event()
        super().__init__(*args, **kwargs)

    def read_loop(self, s):
        # NOTE: non-IS data is truncated - we only read the first 0x5758 bytes of the message
        buf = bytes_aligned(0x5758)
        s.settimeout(self.timeout)
        packets = 0
        i = 0
        while True:
            if self.is_stopped():
                return
            try:
                p = s.recvmsg_into([buf])
                assert p[0] == 0x5758
            except socket.timeout:
                continue
            
            yield (buf, p[1])
            packets += 1

    def run(self):
        print(f"thread {threading.get_native_id()}")
        os.sched_setaffinity(0, self.affinity_set)
        self.e.wait()
        print(f"listening on {self.local_addr}:{self.port}/{GROUP} on {self.iface}")
        
        x_offset = self.idx * 256

        frames_seen = set()
        
        with mcast_socket(self.port, GROUP, self.local_addr, self.iface) as s:
            buf = zeros_aligned((930, 16), dtype=np.uint16)
            buf_flat = buf.reshape((-1,))

            print("entry MsgReaderThread, waiting for first packet")
            
            first_frame_id = None
            i = 0
            for p in self.read_loop(s):
                i += 1
                decode_uint12_le(inp=p[0][40:], out=buf_flat)
                h = np.frombuffer(p[0], dtype=DataBlock.header_dtype, count=1, offset=0)
                if first_frame_id is None:
                    first_frame_id = int(h['frame_id'])
                frame_idx = int(h['frame_id']) - first_frame_id
                
                slice_ = (
                    frame_idx,
                    slice(h['pixel_y_start'][0], h['pixel_y_end'][0] + 1),
                    slice(h['pixel_x_start'][0] + x_offset, h['pixel_x_end'][0] + 1 + x_offset),
                )

                frames_seen.add(frame_idx)
            print(f"{self.idx}, {self.port}, {self.iface}: {i}, {len(frames_seen)}")


@click.command()
@click.argument('index', type=int)
@click.argument('timeout', type=int)
@click.argument('iface', type=str)
def main(index, timeout, iface):
    threads = [
        MsgReaderThread(
            idx=index,
            iface=iface,
            local_addr='225.1.1.1',
            port=2001 + index,
            affinity_set={8 + index}
        ),
    ]

    warmup_buf_out = zeros_aligned((930, 16), dtype=np.uint16).reshape((-1,))
    warmup_buf_inp = zeros_aligned(0x5758, dtype=np.uint8)

    decode_uint12_le(inp=warmup_buf_inp[40:], out=warmup_buf_out)

    try:
        for t in threads:
            t.start()

        # time.sleep(30) # → uncomment for tracing purposes

        for t in threads:
            t.e.set()

        time.sleep(timeout)
    finally:
        for t in threads:
            t.stop()
            t.join()


if __name__ == "__main__":
    main()
