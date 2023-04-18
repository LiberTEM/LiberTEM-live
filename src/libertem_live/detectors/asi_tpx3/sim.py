#!/usr/bin/env python
import threading
from typing import Optional, List
import logging
import socket
import time
import sys
import os

import numpy as np
import click

from libertem_live.detectors.common import (
    ServerThreadMixin,
    UndeadException,
)

logger = logging.getLogger(__name__)


def send_full_file(cache_fd, total_size, conn):
    os.lseek(cache_fd, 0, 0)
    total_sent = 0
    while total_sent < total_size:
        total_sent += os.sendfile(
            conn.fileno(),
            cache_fd,
            total_sent,
            total_size - total_sent,
        )


class CachedDataSource:
    def send_data(self, conn: socket.socket) -> int:
        raise NotImplementedError()

    @property
    def full_size(self) -> int:
        raise NotImplementedError()


class MemfdCachedSource(CachedDataSource):
    def __init__(self, paths: List[str]):
        self._paths = paths
        self._cache_data(paths)

    def _cache_data(self, paths: List[str]):
        import memfd  # local import so it works without it
        logger.info("populating cache...")
        cache_fd = memfd.memfd_create("tpx_cache", 0)
        total_size = 0
        for path in paths:
            written = 0
            logger.info(f"reading {path}")
            with open(path, "rb") as f:
                in_bytes = f.read()
                while written < len(in_bytes):
                    written += os.write(cache_fd, in_bytes[written:])
            total_size += written
        os.lseek(cache_fd, 0, 0)
        self._cache_fd = cache_fd
        self._total_size = total_size
        logger.info(f"cache populated, total_size={total_size}")

    def send_data(self, conn: socket.socket) -> int:
        send_full_file(self._cache_fd, self._total_size, conn)
        return self.full_size

    @property
    def full_size(self) -> int:
        return self._total_size


class BufferedCachedSource(CachedDataSource):
    def __init__(self, paths: List[str]):
        self._paths = paths
        self._cache_data(paths)

    def _cache_data(self, paths: List[str]):
        logger.info("populating cache...")
        total_size = 0
        for path in paths:
            total_size += os.stat(path).st_size

        self._cache = np.empty(total_size, dtype=np.uint8)

        for path in paths:
            cache_view = memoryview(self._cache)  # type: ignore
            offset = 0
            with open(path, "rb") as f:
                in_bytes = f.read()
                length = len(in_bytes)
                cache_view[offset:offset+length] = in_bytes
                offset += length

        self._total_size = total_size
        logger.info(f"cache populated, total_size={total_size}")

    def send_data(self, conn: socket.socket) -> int:
        cache_view = memoryview(self._cache)  # type: ignore
        conn.sendall(cache_view)
        return self.full_size

    @property
    def full_size(self) -> int:
        return self._total_size


class TpxSim:
    def __init__(
        self,
        sleep: float,
        data_source: CachedDataSource,
        stop_event: Optional[threading.Event] = None,
    ):
        self._sleep = sleep
        if stop_event is None:
            stop_event = threading.Event()
        self.stop_event = stop_event
        self._data_source = data_source

    def handle_conn(self, conn: socket.socket):
        total_sent_this_conn = 0
        try:
            while True:
                if self.stop_event.is_set():
                    break
                logger.info("sending full file...")
                t0 = time.perf_counter()
                try:
                    size = self._data_source.send_data(conn)
                except Exception as e:
                    logger.error("exception while sending data: %s", str(e))
                    raise
                t1 = time.perf_counter()
                thp = size / 1024 / 1024 / (t1-t0)
                total_sent_this_conn += size
                logger.info(f"done in {t1-t0:.3f}s, throughput={thp:.3f}MiB/s")
                time.sleep(self._sleep)
                t2 = time.perf_counter()
                thp_with_sleep = size / 1024 / 1024 / (t2-t0)
                logger.info(f"throughput_with_sleep={thp_with_sleep:.3f}MiB/s")
        finally:
            total = total_sent_this_conn / 1024 / 1024 / 1024
            logger.info(
                f"connection closed, total_sent = {total:.3f}GiB"
            )


class DataSocketSimulator(ServerThreadMixin, threading.Thread):
    def __init__(self, sim: TpxSim, host: str, port: int, **kwargs):
        super().__init__(host=host, port=port, name='AsiTpx3Sim', **kwargs)
        self._sim = sim

    def handle_conn(self, connection: socket.socket):
        self._sim.handle_conn(connection)


class TpxCameraSim:
    def __init__(
        self,
        paths: List[str],
        cached: str,
        port: int,
        sleep: float,
    ):
        src: CachedDataSource
        if cached.lower() == 'mem':
            src = BufferedCachedSource(paths=paths)
        elif cached.lower() == 'memfd':
            src = MemfdCachedSource(paths=paths)
        else:
            raise ValueError(f"unknown cache type: {cached}")

        self.stop_event = threading.Event()

        sim = TpxSim(
            sleep=sleep,
            stop_event=self.stop_event,
            data_source=src,
        )

        self.server_t = DataSocketSimulator(
            stop_event=self.stop_event,
            host='localhost',
            port=port,
            sim=sim,
        )
        self.server_t.daemon = True

    def start(self):
        self.server_t.start()

    def is_alive(self):
        return self.server_t.is_alive()

    def maybe_raise(self):
        self.server_t.maybe_raise()

    def wait_for_listen(self):
        self.server_t.wait_for_listen()

    def stop(self):
        logger.info("Stopping...")
        self.stop_event.set()
        timeout = 2
        start = time.time()
        while True:
            self.server_t.maybe_raise()
            if not self.server_t.is_alive():
                break

            if (time.time() - start) >= timeout:
                # Since the threads are daemon threads, they will die abruptly
                # when this main thread finishes. This is at the discretion of the caller.
                raise UndeadException("Server threads won't die")
            time.sleep(0.1)

        logger.info(f"stopping took {time.time() - start}s")


@click.command()
@click.argument(
    'paths',
    type=click.Path(
        exists=True, dir_okay=False, file_okay=True, readable=True
    ),
    nargs=-1
)
@click.option('--sleep', type=float, default=0.0)
@click.option('--port', type=int, default=8283)
@click.option('--cached', default='MEM', type=click.Choice(
    ['MEM', 'MEMFD'], case_sensitive=False)
)
def main(cached: str, paths, sleep: float, port: int = 8283):
    logging.basicConfig(level=logging.INFO)

    logger.info(f"port={port}")

    sim = TpxCameraSim(
        paths=paths,
        cached=cached,
        port=port,
        sleep=sleep,
    )
    sim.start()

    # This allows us to handle Ctrl-C, and the main program
    # stops in a timely fashion when continuous scanning stops.
    try:
        while sim.is_alive():
            sim.maybe_raise()
            time.sleep(1)
    except KeyboardInterrupt:
        # Just to not print "Aborted!"" from click
        sys.exit(0)
    finally:
        logger.info("Stopping...")
        try:
            sim.stop()
        except UndeadException:
            logger.warn("Killing server threads")
            # Since the threads are daemon threads, they will die abruptly
            # when this main thread finishes.
            return


if __name__ == "__main__":
    main()
