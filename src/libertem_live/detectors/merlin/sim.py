import os
import sys
import mmap
import time
import socket
import itertools
import functools
import platform
import threading
from typing import Optional
import logging

import click
import numpy as np
from tqdm import tqdm

from libertem.io.dataset.base import TilingScheme
from libertem.io.dataset.mib import MIBDataSet
from libertem.common import Shape


logger = logging.getLogger(__name__)


# Copied from the k2is branch as a temporary fix.
# FIXME use a proper import as soon as the k2is branch is merged
class StoppableThreadMixin:
    def __init__(self, *args, **kwargs):
        self._stop_event = threading.Event()
        super().__init__(*args, **kwargs)

    def stop(self):
        self._stop_event.set()

    def is_stopped(self):
        return self._stop_event.is_set()


# Copied from the k2is branch as a temporary fix.
# FIXME use a proper import as soon as the k2is branch is merged
class ErrThreadMixin(StoppableThreadMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._error: Optional[Exception] = None

    def get_error(self):
        return self._error

    def error(self, exc):
        logger.error("got exception %r, shutting down thread", exc)
        self._error = exc
        self.stop()

    def maybe_raise(self):
        if self._error is not None:
            raise self._error


@functools.lru_cache
def get_mpx_header(length):
    return b"MPX,%010d," % ((length + 1),)


class MITExecutor:
    '''
    This is an incomplete implementation of a LiberTEM executor that is
    just sufficient to initialize a MIB dataset.

    This avoids a dependency on the GPL-licensed
    :class:`~libertem.executor.inline.InlineJobExecutor`.
    '''
    def run_function(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)


class StopException(Exception):
    pass


class DataSocketSimulator:
    def __init__(self, path: str, continuous=False, rois=None,
                 max_runs=-1):
        """
        Parameters
        ----------

        path
            Path to the HDR file

        continuous
            If set to True, will continuously output data

        rois: List[np.ndarray]
            If a list of ROIs is given, in continuous mode, cycle through
            these ROIs from the source data

        max_runs: int
            Maximum number of continuous runs
        """
        self.stop_event = threading.Event()
        if rois is None:
            rois = []
        if not path.lower().endswith(".hdr"):
            raise ValueError("please pass the path to the HDR file!")
        self._path = path
        self._continuous = continuous
        self._rois = rois
        self._ds = None
        self._max_runs = max_runs
        self._mmaps = {}

    def open(self):
        ds = MIBDataSet(path=self._path)
        ds.initialize(MITExecutor())
        print("dataset shape: %s" % (ds.shape,))
        self._ds = ds
        self._warmup()

    def get_chunks(self):
        """
        generator of `bytes` for the given configuration
        """
        # first, send acquisition header:
        with open(self._path, 'rb') as f:
            # FIXME: possibly change header in continuous mode?
            hdr = f.read()
            yield get_mpx_header(len(hdr))
            yield hdr
        if self._continuous:
            print("yielding from continuous")
            yield from self._get_continuous()
        else:
            print("yielding from single scan")
            roi = np.ones(self._ds.shape.nav, dtype=bool)
            t = tqdm(total=np.count_nonzero(roi))
            try:
                for item in self._get_single_scan(roi):
                    yield item
                    t.update(1)
            finally:
                t.close()

    def _read_frame_w_header(self, fh, frame_idx, full_frame_size):
        """
        Parameters
        ----------

        fh : LocalFile

        frame_idx : int
            File-relative frame index

        full_frame_size : int
            Size of header plus frame in bytes
        """
        if fh._file is None:
            fh.open()
        f = fh._file
        fileno = f.fileno()
        if fileno not in self._mmaps:
            self._mmaps[fileno] = raw_mmap = mmap.mmap(
                fileno=f.fileno(),
                length=0,
                offset=0,
                access=mmap.ACCESS_READ,
            )
        else:
            raw_mmap = self._mmaps[fileno]

        return bytearray(raw_mmap[
            full_frame_size * frame_idx: full_frame_size * (frame_idx + 1)
        ])

    def _warmup(self):
        fileset = self._ds._get_fileset()
        ds_shape = self._ds.shape
        tiling_scheme = TilingScheme.make_for_shape(
            tileshape=Shape((1,) + tuple(ds_shape.sig),
                            sig_dims=ds_shape.sig.dims),
            dataset_shape=ds_shape,
        )
        slices, ranges, scheme_indices = fileset.get_read_ranges(
            start_at_frame=0,
            stop_before_frame=int(np.prod(self._ds.shape.nav)),
            dtype=np.float32,  # FIXME: don't really care...
            tiling_scheme=tiling_scheme,
            roi=None,
        )

    def _get_single_scan(self, roi):
        fileset = self._ds._get_fileset()
        ds_shape = self._ds.shape
        tiling_scheme = TilingScheme.make_for_shape(
            tileshape=Shape((1,) + tuple(ds_shape.sig),
                            sig_dims=ds_shape.sig.dims),
            dataset_shape=ds_shape,
        )
        slices, ranges, scheme_indices = fileset.get_read_ranges(
            start_at_frame=0,
            stop_before_frame=int(np.prod(self._ds.shape.nav)),
            dtype=np.float32,  # FIXME: don't really care...
            tiling_scheme=tiling_scheme,
            roi=roi,
        )

        first_file = self._ds._files_sorted[0]
        header_size = first_file.fields['header_size_bytes']

        full_frame_size = header_size + first_file.fields['image_size_bytes']

        mpx_header = get_mpx_header(full_frame_size)

        for idx in range(slices.shape[0]):
            if self.is_stopped():
                raise StopException("Server stopped")
            origin = slices[idx, 0]
            # shape = slices[idx, 1]
            # origin, shape = slices[idx]
            tile_ranges = ranges[idx][0]
            file_idx = tile_ranges[0]
            fh = fileset[file_idx]
            global_idx = origin[0]
            local_idx = global_idx - fh.start_idx
            frame_w_header = self._read_frame_w_header(fh, local_idx, full_frame_size)
            yield mpx_header
            yield frame_w_header

    def _get_continuous(self):
        if self._rois:
            rois = self._rois
        else:
            rois = [np.ones(self._ds.shape.nav, dtype=bool)]

        i = 0
        for roi in itertools.cycle(rois):
            t0 = time.time()
            yield from self._get_single_scan(roi)
            t1 = time.time()
            print("cycle %d took %.05fs" % (i, t1 - t0))
            i += 1
            if self._max_runs != -1 and i >= self._max_runs:
                raise StopException("max_runs exceeded")

    def handle_conn(self, conn):
        for chunk in self.get_chunks():
            conn.sendall(chunk)
        conn.close()

    def is_stopped(self):
        return self.stop_event.is_set()


class CachedDataSocketSim(DataSocketSimulator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = None

    def _get_single_scan(self, roi):
        first_file = self._ds._files_sorted[0]
        header_size = first_file.fields['header_size_bytes']
        full_frame_size = header_size + first_file.fields['image_size_bytes']
        mpx_size = 15
        num_images = self._ds.shape.nav.size
        if roi is not None:
            num_images = np.count_nonzero(roi)

        cache_size = num_images * (full_frame_size + mpx_size)

        if self._cache is None:
            try:
                self._cache = np.empty(cache_size, dtype=np.uint8)
                cache_view = memoryview(self._cache)
                offset = 0
                for chunk in super()._get_single_scan(roi):
                    yield chunk
                    length = len(chunk)
                    cache_view[offset:offset+length] = chunk
                    offset += length
            except Exception:
                self._cache = None  # discard in case of problem, only keep fully populated cache
        else:
            chunk_size = 16*1024*1024
            cache_view = memoryview(self._cache)
            for offset in range(0, cache_size, chunk_size):
                yield cache_view[offset:offset+chunk_size]


class MemfdSocketSim(DataSocketSimulator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def open(self):
        try:
            # lazy import - make the sim work without pymemfd (example: Windows)
            import memfd
        except ImportError:
            if platform.system() == 'Linux':
                raise RuntimeError("Please install `pymemfd` to use the memfd cache.")
            else:
                raise RuntimeError("The memfd cache is only supported on Linux.")
        super().open()
        self._cache_fd = memfd.memfd_create("sim_cache", 0)
        self._populate_cache()

    def _populate_cache(self):
        print("populating cache, please wait...")
        roi = np.ones(self._ds.shape.nav, dtype=bool)
        total_size = 0
        for chunk in super()._get_single_scan(roi):
            if self.is_stopped():
                raise StopException("Server stopped")
            os.write(self._cache_fd, chunk)
            total_size += len(chunk)
        os.lseek(self._cache_fd, 0, 0)
        self._size = total_size
        print("cache populated, total size = %d MiB" % (total_size / 1024 / 1024))

    def _send_full_file(self, conn):
        os.lseek(self._cache_fd, 0, 0)
        total_sent = 0
        reps = 0
        while total_sent < self._size:
            if self.is_stopped():
                raise StopException("Server stopped")
            total_sent += os.sendfile(
                conn.fileno(),
                self._cache_fd,
                total_sent,  # offset ->
                self._size - total_sent
            )
            reps += 1
        print("_send_full_file took %d reps" % reps)

    def handle_conn(self, conn):
        # first, send acquisition header:
        with open(self._path, 'rb') as f:
            # FIXME: possibly change header in continuous mode?
            hdr = f.read()
            conn.sendall(get_mpx_header(len(hdr)))
            conn.sendall(hdr)
        if self._continuous:
            i = 0
            while True:
                t0 = time.time()
                self._send_full_file(conn)
                t1 = time.time()
                throughput = self._size / (t1 - t0) / 1024 / 1024
                print("cycle %d took %.05fs (%.2fMiB/s)" % (i, t1 - t0, throughput))
                i += 1
                if self._max_runs != -1 and i >= self._max_runs:
                    raise StopException("max_runs exceeded")
        else:
            print("yielding from single scan")
            t0 = time.time()
            self._send_full_file(conn)
            t1 = time.time()
            throughput = self._size / (t1 - t0) / 1024 / 1024
            print("single scan took %.05fs (%.2fMiB/s)" % (t1 - t0, throughput))
        conn.close()


class DataSocketServer(ErrThreadMixin, threading.Thread):
    def __init__(self, sim: DataSocketSimulator, port=6342):
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._port = port
        self._sim = sim
        super().__init__()
        self._sim.stop_event = self._stop_event

    def run(self):
        try:
            self._sim.open()
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._socket.bind(('0.0.0.0', self._port))
            self._socket.settimeout(1)
            self._socket.listen(1)
            print("listening")
            while not self.is_stopped():
                try:
                    connection, client_addr = self._socket.accept()
                    with connection:
                        print("accepted from %s" % (client_addr,))
                        self._sim.handle_conn(connection)
                except socket.timeout:
                    continue
                except ConnectionResetError:
                    print("disconnected")
                except RuntimeError as e:
                    print("exception %s -> stopping" % e)
                    self.error(e)
                    break
                except StopException:
                    break
                except Exception as e:
                    print("exception? %s" % e)
                    self.error(e)
            print("exiting data server")
        except Exception as e:
            return self.error(e)


class ControlSocketServer(ErrThreadMixin, threading.Thread):
    def __init__(self, sim, port=6341):
        self._sim = sim
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._port = port
        self._params = {}
        super().__init__()

    def handle_conn(self, connection):
        connection.settimeout(1)
        print("handling control connection")
        # This code is only a proof of concept. It works just well enough to send
        # commands and parse responses with control.MerlinControl.
        # A few points to improve:
        # * Handle incomplete messages properly. Right now,
        #   the assumption is that a complete command will be received with
        #   recv() and a complete response will be sent with send()
        # * Figure out the missing parts of the response to match Merlin behavior
        # * Actually respond to commands that can be simulated, such as 'numframes'
        # * Possibly emit errors like a real Merlin detector upon bad commands?
        while not self.is_stopped():
            try:
                chunk = connection.recv(1024*1024)
            except socket.timeout:
                continue
            if len(chunk) == 0:
                print("closed control")
                return
            parts = chunk.split(b',')
            print("Control command received: ", chunk)
            parts = [part.decode('ascii') for part in parts]
            method = parts[2]
            param = parts[3]
            if method == 'SET':
                value = parts[4]
            if method == 'GET':
                response_parts = (
                    "noideafirst",
                    "noideasecond",
                    method,
                    "noideafourth",
                    self._params.get(param, "undef"),
                    "0"
                )
            elif method == 'SET':
                self._params[param] = value
                response_parts = (
                    "noideafirst",  # 0
                    "noideasecond",  # 1
                    method,  # 2
                    "noideafourth",  # 3
                    "0",  # 4
                )
            elif method == 'CMD':
                self._params[param] = value
                response_parts = (
                    "noideafirst",  # 0
                    "noideasecond",  # 1
                    method,  # 2
                    "noideafourth",  # 3
                    "0",  # 4
                )
            else:
                raise RuntimeError("Unknown method %s", method)
            response_str = ','.join(response_parts)
            print("Control response: ", response_str)
            connection.send(response_str.encode('ascii'))

    @property
    def sim(self):
        return self._sim

    def run(self):
        try:
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._socket.bind(('0.0.0.0', self._port))
            self._socket.settimeout(1)
            self._socket.listen(1)
            print("control port listening")
            while not self.is_stopped():
                try:
                    connection, client_addr = self._socket.accept()
                    with connection:
                        print("accepted control from %s" % (client_addr,))
                        self.handle_conn(connection)
                except socket.timeout:
                    continue
                except ConnectionResetError:
                    print("control disconnected")
                except RuntimeError as e:
                    self.error(e)
                    print("control exception %s -> stopping" % e)
                    break
                except Exception as e:
                    print("control exception? %s" % e)
                    self.error(e)
            print("exiting control server")
        except Exception as e:
            return self.error(e)


@click.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('--continuous', default=False, is_flag=True)
@click.option('--cached', default='NONE', type=click.Choice(
    ['NONE', 'MEM', 'MEMFD'], case_sensitive=False)
)
@click.option('--data-port', type=int, default=6342)
@click.option('--control-port', type=int, default=6341)
@click.option('--max-runs', type=int, default=-1)
def main(path, continuous, data_port, control_port, cached, max_runs):
    if cached == 'MEM':
        sim = CachedDataSocketSim(path=path, continuous=continuous, max_runs=max_runs)
    elif cached == 'MEMFD':
        sim = MemfdSocketSim(path=path, continuous=continuous, max_runs=max_runs)
    else:
        sim = DataSocketSimulator(path=path, continuous=continuous, max_runs=max_runs)

    control_t = ControlSocketServer(sim=sim, port=control_port)
    # Make sure the thread dies with the main program
    control_t.daemon = True
    control_t.start()

    server_t = DataSocketServer(sim=sim, port=data_port)
    # Make sure the thread dies with the main program
    server_t.daemon = True
    server_t.start()

    # This allows us to handle Ctrl-C, and the main program
    # stops in a timely fashion when continuous scanning stops.
    try:
        while control_t.is_alive() and server_t.is_alive():
            control_t.maybe_raise()
            server_t.maybe_raise()
            time.sleep(1)
    except KeyboardInterrupt:
        # Just to not print "Aborted!"" from click
        sys.exit(0)
    finally:
        print("Stopping...")
        control_t.stop()
        server_t.stop()
        timeout = 2
        start = time.time()
        while True:
            control_t.maybe_raise()
            server_t.maybe_raise()
            if (not control_t.is_alive()) and (not server_t.is_alive()):
                break

            if (time.time() - start) >= timeout:
                print("Killing server threads")
                # Since the threads are daemon threads, they will die abruptly
                # when this main thread finishes.
                break

            time.sleep(0.1)


if __name__ == "__main__":
    main()
