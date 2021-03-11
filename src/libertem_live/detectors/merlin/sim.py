import os
import sys
import mmap
import time
import socket
import itertools
import functools

import click
import numpy as np
from tqdm import tqdm

from libertem.io.dataset.base import TilingScheme
from libertem.executor.inline import InlineJobExecutor
from libertem.api import Context
from libertem.common import Shape


@functools.lru_cache
def get_mpx_header(length):
    return b"MPX,%010d," % ((length + 1),)


class DataSocketSimulator:
    def __init__(self, path: str, continuous=False, rois=None, max_runs=-1):
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
        if rois is None:
            rois = []
        if not path.lower().endswith(".hdr"):
            raise ValueError("please pass the path to the HDR file!")
        self._path = path
        self._continuous = continuous
        self._rois = rois
        self._ctx = Context(executor=InlineJobExecutor())
        self._ds = None
        self._max_runs = max_runs
        self._mmaps = {}

    def open(self):
        ds = self._ctx.load("mib", path=self._path)
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
                raise RuntimeError("max_runs exceeded")

    def handle_conn(self, conn):
        for chunk in self.get_chunks():
            conn.sendall(chunk)
        conn.close()


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
            raise RuntimeError("Please install `pymemfd` to use the memfd cache")
        super().open()
        self._cache_fd = memfd.memfd_create("sim_cache", 0)
        self._populate_cache()

    def _populate_cache(self):
        print("populating cache, please wait...")
        roi = np.ones(self._ds.shape.nav, dtype=bool)
        total_size = 0
        for chunk in super()._get_single_scan(roi):
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
                    raise RuntimeError("max_runs exceeded")
        else:
            print("yielding from single scan")
            t0 = time.time()
            self._send_full_file(conn)
            t1 = time.time()
            throughput = self._size / (t1 - t0) / 1024 / 1024
            print("single scan took %.05fs (%.2fMiB/s)" % (t1 - t0, throughput))
        conn.close()


class DataSocketServer:
    def __init__(self, sim: DataSocketSimulator, port=6342):
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._port = port
        self._sim = sim

    def run(self):
        self._sim.open()
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind(('0.0.0.0', self._port))
        self._socket.listen(1)
        print("listening")
        while True:
            try:
                connection, client_addr = self._socket.accept()
                with connection:
                    print("accepted from %s" % (client_addr,))
                    self._sim.handle_conn(connection)
            except ConnectionResetError:
                print("disconnected")
            except RuntimeError as e:
                print("exception %s -> stopping" % e)
                break
            except Exception as e:
                print("exception? %s" % e)


@click.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('--continuous', default=False, is_flag=True)
@click.option('--cached', default='NONE', type=click.Choice(
    ['NONE', 'MEM', 'MEMFD'], case_sensitive=False)
)
@click.option('--port', type=int, default=6342)
@click.option('--max-runs', type=int, default=-1)
def main(path, continuous, port, cached, max_runs):
    if cached == 'MEM':
        sim = CachedDataSocketSim(path=path, continuous=continuous, max_runs=max_runs)
    elif cached == 'MEMFD':
        sim = MemfdSocketSim(path=path, continuous=continuous, max_runs=max_runs)
    else:
        sim = DataSocketSimulator(path=path, continuous=continuous, max_runs=max_runs)
    server = DataSocketServer(sim=sim, port=port)
    server.run()
    sys.exit(0)


if __name__ == "__main__":
    main()
