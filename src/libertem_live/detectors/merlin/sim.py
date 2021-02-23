import sys
import mmap
import time
import socket
import itertools

import click
import numpy as np
from tqdm import tqdm

from libertem.io.dataset.base import TilingScheme
from libertem.executor.inline import InlineJobExecutor
from libertem.api import Context
from libertem.common import Shape


class DataSocketSimulator:
    def __init__(self, path: str, continuous=False, rois=None):
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

    def open(self):
        ds = self._ctx.load("mib", path=self._path)
        print("dataset shape: %s" % (ds.shape,))
        self._ds = ds
        self._warmup()

    def make_chunk(self, buf):
        # plus 1 because the length includes the comma
        return b"MPX,%010d,%s" % (len(buf) + 1, buf)

    def get_chunks(self):
        """
        generator of `bytes` for the given configuration
        """
        # first, send acquisition header:
        with open(self._path, 'rb') as f:
            # FIXME: possibly change header in continuous mode?
            hdr = f.read()
            yield self.make_chunk(hdr)
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
        raw_mmap = mmap.mmap(
            fileno=f.fileno(),
            length=0,
            offset=0,
            access=mmap.ACCESS_READ,
        )
        return raw_mmap[
            full_frame_size * frame_idx: full_frame_size * (frame_idx + 1)
        ]

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

        for idx in range(slices.shape[0]):
            origin, shape = slices[idx]
            tile_ranges = ranges[idx][0]
            file_idx = tile_ranges[0]
            fh = fileset[file_idx]
            global_idx = origin[0]
            local_idx = global_idx - fh.start_idx
            yield self.make_chunk(
                self._read_frame_w_header(fh, local_idx, full_frame_size)
            )

    def _get_continuous(self):
        if self._rois:
            rois = self._rois
        else:
            rois = [np.ones(self._ds.shape.nav, dtype=bool)]

        for roi in itertools.cycle(rois):
            t0 = time.time()
            yield from self._get_single_scan(roi)
            t1 = time.time()
            print("cycle took %.05fs" % (t1 - t0))


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
            self._cache = np.zeros(cache_size, dtype=np.uint8)
            offset = 0
            for chunk in super()._get_single_scan(roi):
                yield chunk
                self._cache[offset:offset+len(chunk)] = np.frombuffer(chunk, dtype=np.uint8)
                offset += len(chunk)
        else:
            chunk_size = 2*1024*1024  # 1MiB
            for offset in range(0, cache_size, chunk_size):
                yield self._cache[offset:offset+chunk_size]


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
                    chunks = self._sim.get_chunks()
                    for chunk in chunks:
                        connection.sendall(chunk)
            except ConnectionResetError:
                print("disconnected")
            except Exception as e:
                print("exception? %s" % e)


@click.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('--continuous', default=False, is_flag=True)
@click.option('--port', type=int, default=6342)
def main(path, continuous, port):
    # sim = DataSocketSimulator(path=path, continuous=continuous)
    sim = CachedDataSocketSim(path=path, continuous=continuous)
    sim.open()
    server = DataSocketServer(sim=sim, port=port)
    server.run()
    sys.exit(0)


if __name__ == "__main__":
    main()
