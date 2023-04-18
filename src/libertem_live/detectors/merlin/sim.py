import os
import sys
import mmap
import time
import socket
import itertools
import functools
import platform
import threading
import logging
import select
from typing import List, Dict

import click
import numpy as np
from tqdm import tqdm

from libertem.io.dataset.base import TilingScheme
from libertem.io.dataset.mib import MIBDataSet, is_valid_hdr
from libertem.common import Shape

from libertem_live.detectors.merlin.data import AcquisitionHeader
from libertem_live.detectors.common import (
    UndeadException, StopException, ServerThreadMixin,
)


logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=None)
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


class HeaderSocketSimulator:
    def __init__(
        self,
        path: str,
        first_frame_headers: Dict,
        stop_event=None,
        nav_shape=None,
        continuous=False,
        rois=None,
    ):
        """
        This class handles sending out acquisition header - calling the
        `handle_conn` method will send the header to the give connection.

        Parameters
        ----------

        path
            Path to the HDR file

        stop_event
            Will stop gracefully if this event is set.

        continuous
            If set to True, will continuously output data

        rois: List[np.ndarray]
            If a list of ROIs is given, in continuous mode, cycle through
            these ROIs from the source data

        first_frame_headers
            The frame headers of the first frame, as a dict (as LiberTEM reads
            them)
        """
        if stop_event is None:
            stop_event = threading.Event()
        self.stop_event = stop_event
        if rois is None:
            rois = []
        if nav_shape is None and not is_valid_hdr(path):
            raise ValueError("please pass the path to a valid HDR file or specify a nav shape!")
        self._path = path
        self._nav_shape = nav_shape
        self._continuous = continuous
        self._rois = rois
        self._ds = None
        self._first_frame_headers = first_frame_headers

    def _make_hdr(self):
        # FIXME: support for continuous mode - need to fake a better header here in that case
        bpp = self._first_frame_headers['bits_per_pixel']
        hdr = (
            f"HDR,\n"
            f"Frames in Acquisition (Number):\t{np.prod(self._nav_shape, dtype=np.int64)}\n"
            f"Frames per Trigger (Number):\t{self._nav_shape[1]}\n"
            f"Counter Depth (number):\t{bpp}\n"
            f"End\t"
        )
        return hdr.encode('latin1')

    @property
    def hdr(self) -> bytes:
        if is_valid_hdr(self._path):
            with open(self._path, 'rb') as f:
                # FIXME: possibly change header in continuous mode?
                hdr = f.read()
        else:
            hdr = self._make_hdr()
        return hdr

    @property
    def parsed(self) -> AcquisitionHeader:
        return AcquisitionHeader.from_raw(self.hdr)

    def get_chunks(self):
        """
        generator of `bytes` for the given configuration
        """
        # first, send acquisition header:
        hdr = self.hdr
        yield get_mpx_header(len(hdr))
        yield hdr

    def handle_conn(self, conn):
        for chunk in self.get_chunks():
            conn.sendall(chunk)

    def is_stopped(self):
        return self.stop_event.is_set()


class DataSocketSimulator:
    def __init__(self, path: str, stop_event=None, nav_shape=None, continuous=False, rois=None,
                 max_runs=-1):
        """
        Parameters
        ----------

        path
            Path to the HDR file

        stop_event
            Will stop gracefully if this event is set.

        continuous
            If set to True, will continuously output data

        rois: List[np.ndarray]
            If a list of ROIs is given, in continuous mode, cycle through
            these ROIs from the source data

        max_runs: int
            Maximum number of continuous runs
        """
        if stop_event is None:
            stop_event = threading.Event()
        self.stop_event = stop_event
        if rois is None:
            rois = []
        if nav_shape is None and not is_valid_hdr(path):
            raise ValueError("please pass the path to a valid HDR file or specify a nav shape!")
        self._path = path
        self._nav_shape = nav_shape
        self._continuous = continuous
        self._rois = rois
        self._ds = None
        self._max_runs = max_runs
        self._mmaps = {}

    def get_ds_shape(self) -> Shape:
        self._load()
        assert self._ds is not None
        return self._ds.shape

    def _load(self):
        if self._ds is None:
            ds = MIBDataSet(path=self._path, nav_shape=self._nav_shape)
            ds.initialize(MITExecutor())
            self._ds = ds

    def open(self):
        self._load()
        assert self._ds is not None
        logger.info(f"dataset shape: {self._ds.shape}")
        self._warmup()

    def get_chunks(self):
        """
        generator of `bytes` for the given configuration
        """
        if self._continuous:
            logger.info("yielding from continuous")
            yield from self._get_continuous()
        else:
            logger.info("yielding from single scan")
            roi = np.ones(self._ds.shape.nav, dtype=bool)
            # Times two since _get_single_scan() returns header
            # and frame separately
            t = tqdm(total=np.count_nonzero(roi)*2)
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

        fh : MMapFile or LocalFile

        frame_idx : int
            File-relative frame index

        full_frame_size : int
            Size of header plus frame in bytes
        """
        if hasattr(fh, '_file'):
            if fh._file is None:
                fh.open()
            f = fh._file
        else:
            f = open(fh._path, 'rb')
        path = fh._path
        if path not in self._mmaps:
            self._mmaps[path] = raw_mmap = mmap.mmap(
                fileno=f.fileno(),
                length=0,
                offset=0,
                access=mmap.ACCESS_READ,
            )
        else:
            raw_mmap = self._mmaps[path]

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

    @property
    def first_frame_headers(self) -> Dict:
        self.open()
        first_file = self._ds._files_sorted[0]
        return first_file.fields

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
            frame_w_header = self._read_frame_w_header(
                fh, local_idx, full_frame_size
            )
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
            logger.info("cycle %d took %.05fs" % (i, t1 - t0))
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
        logger.info("populating cache, please wait...")
        roi = np.ones(self._ds.shape.nav, dtype=bool)
        total_size = 0
        for chunk in super()._get_single_scan(roi):
            if self.is_stopped():
                raise StopException("Server stopped")
            written = 0
            while written < len(chunk):
                written += os.write(self._cache_fd, chunk[written:])
            total_size += len(chunk)
        os.lseek(self._cache_fd, 0, 0)
        self._size = total_size
        logger.info(
            "cache populated, total size = %d MiB (%d bytes)" % (
                total_size / 1024 / 1024, total_size
            )
        )

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
        logger.info("_send_full_file took %d reps" % reps)

    def handle_conn(self, conn):
        if self._continuous:
            i = 0
            while True:
                t0 = time.time()
                self._send_full_file(conn)
                t1 = time.time()
                throughput = self._size / (t1 - t0) / 1024 / 1024
                logger.info("cycle %d took %.05fs (%.2fMiB/s)" % (i, t1 - t0, throughput))
                i += 1
                if self._max_runs != -1 and i >= self._max_runs:
                    raise StopException("max_runs exceeded")
        else:
            logger.info("yielding from single scan")
            t0 = time.time()
            self._send_full_file(conn)
            t1 = time.time()
            throughput = self._size / (t1 - t0) / 1024 / 1024
            logger.info(f"single scan took {t1 - t0:.05f}s ({throughput:.2f}MiB/s)")
        conn.close()


def wait_with_socket(event: threading.Event, connection):
    while True:
        if event.wait(0.1):
            # the event is set, break out of the loop and
            # return True to signal that normal operation
            # can continue
            return True
        (readable, writable, exceptional) = select.select(
            [connection], [connection], [connection], 0.1
        )
        if readable:
            # The connection is unidirectional, we don't receive any data
            # normally, but only send
            res = connection.recv(1)
            if not res:
                logger.info("Readable socket yielded no result, probably closed")
                # The socket was closed, return False to signal abort
                return False


class DataSocketServer(ServerThreadMixin, threading.Thread):
    def __init__(self, headers: HeaderSocketSimulator, sim: DataSocketSimulator,
            stop_event, acquisition_event, trigger_event, finish_event,
            host='0.0.0.0', port=6342, wait_trigger=False, garbage=False, manual_trigger=False):
        self.acquisition_event = acquisition_event
        self.trigger_event = trigger_event
        self.finish_event = finish_event
        self._headers = headers
        self._sim = sim
        super().__init__(host=host, port=port, name=self.__class__.__name__, stop_event=stop_event)

        self._wait_trigger = wait_trigger
        if garbage and not wait_trigger:
            raise ValueError("Can only send garbage if wait_trigger is set!")
        self._manual_trigger = manual_trigger
        if wait_trigger and manual_trigger:
            raise ValueError('Cannot have both wait softtrigger and manual trigger')
        self._garbage = garbage

    def get_sim(self):
        return self._sim

    def run(self):
        try:
            self._sim.open()
            super().run()
        except Exception as e:
            return self.error(e)

    def handle_conn(self, connection):
        try:
            if self._wait_trigger or self._manual_trigger:
                if self._garbage:
                    logger.info("Sending some garbage...")
                    connection.send(b'GARBAGEGARBAGEGARBAGE'*1024)
                logger.info("Waiting for acquisition start...")
                if not wait_with_socket(self.acquisition_event, connection):
                    logger.info("Readable socket yielded no result, probably closed")
                    connection.close()
                    return
                self.acquisition_event.clear()
            self._headers.handle_conn(connection)
            if self._manual_trigger:
                _ = input("Press key to trigger...\n")
                self.trigger_event.set()
            if self._wait_trigger or self._manual_trigger:
                logger.info("Waiting for trigger...")
                if not wait_with_socket(self.trigger_event, connection):
                    logger.info("Readable socket yielded no result, probably closed")
                    connection.close()
                    return
                self.trigger_event.clear()
            return self._sim.handle_conn(connection)
        finally:
            self.finish_event.set()


class ControlSocketServer(ServerThreadMixin, threading.Thread):
    def __init__(
        self, acquisition_event, trigger_event, stop_event=None,
        host='0.0.0.0', port=6341, initial_params=None,
    ):
        self._trigger_event = trigger_event
        self._acquisition_event = acquisition_event
        self._params = {}
        if initial_params is not None:
            self._params = initial_params
        super().__init__(host=host, port=port, name=self.__class__.__name__, stop_event=stop_event)

    def encode_response(self, response_parts: List[str]) -> bytes:
        resp_len = len(",".join(response_parts).encode("ASCII"))
        parts = [
            "MPX",
            f"{resp_len:010d}",  # 1
        ] + response_parts
        response_str = ','.join(parts)
        return response_str.encode("ascii")

    def handle_conn(self, connection):
        connection.settimeout(1)
        logger.info("handling control connection")
        # This code is only a proof of concept. It works just well enough to send
        # commands and parse responses with control.MerlinControl.
        # A few points to improve:
        # * Handle incomplete messages properly. Right now,
        #   the assumption is that a complete command will be received with
        #   recv() and a complete response will be sent with send()
        # * Figure out the missing parts of the response to match Merlin behavior
        # * Actually respond to commands that can be simulated, such as 'numframes'
        # * Possibly emit errors like a real Merlin detector upon bad commands?
        #       error codes:
        #       0 -> success
        #       1 -> busy
        #       2 -> command not recognized
        #       3 -> parameter out of range
        while not self.is_stopped():
            try:
                chunk = connection.recv(1024*1024)
            except socket.timeout:
                continue
            if len(chunk) == 0:
                logger.info("closed control")
                return
            parts = chunk.split(b',')
            logger.info(f"Control command received: {chunk}")
            parts = [part.decode('ascii') for part in parts]
            method = parts[2]
            param = parts[3]
            if method == 'GET':
                response = self.encode_response(
                    response_parts=[
                        method,
                        param,
                        self._params.get(param, "undef"),
                        "0"  # error code: success
                    ]
                )
            elif method == 'SET':
                # handle the actual operation:
                value = parts[4]
                self._params[param] = value

                # and generate a response
                response = self.encode_response(
                    response_parts=[
                        method,  # 2
                        param,  # 3
                        "0",  # 4 - error code: success
                    ]
                )
            elif method == 'CMD':
                if param == 'SOFTTRIGGER':
                    self._trigger_event.set()
                elif param == 'STARTACQUISITION':
                    self._acquisition_event.set()

                response = self.encode_response(
                    response_parts=[
                        method,  # 2
                        param,  # 3
                        "0",  # error code: success
                    ]
                )
            else:
                raise RuntimeError("Unknown method %s", method)
            logger.info(f"Control response: {str(response)}")
            connection.send(response)


class TriggerSocketServer(ServerThreadMixin, threading.Thread):
    def __init__(self, trigger_event, finish_event, stop_event=None, host='0.0.0.0', port=6343):
        self._trigger_event = trigger_event
        self._finish_event = finish_event
        super().__init__(host, port, name=self.__class__.__name__, stop_event=stop_event)

    def handle_conn(self, connection):
        connection.settimeout(1)
        logger.info("handling trigger connection")
        while not self.is_stopped():
            try:
                chunk = connection.recv(1024)
            except socket.timeout:
                continue
            if len(chunk) == 0:
                logger.info("closed trigger control")
                return
            if chunk == b'TRIGGER\n':
                logger.info("Triggered, waiting for finish!")
                self._finish_event.clear()
                self._trigger_event.set()
                self._finish_event.wait()
                connection.send(b'FINISH\n')
                logger.info("Finished!")
                self._finish_event.clear()


class TriggerClient():
    def __init__(self, host='localhost', port=6343):
        self._host = host
        self._port = port
        self._socket = None

    def connect(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self._host, self._port))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.settimeout(1)
        self._socket = s

    def trigger(self):
        assert self._socket is not None, "need to be connected"
        self._socket.send(b'TRIGGER\n')

    def wait(self):
        assert self._socket is not None, "need to be connected"
        while True:
            try:
                res = self._socket.recv(1024)
                if res == b'FINISH\n':
                    break
            except socket.timeout:
                pass

    def close(self):
        if self._socket is not None:
            self._socket.close()
            self._socket = None


class CameraSim:
    def __init__(
        self, path, nav_shape, continuous=False,
        host='0.0.0.0', data_port=6342, control_port=6341,
        trigger_port=6343, wait_trigger=False, garbage=False,
        cached=None, max_runs=-1, initial_params=None,
        manual_trigger=False,
    ):
        if garbage:
            wait_trigger = True

        if cached == 'MEM':
            cls = CachedDataSocketSim
        elif cached == 'MEMFD':
            cls = MemfdSocketSim
        else:
            cls = DataSocketSimulator

        if nav_shape == (0, 0):
            nav_shape = None

        self.stop_event = threading.Event()
        self.acquisition_event = threading.Event()
        self.trigger_event = threading.Event()
        self.finish_event = threading.Event()

        self.sim = cls(
            path=path, nav_shape=nav_shape, continuous=continuous, max_runs=max_runs,
            stop_event=self.stop_event,
        )

        self.headers = HeaderSocketSimulator(
            path=path, nav_shape=nav_shape, continuous=continuous,
            stop_event=self.stop_event, first_frame_headers=self.sim.first_frame_headers,
        )

        self.server_t = DataSocketServer(
            headers=self.headers, sim=self.sim, host=host, port=data_port,
            wait_trigger=wait_trigger, garbage=garbage, manual_trigger=manual_trigger,
            stop_event=self.stop_event,
            acquisition_event=self.acquisition_event,
            trigger_event=self.trigger_event,
            finish_event=self.finish_event,
        )
        # Make sure the thread dies with the main program
        self.server_t.daemon = True

        if initial_params is None:
            # make some effort to populate useful parameters:
            shape = self.server_t.get_sim().get_ds_shape()
            acq_hdr = self.headers.parsed
            initial_params = {
                'IMAGEX': str(shape.sig[1]),
                'IMAGEY': str(shape.sig[0]),
                'COUNTERDEPTH': acq_hdr.raw_keys['Counter Depth (number)'],
            }

        self.control_t = ControlSocketServer(
            host=host,
            port=control_port,
            stop_event=self.stop_event,
            acquisition_event=self.acquisition_event,
            trigger_event=self.trigger_event,
            initial_params=initial_params,
        )
        # Make sure the thread dies with the main program
        self.control_t.daemon = True

        self.trigger_t = TriggerSocketServer(
            host=host, port=trigger_port,
            stop_event=self.stop_event,
            trigger_event=self.trigger_event,
            finish_event=self.finish_event,
        )
        # Make sure the thread dies with the main program
        self.trigger_t.daemon = True

    def start(self):
        self.server_t.start()
        self.control_t.start()
        self.trigger_t.start()

    def wait_for_listen(self):
        self.server_t.wait_for_listen()
        self.control_t.wait_for_listen()
        self.trigger_t.wait_for_listen()

    def is_alive(self):
        return self.control_t.is_alive() and self.server_t.is_alive() and self.trigger_t.is_alive()

    def maybe_raise(self):
        self.control_t.maybe_raise()
        self.server_t.maybe_raise()
        self.trigger_t.maybe_raise()

    def stop(self):
        logger.info("Stopping...")
        self.stop_event.set()
        timeout = 2
        start = time.time()
        while True:
            self.control_t.maybe_raise()
            self.server_t.maybe_raise()
            self.trigger_t.maybe_raise()
            if (
                    (not self.control_t.is_alive())
                    and (not self.server_t.is_alive())
                    and (not self.trigger_t.is_alive())
            ):
                break

            if (time.time() - start) >= timeout:
                # Since the threads are daemon threads, they will die abruptly
                # when this main thread finishes. This is at the discretion of the caller.
                raise UndeadException("Server threads won't die")
            time.sleep(0.1)
        logger.info(f"stopping took {time.time() - start}s")


@click.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('--nav-shape', type=(int, int), default=(0, 0))
@click.option('--continuous', default=False, is_flag=True)
@click.option('--cached', default='NONE', type=click.Choice(
    ['NONE', 'MEM', 'MEMFD'], case_sensitive=False)
)
@click.option(
    '--host', type=str, default='0.0.0.0',
    help="Address to listen on (data, control, and trigger sockets)",
)
@click.option('--data-port', type=int, default=6342)
@click.option('--control-port', type=int, default=6341)
@click.option('--trigger-port', type=int, default=6343)
@click.option(
    '--wait-trigger', default=False, is_flag=True,
    help="Wait for a SOFTTRIGGER command on the control port, "
         "or a trigger signal on the trigger socket",
)
@click.option(
    '--manual-trigger', default=False, is_flag=True,
    help="Wait for a manual trigger by user input after ARM",
)
@click.option(
    '--garbage', default=False, is_flag=True,
    help="Send garbage before trigger. Implies --wait-trigger"
)
@click.option('--max-runs', type=int, default=-1)
def main(path, nav_shape, continuous,
        host, data_port, control_port, trigger_port, wait_trigger,
        manual_trigger, garbage, cached, max_runs):
    logging.basicConfig(level=logging.INFO)
    camera_sim = CameraSim(
        path=path, nav_shape=nav_shape, continuous=continuous,
        host=host, data_port=data_port, control_port=control_port, trigger_port=trigger_port,
        wait_trigger=wait_trigger, garbage=garbage, cached=cached, max_runs=max_runs,
        manual_trigger=manual_trigger,
    )

    camera_sim.start()
    # This allows us to handle Ctrl-C, and the main program
    # stops in a timely fashion when continuous scanning stops.
    try:
        while camera_sim.is_alive():
            camera_sim.maybe_raise()
            time.sleep(1)
    except KeyboardInterrupt:
        # Just to not print "Aborted!"" from click
        sys.exit(0)
    finally:
        print("Stopping...")
        try:
            camera_sim.stop()
        except UndeadException:
            print("Killing server threads")
            # Since the threads are daemon threads, they will die abruptly
            # when this main thread finishes.
            return


if __name__ == "__main__":
    main()
