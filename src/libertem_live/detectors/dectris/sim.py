import sys
import time
import logging
import threading
import signal
import multiprocessing
from multiprocessing.synchronize import Event as EventClass
import json
import mmap
from typing import TYPE_CHECKING, Dict, Generator, Optional, Tuple
import urllib

import zmq
import click
import numpy as np
from flask import Flask, request, Blueprint, current_app
from werkzeug.serving import make_server

from libertem_live.detectors.common import (
    ErrThreadMixin, UndeadException, set_thread_name,
)

if TYPE_CHECKING:
    import socket

try:
    from werkzeug.serving import prepare_socket
except ImportError:
    def prepare_socket(hostname: str, port: int) -> "socket.socket":
        import socket
        from werkzeug.serving import (
            select_address_family, get_sockaddr, LISTEN_QUEUE,
        )
        address_family = select_address_family(hostname, port)
        server_address = get_sockaddr(hostname, port, address_family)
        s = socket.socket(address_family, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.set_inheritable(True)
        s.bind(server_address)
        s.listen(LISTEN_QUEUE)
        return s


logger = logging.getLogger(__name__)


class StopException(Exception):
    pass


def chunks(mm: mmap.mmap) -> Generator[Tuple[bytes, int], None, None]:
    """
    Yield messages from memory map, including the offset to the start of the
    message (more correctly: the length field of the message).
    """
    index = 0
    while index < len(mm):
        start_index = index
        len_field_bytes = mm[index:index+8]
        len_field = np.frombuffer(len_field_bytes, dtype=np.int64, count=1)
        index += 8
        data = mm[index:index+len_field[0]]
        yield data, start_index
        index += len(data)


def find_start_offset(mm):
    for chunk, index in chunks(mm):
        try:
            data = json.loads(chunk)
            if data['htype'] == "dheader-1.0":
                return index
        except Exception:
            pass


class ZMQReplay(ErrThreadMixin, threading.Thread):
    def __init__(
            self, uri, random_port, path, name,
            arm_event: EventClass,
            trigger_event: EventClass,
            data_filter=None,
            verbose=True,
            tolerate_timeouts=True,
            *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._uri = uri
        self._random_port = random_port
        self._port = None
        self._path = path
        self._name = name
        self._arm_event = arm_event
        self._trigger_event = trigger_event
        self.listen_event = threading.Event()
        if data_filter is None:
            def data_filter(data):
                return data
        self._data_filter = data_filter
        self._verbose = verbose
        self._tolerate_timeouts = tolerate_timeouts

    @property
    def port(self):
        return self._port

    def wait_for_listen(self, timeout=10):
        """
        To be called from the main thread
        """
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.listen_event.wait(timeout=0.1):
                return
            self.maybe_raise()
        if not self.listen_event.is_set():
            raise RuntimeError("failed to start in %f seconds" % timeout)

    def run(self):
        set_thread_name('ZMQReplay')
        headers = read_headers(self._path)

        def send_line(index, zmq_socket, mm, more: Optional[bool] = None):
            len_field_bytes = mm[index:index+8]
            len_field = np.frombuffer(len_field_bytes, dtype=np.int64, count=1)
            index += 8
            data = mm[index:index+len_field[0]]
            res = 0
            while not res:
                if self.is_stopped():
                    raise StopException("Server is stopped")
                res = zmq_socket.poll(100, flags=zmq.POLLOUT)
            # XXX quite bizarrely, for this kind of data stream, using
            # `send(..., copy=True)` is faster than `send(..., copy=False)`.
            # -> possibly because message sizes are on average quite small,
            # see the discussion here: http://aosabook.org/en/zeromq.html
            filtered = self._data_filter(data)
            flags = 0
            if more:
                flags |= zmq.SNDMORE
            if filtered is not None:
                zmq_socket.send(filtered, copy=True, flags=flags)
            index += len(data)
            return index

        def send_msg(msg, zmq_socket):
            res = 0
            while not res:
                if self.is_stopped():
                    raise StopException("Server is stopped")
                res = zmq_socket.poll(100, flags=zmq.POLLOUT)
            zmq_socket.send(msg, copy=True, flags=0)

        try:
            context = zmq.Context()
            zmq_socket = context.socket(zmq.PUSH)
            logger.debug("about to bind")
            if self._random_port:
                self._port = zmq_socket.bind_to_random_port(self._uri)
            else:
                sock = zmq_socket.bind(self._uri)
                self._port = urllib.parse.urlparse(sock.addr).port
            zmq_socket.set_hwm(18000)
            logger.debug("bound")
            with open(self._path, mode='rb') as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                self.listen_event.set()
                start_index = find_start_offset(mm)
                logger.info(
                    f"ZMQReplay {self._name} sending on {self._uri} port {self.port},"
                    f" start_index={start_index}"
                )
                try:
                    while True:
                        # offset into the file:
                        index = start_index
                        count = 0
                        # index of the frame that is currently being sent:
                        frame_index = 0
                        if headers[1]['trigger_mode'] in ('exte', 'inte'):
                            nimages = headers[1]['ntrigger']
                        elif headers[1]['trigger_mode'] in ('exts', 'ints'):
                            nimages = headers[1]['nimages']
                        else:
                            raise RuntimeError(f"Unknown trigger mode {headers[1]['trigger_mode']}")
                        logger.info(f"ZMQReplay: Waiting for arm; will send {nimages} frames")
                        while not self._arm_event.wait(timeout=0.1):
                            if self.is_stopped():
                                raise StopException("Server is stopped")
                        self._arm_event.clear()
                        logger.info("sending acquisition headers")
                        while count < 2 and not self.is_stopped() and index < len(mm):
                            index = send_line(index, zmq_socket, mm)
                            count += 1

                        # external trigger is always set for now
                        if headers[1]['trigger_mode'] in ('exte', 'exts'):
                            self._trigger_event.set()
                        else:
                            logger.info("waiting for trigger(s)")

                        sent_message = False
                        have_sent_footer = False
                        while not self.is_stopped() and index < len(mm):
                            while not self._trigger_event.wait(timeout=0.1):
                                if self.is_stopped():
                                    raise StopException("Server is stopped")
                            count = 0
                            if not sent_message:
                                logger.info("sending frame data")
                                sent_message = True
                            more = True
                            while count < 4 and not self.is_stopped() and index < len(mm):
                                if count == 3:
                                    more = False
                                if frame_index >= nimages:
                                    logger.info("'footer', no longer multi part...")
                                    more = False
                                    have_sent_footer = True
                                index = send_line(index, zmq_socket, mm, more)
                                count += 1
                            frame_index += 1
                            # Internal frame trigger: wait for next trigger
                            # in next loop iteration
                            if headers[1]['trigger_mode'] == 'inte':
                                self._trigger_event.clear()
                        logger.info("finished frame data")
                        # the file may be missing the footer message;
                        # we emulate it here:
                        if not have_sent_footer:
                            logger.info("did not see footer, emulating")
                            footer = {
                                "htype": "dseries_end-1.0",
                                "series": headers[0]['series'],
                            }
                            send_msg(json.dumps(footer).encode("utf8"), zmq_socket)

                        # Internal trigger: wait on next repetition
                        if headers[1]['trigger_mode'] == 'ints':
                            self._trigger_event.clear()
                finally:
                    mm.close()
        except StopException:
            pass
        except Exception as e:
            return self.error(e)
        finally:
            logger.info(f"{self._name} exiting")
            zmq_socket.close()


class RustedReplay(ZMQReplay):
    def __init__(
        self,
        *args,
        dwelltime: Optional[int] = None,
        tolerate_timeouts=True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.dwelltime = dwelltime
        self._tolerate_timeouts = tolerate_timeouts

    def run(self):
        try:
            import libertem_dectris
            set_thread_name("RustedReplay")
            _sim = libertem_dectris.DectrisSim(
                uri=self._uri,
                filename=self._path,
                dwelltime=self.dwelltime,
                random_port=self._random_port,
            )

            real_uri = _sim.get_uri()
            self._port = urllib.parse.urlparse(real_uri).port
            if self._verbose:
                logger.info(f"RustedReplay listening on {real_uri}")

            self.listen_event.set()
            while True:
                if self._verbose:
                    logger.info("RustedReplay: Waiting for arm")
                while not self._arm_event.wait(timeout=0.1):
                    if self.is_stopped():
                        raise StopException("Server is stopped")
                self._arm_event.clear()

                # Different trigger settings:
                #
                # INTS: record `nimages` after the first `trigger` command received,
                #       according to `frame_time` and `count_time`; repeated `ntrigger` times
                #
                #       => exposure time is not adjusted and just sent as it is written
                #       in the input file
                #
                # INTE: record one image per trigger, with exposure time taken
                #       from the `count_time` parameter of `trigger` command
                #
                #       => exposure time is not yet taken from the `trigger` command
                #
                # EXTS: acquire `nimages` after receiving a single `trigger` command,
                #       according to `frame_time` and `count_time`
                #
                #       => in the sim, we trigger as fast as possible at the beginning
                #       of the acquisiiton
                #
                # EXTE: acquire one image per trigger, exposing as long as the trigger
                #       signal is high.
                #
                #       => in the sim, we assume a constant dwell time, or trigger as
                #       fast as we can send the data
                try:
                    if self._verbose:
                        logger.info("sending acquisition headers")
                    _sim.send_headers()
                    if self._verbose:
                        logger.info("headers sent")
                    det_config = _sim.get_detector_config()
                    trigger_mode = det_config.get_trigger_mode()
                    if trigger_mode == libertem_dectris.TriggerMode.INTE:
                        # FIXME: check stop event from _sim in send_frames
                        if self._verbose:
                            logger.info("sending one frame per trigger")
                        for _ in range(det_config.ntrigger):
                            while not self._trigger_event.wait(timeout=0.1):
                                if self.is_stopped():
                                    raise StopException("Server is stopped")
                            self._trigger_event.clear()
                            _sim.send_frames(1)
                        _sim.send_footer()

                    elif trigger_mode in (
                        libertem_dectris.TriggerMode.EXTE,
                        libertem_dectris.TriggerMode.EXTS,
                    ):
                        # FIXME: check stop event from _sim in send_frames
                        if self._verbose:
                            logger.info("sending all frames")
                        _sim.send_frames()
                        _sim.send_footer()
                    elif trigger_mode in (
                        libertem_dectris.TriggerMode.INTS,
                    ):
                        for _ in range(det_config.get_num_frames()):
                            if trigger_mode == libertem_dectris.TriggerMode.INTS:
                                while not self._trigger_event.wait(timeout=0.1):
                                    if self.is_stopped():
                                        raise StopException("Server is stopped")
                                self._trigger_event.clear()
                            if self._verbose:
                                logger.info("sending next series")
                            # FIXME: check stop event from _sim in send_frames
                            _sim.send_frames(det_config.get_num_frames())
                        _sim.send_footer()

                except libertem_dectris.TimeoutError:
                    if not self._tolerate_timeouts:
                        raise
                    if self._verbose:
                        logger.info("Timeout, resetting")
                    self._trigger_event.clear()
        except StopException:
            pass


def read_headers(filename):
    """
    Read the header with type 'dheader-1.0'.

    Only the 'header_detail': 'basic' configuration is tested,
    but 'all' should work, too.

    The header is sent in two parts; first, one JSON-encoded message that looks like
    this:

        {'header_detail': 'basic', 'htype': 'dheader-1.0', 'series': 34}

    Then, another JSON-encoded message with the detector configuration. Depending
    on the 'header_detail' field above, this message can include more or less fields.

    This function also ignores superfluous 'dseries_end-1.0' headers that
    may appear at the beginning of a capture.
    """
    with open(filename, mode='rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        result = []
        try:
            index = 0
            header_type = None
            while header_type != 'dimage-1.0':
                len_field_bytes = mm[index:index+8]
                len_field = np.frombuffer(len_field_bytes, dtype=np.int64, count=1)
                index += 8
                data = mm[index:index+len_field[0]]
                index += len(data)
                data_decoded = json.loads(data)
                if 'htype' in data_decoded:
                    # we expect only the dheader-1.0 and the following detector
                    # configuration; but there may be other messages. skip the
                    # series end message from a previous series which might be
                    # at the beginning:
                    if data_decoded['htype'] == 'dseries_end-1.0':
                        continue
                    header_type = data_decoded['htype']
                result.append(data_decoded)
        finally:
            mm.close()
        return result


api = Blueprint('api', __name__)


@api.route("/detector/api/version/", methods=["GET"])
def get_version() -> Dict:
    return {
        'value': '1.8.0',
        'value_type': 'string'
    }


@api.route("/detector/api/1.8.0/command/<parameter>", methods=["PUT"])
def send_detector_command(parameter) -> Dict:
    if parameter == 'arm':
        current_app.config['arm_event'].set()
    elif parameter == 'disarm':
        current_app.config['arm_event'].clear()
    elif parameter == 'trigger':
        current_app.config['trigger_event'].set()
    if parameter in ('abort', 'arm', 'cancel', 'disarm'):
        return {
            'sequence id': current_app.config['headers'][0]['series']
        }
    elif parameter in ('hv_reset', 'initialize', 'trigger'):
        return {}
    else:
        return {"error": f'Parameter {parameter} not implemented.'}, 500


@api.route("/stream/api/1.8.0/config/<parameter>", methods=["GET"])
def get_stream_config(parameter) -> Dict:
    if parameter == 'mode':
        return {
            'access_mode': 'rw',
            'allowed_values': ['enabled', 'disabled'],
            'value': 'enabled',
            'value_type': 'string'
        }
    elif parameter == 'header_detail':
        return {
            'access_mode': 'rw',
            'allowed_values': ['all', 'basic', 'none'],
            'value': 'basic',
            'value_type': 'string'
        }
    else:
        return {"error": f'Parameter {parameter} not implemented.'}, 500


@api.route("/stream/api/1.8.0/config/<parameter>", methods=["PUT"])
def set_stream_config(parameter):
    request_data = request.json
    if parameter == 'mode':
        assert request_data['value'] in ('enabled', )
    elif parameter == 'header_detail':
        assert request_data['value'] in ('basic', )
    else:
        return f'Parameter {parameter} not implemented.', 500
    return ""


@api.route("/detector/api/1.8.0/config/<parameter>", methods=["GET"])
def get_detector_config(parameter) -> Dict:
    defaults = {
        'x_pixels_in_detector': {
            'access_mode': 'r',
            'value': 1028,
            'value_type': 'uint'
        },
        'y_pixels_in_detector': {
            'access_mode': 'r',
            'value': 512,
            'value_type': 'uint'
        },
        'bit_depth_image': {
            'access_mode': 'r',
            'value': 32,
            'value_type': 'uint'
        },
        'count_time': {
            'access_mode': 'rw',
            'max': 3600.0,
            'min': 5e-08,
            'unit': 's',
            'value': 0.001,
            'value_type': 'float'
        },
        'frame_time': {
            'access_mode': 'rw',
            'min': 0.018518519,
            'unit': 's',
            'value': 0.0010001,
            'value_type': 'float'
        },
        'nimages': {
            'access_mode': 'rw',
            'max': 2000000000,
            'min': 1,
            'value': 1,
            'value_type': 'uint'
        },
        'compression': {
            'access_mode': 'rw',
            'allowed_values': ['lz4', 'bslz4', 'none'],
            'value': 'bslz4',
            'value_type': 'string'
        },
        'trigger_mode': {
            'access_mode': 'rw',
            'allowed_values': ['exte', 'exts', 'inte', 'ints'],
            'value': 'exte',
            'value_type': 'string'
        },
        'ntrigger': {
            'access_mode': 'rw',
            'max': 2000000000,
            'min': 1,
            'value': 16384,
            'value_type': 'uint'
        }
    }
    if parameter in defaults:
        res = defaults[parameter]
        logger.info(current_app.config['headers'])
        if parameter not in current_app.config['headers'][1]:
            keys = list(current_app.config['headers'][1])
            return {"error": f'Parameter not found in header, only have: {keys}'}, 500
        res['value'] = current_app.config['headers'][1][parameter]
        return res
    else:
        return {"error": f'Parameter {parameter} not implemented.'}, 500


@api.route("/detector/api/1.8.0/config/<parameter>", methods=["PUT"])
def set_detector_config(parameter):
    request_data = request.json
    headers = current_app.config['headers']
    if parameter in headers[1]:
        if (request_data['value'] != headers[1][parameter]
                and (parameter not in {'ntrigger', 'nimages'}
                    or request_data['value'] != 1)):
            error = (
                f'Value {request_data["value"]} for parameter {parameter} '
                f'does not match file content {headers[1][parameter]}.'
            )
            return {"error": error}, 500
        else:
            return json.dumps([parameter])
    else:
        return {"error": f'Parameter {parameter} not implemented.'}, 500


def run_api(
    port, headers, arm_event, trigger_event,
    listen_event, port_value, detector_config=None,
):
    set_thread_name('dectris.sim:run_api')
    app = Flask("zmqreplay")
    app.config['headers'] = headers
    app.config['arm_event'] = arm_event
    app.config['trigger_event'] = trigger_event
    app.config['detector_config'] = detector_config
    app.register_blueprint(api)

    sock = prepare_socket('localhost', port)
    sockname = sock.getsockname()
    port_value.value = sockname[1]
    fd = sock.fileno()
    logger.info("API server listening on %s", sockname)
    server = make_server(host='localhost', port=port, app=app, fd=fd)
    listen_event.set()
    return server.serve_forever()


class DectrisSim:
    def __init__(
        self,
        path: str,
        port: int,
        zmqport: int,
        dwelltime: Optional[int] = None,
        data_filter=None,
        tolerate_timeouts: bool = True,
        verbose: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        path
            _description_
        port
            HTTP REST API port
        zmqport
            ZeroMQ port we should bind to
        dwelltime : int
            Take at least this much time for sending each frame, in microseconds
        verbose
            Print progress messages to stdout
        tolerate_timeouts
            If True, raise on timeouts instead of resetting simulator
        """
        headers = read_headers(path)

        arm_event = multiprocessing.Event()
        trigger_event = multiprocessing.Event()
        self.stop_event = multiprocessing.Event()
        self.api_listen_event = multiprocessing.Event()
        self.api_port = multiprocessing.Value('l', -1)
        self.dwelltime = dwelltime
        self.verbose = verbose

        if data_filter is None:
            self.zmq_replay = RustedReplay(
                uri=f"tcp://127.0.0.1:{zmqport}" if zmqport else "tcp://127.0.0.1",
                random_port=not bool(zmqport),
                path=path,
                name='zmq_replay',
                arm_event=arm_event,
                trigger_event=trigger_event,
                stop_event=self.stop_event,
                data_filter=data_filter,
                dwelltime=dwelltime,
                tolerate_timeouts=tolerate_timeouts,
                verbose=verbose,
            )
        else:
            self.zmq_replay = ZMQReplay(
                uri=f"tcp://127.0.0.1:{zmqport}" if zmqport else "tcp://127.0.0.1",
                random_port=not bool(zmqport),
                path=path,
                name='zmq_replay',
                arm_event=arm_event,
                trigger_event=trigger_event,
                stop_event=self.stop_event,
                data_filter=data_filter,
                tolerate_timeouts=tolerate_timeouts,
                verbose=verbose,
            )

        self.zmq_replay.daemon = True

        self.api_server = multiprocessing.Process(
            target=run_api,
            kwargs={
                'port': port,
                'headers': headers,
                'arm_event': arm_event,
                'trigger_event': trigger_event,
                'listen_event': self.api_listen_event,
                'port_value': self.api_port,
            },
            daemon=True,
        )

    @property
    def port(self):
        return self.api_port.value

    @property
    def zmqport(self):
        return self.zmq_replay.port

    def start(self):
        self.api_server.start()
        self.zmq_replay.start()

    def wait_for_listen(self):
        self.zmq_replay.wait_for_listen()
        if not self.api_listen_event.wait(timeout=10):
            raise RuntimeError("failed to start in %f seconds" % 10)

    def is_alive(self):
        return self.zmq_replay.is_alive() and self.api_server.is_alive()

    def maybe_raise(self):
        self.zmq_replay.maybe_raise()
        exitcode = self.api_server.exitcode
        # -15 seems to be the return code when the process is terminated
        if exitcode and self.api_server.exitcode != -15:
            raise RuntimeError(f'API server exit code {self.api_server.exitcode}')

    def stop(self):
        if self.verbose:
            logger.info("Stopping...")
        self.stop_event.set()
        self.api_server.terminate()
        timeout = 32
        start = time.time()
        while True:
            self.maybe_raise()
            if (
                    (not self.zmq_replay.is_alive())
                    and (not self.api_server.is_alive())
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
@click.option('--port', type=int, default=8910)
@click.option('--zmqport', type=int, default=9999)
@click.option('--dwelltime', type=int, default=None)
@click.option('--verbose', type=bool, default=True)
def main(path, port, zmqport, dwelltime, verbose):
    logging.basicConfig(level=logging.INFO)
    dectris_sim = DectrisSim(
        path=path, port=port, zmqport=zmqport, dwelltime=dwelltime, verbose=verbose,
    )
    dectris_sim.start()

    def handler_term(signum, frame):
        dectris_sim.stop()

    signal.signal(signal.SIGTERM, handler_term)

    dectris_sim.wait_for_listen()
    # This allows us to handle Ctrl-C, and the main program
    # stops in a timely fashion when continuous scanning stops.
    try:
        while dectris_sim.is_alive():
            dectris_sim.maybe_raise()
            time.sleep(1)
    except KeyboardInterrupt:
        # Just to not print "Aborted!"" from click
        sys.exit(0)
    finally:
        print("Stopping...")
        try:
            dectris_sim.stop()
        except UndeadException:
            print("Killing server threads")
            # Since the threads are daemon threads, they will die abruptly
            # when this main thread finishes.
            return


if __name__ == '__main__':
    main()
