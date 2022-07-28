import sys
import time
import threading
import multiprocessing
import json
import mmap
from typing import Dict
import urllib

import zmq
import click
import numpy as np
from flask import Flask, request, Blueprint, current_app
from werkzeug.serving import prepare_socket, make_server

from libertem_live.detectors.common import ErrThreadMixin, UndeadException


class StopException(Exception):
    pass


class ZMQReplay(ErrThreadMixin, threading.Thread):
    def __init__(
            self, uri, random_port, path, name,
            arm_event: threading.Event, trigger_event: threading.Event,
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
        headers = read_headers(self._path)

        def send_line(index, zmq_socket, mm):
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
            zmq_socket.send(data, copy=True)
            index += len(data)
            return index

        try:
            context = zmq.Context()
            zmq_socket = context.socket(zmq.PUSH)
            print("about to bind")
            if self._random_port:
                self._port = zmq_socket.bind_to_random_port(self._uri)
            else:
                sock = zmq_socket.bind(self._uri)
                self._port = urllib.parse.urlparse(sock.addr).port
            print("bound")
            with open(self._path, mode='rb') as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                self.listen_event.set()
                print(f"ZMQReplay {self._name} sending on {self._uri} port {self.port}")
                try:
                    while True:
                        index = 0
                        count = 0
                        print("Waiting for arm")
                        while not self._arm_event.wait(timeout=0.1):
                            if self.is_stopped():
                                raise StopException("Server is stopped")
                        self._arm_event.clear()
                        print("sending acquisition headers")
                        while count < 2 and not self.is_stopped() and index < len(mm):
                            index = send_line(index, zmq_socket, mm)
                            count += 1

                        # external trigger is always set for now
                        if headers[1]['trigger_mode'] in ('exte', 'exts'):
                            self._trigger_event.set()
                        else:
                            print("waiting for trigger(s)")

                        sent_message = False
                        while not self.is_stopped() and index < len(mm):
                            while not self._trigger_event.wait(timeout=0.1):
                                if self.is_stopped():
                                    raise StopException("Server is stopped")
                            count = 0
                            if not sent_message:
                                print("sending frame data")
                                sent_message = True
                            while count < 4 and not self.is_stopped() and index < len(mm):
                                index = send_line(index, zmq_socket, mm)
                                count += 1
                            # Internal frame trigger: wait for next trigger
                            # in next loop iteration
                            if headers[1]['trigger_mode'] == 'inte':
                                self._trigger_event.clear()
                        print("finished frame data")
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
            print(f"{self._name} exiting")
            zmq_socket.close()


def read_headers(filename):
    with open(filename, mode='rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        result = []
        try:
            index = 0
            for i in range(2):
                len_field_bytes = mm[index:index+8]
                len_field = np.frombuffer(len_field_bytes, dtype=np.int64, count=1)
                index += 8
                data = mm[index:index+len_field[0]]
                result.append(json.loads(data))
                index += len(data)
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
        raise ValueError(f'Parameter {parameter} not implemented.')


@api.route("/stream/api/1.8.0/config/<parameter>", methods=["GET"])
def get_stream_config(parameter) -> Dict:
    if parameter == 'mode':
        return {
            'access_mode': 'rw',
            'allowed_values': ['enabled', 'disabled'],
            'value': 'enabled',
            'value_type': 'string'
        }
    else:
        raise ValueError(f'Parameter {parameter} not implemented.')


@api.route("/stream/api/1.8.0/config/<parameter>", methods=["PUT"])
def set_stream_config(parameter):
    request_data = request.json
    if parameter == 'mode':
        assert request_data['value'] in ('enabled', )
    else:
        raise ValueError(f'Parameter {parameter} not implemented.')
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
        res['value'] = current_app.config['headers'][1][parameter]
        return res
    else:
        raise ValueError(f'Parameter {parameter} not implemented.')


@api.route("/detector/api/1.8.0/config/<parameter>", methods=["PUT"])
def set_detector_config(parameter):
    request_data = request.json
    headers = current_app.config['headers']
    if parameter in headers[1]:
        if (request_data['value'] != headers[1][parameter]
                and (parameter not in {'ntrigger', 'nimages'}
                    or request_data['value'] != 1)):
            raise ValueError(
                f'Value {request_data["value"]} for parameter {parameter} '
                f'does not match file content {headers[1][parameter]}.'
            )
        else:
            return json.dumps([parameter])
    else:
        raise ValueError(f'Parameter {parameter} not implemented.')


def run_api(port, headers, arm_event, trigger_event, listen_event, port_value):
    app = Flask("zmqreplay")
    app.config['headers'] = headers
    app.config['arm_event'] = arm_event
    app.config['trigger_event'] = trigger_event
    app.register_blueprint(api)

    sock = prepare_socket('localhost', port)
    sockname = sock.getsockname()
    port_value.value = sockname[1]
    fd = sock.fileno()
    print("API server listening on", sockname)
    server = make_server(host='localhost', port=port, app=app, fd=fd)
    listen_event.set()
    return server.serve_forever()


class DectrisSim:
    def __init__(self, path, port, zmqport) -> None:
        headers = read_headers(path)

        arm_event = multiprocessing.Event()
        trigger_event = multiprocessing.Event()
        self.stop_event = multiprocessing.Event()
        self.api_listen_event = multiprocessing.Event()
        self.api_port = multiprocessing.Value('l', -1)

        self.zmq_replay = ZMQReplay(
            uri=f"tcp://127.0.0.1:{zmqport}" if zmqport else "tcp://127.0.0.1",
            random_port=not bool(zmqport),
            path=path,
            name='zmq_replay',
            arm_event=arm_event,
            trigger_event=trigger_event,
            stop_event=self.stop_event,
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
                'port_value': self.api_port
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
        print("Stopping...")
        self.stop_event.set()
        self.api_server.terminate()
        timeout = 2
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


@click.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('--port', type=int, default=8910)
@click.option('--zmqport', type=int, default=9999)
def main(path, port, zmqport):
    dectris_sim = DectrisSim(path=path, port=port, zmqport=zmqport)
    dectris_sim.start()
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