import time
import threading
import json
import mmap
from typing import Dict

import zmq
import click
import numpy as np
from flask import Flask, request
from werkzeug.serving import run_simple

from libertem_live.detectors.common import ErrThreadMixin


class StopException(Exception):
    pass


class ZMQReplay(ErrThreadMixin, threading.Thread):
    def __init__(
            self, uri, path, name,
            arm_event: threading.Event, trigger_event: threading.Event,
            *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._uri = uri
        self._path = path
        self._name = name
        self._arm_event = arm_event
        self._trigger_event = trigger_event
        self.listen_event = threading.Event()

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
            zmq_socket.send(data)
            index += len(data)
            return index

        try:
            context = zmq.Context()
            zmq_socket = context.socket(zmq.PUSH)
            print("about to bind")
            zmq_socket.bind(self._uri)
            print("bound")
            with open(self._path, mode='rb') as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                self.listen_event.set()
                print(f"ZMQReplay {self._name} sending on {self._uri}")
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


@click.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('--port', type=int, default=8910)
@click.option('--zmqport', type=int, default=9999)
def main(path, port, zmqport):
    headers = read_headers(path)
    app = Flask("zmqreplay")

    arm_event = threading.Event()
    trigger_event = threading.Event()

    zmq_replay = ZMQReplay(
        uri=f"tcp://127.0.0.1:{zmqport}",
        path=path,
        name='zmq_replay',
        arm_event=arm_event,
        trigger_event=trigger_event
    )

    zmq_replay.daemon = True
    zmq_replay.start()
    zmq_replay.wait_for_listen()

    @app.route("/detector/api/version/", methods=["GET"])
    def get_version() -> Dict:
        return {
            'value': '1.8.0',
            'value_type': 'string'
        }

    @app.route("/detector/api/1.8.0/command/<parameter>", methods=["PUT"])
    def send_detector_command(parameter) -> Dict:
        if parameter == 'arm':
            arm_event.set()
        elif parameter == 'disarm':
            arm_event.clear()
        elif parameter == 'trigger':
            trigger_event.set()
        if parameter in ('abort', 'arm', 'cancel', 'disarm'):
            return {
                'sequence id': headers[0]['series']
            }
        elif parameter in ('hv_reset', 'initialize', 'trigger'):
            return {}
        else:
            raise ValueError(f'Parameter {parameter} not implemented.')

    @app.route("/stream/api/1.8.0/config/<parameter>", methods=["GET"])
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

    @app.route("/stream/api/1.8.0/config/<parameter>", methods=["PUT"])
    def set_stream_config(parameter):
        request_data = request.json
        if parameter == 'mode':
            assert request_data['value'] in ('enabled', )
        else:
            raise ValueError(f'Parameter {parameter} not implemented.')
        return ""

    @app.route("/detector/api/1.8.0/config/<parameter>", methods=["GET"])
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
            res['value'] = headers[1][parameter]
            return res
        else:
            raise ValueError(f'Parameter {parameter} not implemented.')

    @app.route("/detector/api/1.8.0/config/<parameter>", methods=["PUT"])
    def set_detector_config(parameter):
        request_data = request.json
        if parameter in headers[1]:
            if request_data['value'] != headers[1][parameter]:
                raise ValueError(
                    f'Value {request_data["value"]} for parameter {parameter} '
                    f'does not match file content {headers[1][parameter]}.'
                )
            else:
                return json.dumps([parameter])
        else:
            raise ValueError(f'Parameter {parameter} not implemented.')
    try:
        run_simple('localhost', port=port, application=app, threaded=True)

    finally:
        print("Stopping...")
        zmq_replay.stop()
        timeout = 2
        start = time.time()
        while True:
            zmq_replay.maybe_raise()
            if not zmq_replay.is_alive():
                break

            if (time.time() - start) >= timeout:
                print("Killing server threads")
                # Since the threads are daemon threads, they will die abruptly
                # when this main thread finishes.
                break

            time.sleep(0.1)


if __name__ == '__main__':
    main()
