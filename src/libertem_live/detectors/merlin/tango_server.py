from tango.server import Device, DeviceProxy
from tango.server import command, device_property

from .control import MerlinControl


class Merlin(Device):
    host = device_property(dtype=str, mandatory=False)
    port = device_property(dtype=int, mandatory=False)

    def init_device(self):
        super().init_device()
        self._control = MerlinControl(host=self.host, port=self.port)

    @command
    def connect(self):
        self._control.connect()

    @command
    def close(self):
        self._control.close()

    @command(dtype_in=str, dtype_out=str)
    def get(self, param):
        with self._control:
            return self._control.get(param)

    @command(dtype_in=(str, ))
    def set(self, param_value):
        param, value = param_value
        with self._control:
            return self._control.set(param, value)

    @command(dtype_in=str)
    def cmd(self, cmd):
        with self._control:
            return self._control.cmd(cmd)

    @command(dtype_out=str)
    def hello(self):
        return "world"


class ProtectedDeviceProxy(DeviceProxy):
    '''
    Allow connecting and disconnecting with a :code:`with` statement.
    '''

    def __enter__(self):
        self.connect()

    def __exit__(self, type, value, traceback):
        self.close()


if __name__ == "__main__":
    Merlin.run_server()
