import os
from libertem_live.detectors.common import UndeadException


def get_testdata_path():
    return os.environ.get(
        'TESTDATA_BASE_PATH',
        os.path.normpath(
            os.path.join(os.path.dirname(__file__), '..', 'data')
        )
    )


def run_camera_sim(*args, cls, **kwargs):
    server = cls(
        *args, **kwargs
    )
    server.start()
    server.wait_for_listen()
    print("camera sim started")
    yield server
    print("cleaning up server thread")
    server.maybe_raise()
    print("stopping server thread")
    try:
        server.stop()
    except UndeadException:
        raise RuntimeError("Server didn't stop gracefully")
