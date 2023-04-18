import os
from contextlib import contextmanager
from typing import Generator
import logging

import pytest
from libertem.common.backend import get_use_cpu, get_use_cuda, set_use_cpu, set_use_cuda
from libertem.utils.devices import detect

from libertem_live.detectors.common import UndeadException

logger = logging.getLogger(__name__)


def get_testdata_path():
    return os.environ.get(
        'TESTDATA_BASE_PATH',
        os.path.normpath(
            os.path.join(os.path.dirname(__file__), '..', 'data')
        )
    )


def run_camera_sim(*args, cls, **kwargs) -> Generator:
    server = cls(
        *args, **kwargs
    )
    server.start()
    server.wait_for_listen()
    logger.info(f"{cls.__name__}: camera sim started")
    yield server
    logger.info(f"{cls.__name__}: cleaning up server thread")
    server.maybe_raise()
    logger.info(f"{cls.__name__}: stopping server thread")
    try:
        server.stop()
    except UndeadException:
        raise RuntimeError("Server didn't stop gracefully")


@contextmanager
def set_device_class(device_class):
    '''
    This context manager is designed to work with the inline executor.
    It simplifies running tests with several device classes by skipping
    unavailable device classes and handling setting and re-setting the environment variables
    correctly.
    '''
    prev_cuda_id = get_use_cuda()
    prev_cpu_id = get_use_cpu()
    try:
        if device_class in ('cupy', 'cuda'):
            d = detect()
            cudas = d['cudas']
            if not d['cudas']:
                pytest.skip(f"No CUDA device, skipping test with device class {device_class}.")
            if device_class == 'cupy' and not d['has_cupy']:
                pytest.skip(f"No CuPy, skipping test with device class {device_class}.")
            set_use_cuda(cudas[0])
        else:
            set_use_cpu(0)
        logger.info(f'running with {device_class}')
        yield
    finally:
        if prev_cpu_id is not None:
            assert prev_cuda_id is None
            set_use_cpu(prev_cpu_id)
        elif prev_cuda_id is not None:
            assert prev_cpu_id is None
            set_use_cuda(prev_cuda_id)
        else:
            raise RuntimeError('No previous device ID, this should not happen.')
