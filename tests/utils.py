import os


def get_testdata_path():
    return os.environ.get(
        'TESTDATA_BASE_PATH',
        os.path.normpath(
            os.path.join(os.path.dirname(__file__), '..', 'data')
        )
    )
