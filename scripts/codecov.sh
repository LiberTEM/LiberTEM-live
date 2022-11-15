#!/bin/bash

set -e

# pin the version so we can compare sha
curl -Os https://uploader.codecov.io/v0.3.2/linux/codecov
echo '20f9c9d78483fce977b6cc39e231a734a23bcd36f4d536bb7355222fb88d02bc codecov' | sha256sum -c

chmod +x codecov
./codecov $*
