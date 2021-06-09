#!/bin/bash
# run both nbval and libertem-live-mib-sim

.tox/notebooks/bin/libertem-live-mib-sim --host 127.0.0.1 "$TESTDATA_BASE_PATH/20200518 165148/default.hdr" --cached=MEMFD --wait-trigger&
SIM_PID=$!

cleanup() {
    kill "$SIM_PID"
}

trap cleanup EXIT

echo "started simulator in background, pid=$SIM_PID"
echo "waiting for data socket to be ready..."

while ! (echo "" > /dev/tcp/127.0.0.1/6342) 2>/dev/null; do echo "Waiting..."; sleep 1; done

echo "done waiting, data socket ready!"

"$@"
