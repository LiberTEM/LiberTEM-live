#!/bin/bash
# run both nbval and libertem-live-mib-sim
# assumes default ports are usable (i.e. isolated environment)

set -eu

.tox/notebooks/bin/libertem-live-mib-sim --host 127.0.0.1 "$TESTDATA_BASE_PATH/20200518 165148/default.hdr" --cached=MEMFD --wait-trigger&
MERLIN_SIM_PID=$!

.tox/notebooks/bin/libertem-live-dectris-sim "$TESTDATA_BASE_PATH/dectris/zmqdump.dat.128x128-id34-exte-bslz4" --port 8910 --zmqport 9999&
DECTRIS_SIM_PID=$!

cleanup() {
    kill "$MERLIN_SIM_PID" || true
    kill "$DECTRIS_SIM_PID" || true
    sleep 5
    kill -9 "$MERLIN_SIM_PID" || true
    kill -9 "$DECTRIS_SIM_PID" || true
}

trap cleanup EXIT

echo "started merlin simulator in background, pid=$MERLIN_SIM_PID"
echo "started dectris simulator in background, pid=$DECTRIS_SIM_PID"
echo "waiting for data sockets to be ready..."

while ! (echo "" > /dev/tcp/127.0.0.1/6342) 2>/dev/null; do echo "Waiting for 6342..."; sleep 1; done
while ! (echo "" > /dev/tcp/127.0.0.1/8910) 2>/dev/null; do echo "Waiting for 8910..."; sleep 1; done
while ! (echo "" > /dev/tcp/127.0.0.1/9999) 2>/dev/null; do echo "Waiting for 9999..."; sleep 1; done

echo "done waiting, sockets ready!"

"$@"
