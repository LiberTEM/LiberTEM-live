# This example shows how to process live data from a
# Quantum Detectors Merlin camera within Gatan Digital Micrograph
# using their Python scripting.

import sys
import os
import multiprocessing
import threading
from contextlib import contextmanager

from libertem_live import api
from libertem_live.udf.monitor import SignalMonitorUDF

from libertem.viz.gms import GMSLive2DPlot
# Sum all detector frames, result is a map of the detector
from libertem.udf.sum import SumUDF
# Sum up each detector frame, result is a bright field STEM image of the scan area
from libertem.udf.sumsigudf import SumSigUDF


# Adjust to match experiment
MERLIN_DATA_SOCKET = ('192.168.116.1', 6342)
SCAN_SIZE = (128, 128)

# Change to a writable folder. GMS may run in C:\Windows\system32
# depending on the starting method.
os.chdir(os.environ['USERPROFILE'])

multiprocessing.set_executable(os.path.join(sys.exec_prefix, 'pythonw.exe'))

@contextmanager
def medipix_setup(dataset, udfs):
    print("priming camera for acquisition")
    # TODO: medipix control socket commands go here
    # TODO interface to be tested, not supported in simulator yet

    # dataset.control.set('numframes', np.prod(SCAN_SIZE, dtype=np.int64))
    # dataset.control.set(...)

    # microscope.configure_scan()
    # microscope.start_scanning()
    print("running acquisition")
    with dataset.start_acquisition():
        yield
    print("camera teardown")
    # teardown routines go here

# The workload is wrapped into a `main()` function
# to run it in a separate background thread since using Numba
# can hang when used directly in a GMS Python background thread
def main():
    with api.LiveContext() as ctx:
        ds = ctx.prepare_acquisition(
            'merlin',
            medipix_setup,
            scan_size=SCAN_SIZE,
            host=MERLIN_DATA_SOCKET[0],
            port=MERLIN_DATA_SOCKET[1],
            control_port=None,  # deactivate control interface, not supported in simulator yet
            frames_per_partition=800,
            pool_size=2,
        )
        
        udfs = [SumUDF(), SumSigUDF(), SignalMonitorUDF()]

        plots = [GMSLive2DPlot(ds, udf) for udf in udfs]
        for plot in plots:
            plot.display()
        
        ctx.run_udf(dataset=ds, udf=udfs, plots=plots)

if __name__ == "__main__":
    # Start the workload and wait for it to finish
    th = threading.Thread(target=main)
    th.start()
    th.join()
