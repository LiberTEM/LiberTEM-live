# This example shows how to process live data from a
# Quantum Detectors Merlin camera within Gatan Digital Micrograph
# using their Python scripting.

# If you want to use this with the simulated data source, run a simple Merlin
# simulator in the background that replays an MIB dataset:

# libertem-live-mib-sim ~/Data/default.hdr --cached=MEM --wait-trigger

# The --wait-trigger option is important for this notebook to function correctly
# since that allows to drain the data socket before an acquisition like it is
# necessary for a real-world Merlin detector.

# A suitable dataset is available at https://zenodo.org/record/5113449

import sys
import os
import multiprocessing
import threading
import concurrent.futures
import time

from libertem_live import api
from libertem_live.udf.monitor import SignalMonitorUDF
from libertem_live.detectors.merlin.control import MerlinControl

from libertem.viz.gms import GMSLive2DPlot
# Sum all detector frames, result is a map of the detector
from libertem.udf.sum import SumUDF
# Sum up each detector frame, result is a bright field STEM image of the scan area
from libertem.udf.sumsigudf import SumSigUDF


# Adjust to match experiment
MERLIN_DATA_SOCKET = ('127.0.0.1', 6342)
MERLIN_CONTROL_SOCKET = ('127.0.0.1', 6341)
NAV_SHAPE = (128, 128)

# Change to a writable folder. GMS may run in C:\Windows\system32
# depending on the starting method.
os.chdir(os.environ['USERPROFILE'])

multiprocessing.set_executable(os.path.join(sys.exec_prefix, 'pythonw.exe'))


def merlin_setup(c: MerlinControl, dwell_time=1e-3, depth=12, save_path=None):
    print("Setting Merlin acquisition parameters")
    # Here go commands to control the camera and the rest of the setup
    # to perform an acquisition.

    # The Merlin simulator currently accepts all kinds of commands
    # and doesn't respond like a real Merlin detector.
    c.set('CONTINUOUSRW', 1)
    c.set('ACQUISITIONTIME', dwell_time * 1e3)  # Time in miliseconds
    c.set('COUNTERDEPTH', depth)
    c.set('TRIGGERSTART', 5)
    c.set('RUNHEADLESS', 1)
    c.set('FILEFORMAT', 2)  # 0 binary, 2 raw binary

    if save_path is not None:
        c.set('IMAGESPERFILE', 256)
        c.set('FILEENABLE', 1)
        # raw format with timestamping is buggy, we need to do it ourselves
        c.set('USETIMESTAMPING', 0)
        c.set('FILEDIRECTORY', save_path)
    else:
        c.set('FILEENABLE', 0)

    print("Finished Merlin setup.")


def microscope_setup(dwell_time=1e-3):
    # Here go instructions to set dwell time and
    # other scan parameters
    # microscope.set_dwell_time(dwell_time)
    pass


def arm(c: MerlinControl):
    print("Arming Merlin...")
    c.cmd('STARTACQUISITION')
    print("Merlin ready for trigger.")


def set_nav(c: MerlinControl, aq):
    height, width = aq.shape.nav
    print("Setting resolution...")
    c.set('NUMFRAMESTOACQUIRE', height * width)
    # Only one trigger for the whole scan with SOFTTRIGGER
    # This has to be adapted to the real trigger setup.
    # Set to `width` for line trigger and to `1` for pixel trigger.
    c.set('NUMFRAMESPERTRIGGER', height * width)

    # microscope.configure_scan(shape=aq.shape.nav)


class AcquisitionState:
    def __init__(self):
        self.trigger_result = None

    def set_trigger_result(self, result):
        self.trigger_result = result


# The workload is wrapped into a `main()` function
# to run it in a separate background thread since using Numba
# can hang when used directly in a GMS Python background thread
def main():
    acquisition_state = AcquisitionState()
    pool = concurrent.futures.ThreadPoolExecutor(1)

    # This uses the above variables as a closure
    class MyHooks(api.Hooks):
        def on_ready_for_data(self, env):
            print("Triggering!")
            # microscope.start_scanning()

            time.sleep(1)
            height, width = env.aq.shape.nav

            # Real-world example: Function call to trigger the scan engine
            # do_scan = lambda: ceos.call.acquireScan(
            #    width=width,
            #    height=height+1,
            #    imageName="test"
            # )

            # Testing: Use soft trigger
            def do_scan():
                '''
                Emulated blocking scan function using the Merlin simulator.

                This function doesn't actually block, but it could!
                '''
                print("do_scan()")
                with c:
                    c.cmd('SOFTTRIGGER')

            fut = pool.submit(do_scan)
            acquisition_state.set_trigger_result(fut)

    with api.LiveContext() as ctx:
        with ctx.make_connection('merlin').open(
            api_host=MERLIN_CONTROL_SOCKET[0],
            api_port=MERLIN_CONTROL_SOCKET[1],
            data_host=MERLIN_DATA_SOCKET[0],
            data_port=MERLIN_DATA_SOCKET[1],
        ) as conn:
            aq = ctx.make_acquisition(
                conn=conn,
                hooks=MyHooks(),
                nav_shape=NAV_SHAPE,
                frames_per_partition=800,
            )

            udfs = [SumUDF(), SumSigUDF(), SignalMonitorUDF()]

            plots = [GMSLive2DPlot(aq, udf) for udf in udfs]
            for plot in plots:
                plot.display()

            c = MerlinControl(*MERLIN_CONTROL_SOCKET)

            print("Connecting Merlin control...")
            with c:
                merlin_setup(c)
                microscope_setup()

                set_nav(c, aq)
                arm(c)
            try:
                ctx.run_udf(dataset=aq, udf=udfs, plots=plots)
            finally:
                try:
                    if acquisition_state.trigger_result is not None:
                        print("Waiting for blocking scan function...")
                        print(f"result = {acquisition_state.trigger_result.result()}")
                finally:
                    pass  # microscope.stop_scanning()
            print("Finished.")


if __name__ == "__main__":
    # Start the workload and wait for it to finish
    th = threading.Thread(target=main)
    th.start()
    th.join()
