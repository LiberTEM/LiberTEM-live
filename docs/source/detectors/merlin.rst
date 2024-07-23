.. _`merlin detector`:

Quantum Detectors Merlin
========================

LiberTEM-live supports the Merlin Medipix detectors using the TCP control and
data interface.

Supported are currently 1 bit, 6 bit and 12 bit :code:`COUNTERDEPTH` for both the "binary"
(:code:`FILEFORMAT 0`) and the "raw binary" format (:code:`FILEFORMAT 2`).

For testing, an acquisition with soft trigger (:code:`TRIGGERSTART 5`) is
recommended since internal trigger (:code:`TRIGGERSTART 0`) may cause issues
with finding the beginning of the data stream. For a real STEM acquisition a
hardware trigger setup that matches the given instrument is required. See the
MerlinEM User Manual from Quantum Detectors for details!

Hardware and Software Requirements
----------------------------------

You can run LiberTEM-live on the PC shipped with the Merlin detector, or you can
use an external computer connected using a fast network connection (10Gbit+).
Especially in a quad setup, using an external computer is recommended to keep up
with the data rates at the fastest frame rates.

Usage examples
--------------

Here we briefly show how to connect to the Merlin detector system,
and how to run a LiberTEM UDF on the data stream.

.. testsetup::
    :skipif: not HAVE_MERLIN_TESTDATA

    from libertem_live.detectors.merlin.sim import CameraSim
    from libertem_live.api import LiveContext

    ctx = LiveContext()

    server = CameraSim(
        path=MERLIN_TESTDATA_PATH,
        data_port=0,
        control_port=0,
        trigger_port=0,
        nav_shape=(32, 32),
    )
    server.start()
    server.wait_for_listen()
    MERLIN_API_PORT = server.control_t.sockname[1]
    MERLIN_DATA_PORT = server.server_t.sockname[1]

    class MockMic:
        def trigger_scan(self, width, height, dwelltime):
            pass

    microscope = MockMic()


.. testcode::
    :skipif: not HAVE_MERLIN_TESTDATA

    from libertem.viz.bqp import BQLive2DPlot
    from libertem.udf.sum import SumUDF
    from libertem_live.api import LiveContext, Hooks

    class MyHooks(Hooks):
        def on_ready_for_data(self, env):
            """
            You can trigger the scan here, if you have a microscope control API
            """
            print("Triggering!")
            height, width = env.aq.shape.nav
            microscope.trigger_scan(width, height, dwelltime=10e-6)

    ctx = LiveContext(plot_class=BQLive2DPlot)

    # make a connection to the detector system:
    conn = ctx.make_connection('merlin').open(
        api_host="127.0.0.1",
        api_port=MERLIN_API_PORT,
        data_host="127.0.0.1",
        data_port=MERLIN_DATA_PORT,
    )

    # prepare for acquisition, setting up scan parameters etc.
    aq = ctx.make_acquisition(
        conn=conn,
        nav_shape=(32, 32),
        hooks=MyHooks(),
        frames_per_partition=512,
    )

    # run one or more UDFs on the live data stream:
    ctx.run_udf(dataset=aq, udf=SumUDF())

.. testoutput::

    Triggering!

Simulator
---------

A simple simulator that replays MIB datasets is included in LiberTEM-live.

Note that it is not a complete simulation of the Merlin software and
detector, but rather only meant for integration testing and development without
access to a real detector.

The simulator accepts the following parameters:

.. code-block:: shell

    (libertem) $ libertem-live-mib-sim --help
    Usage: libertem-live-mib-sim [OPTIONS] PATH

      Minimal Merlin simulator. Point PATH at a .hdr file, and that mib dataset
      will be replayed over the data socket.

    Options:
      --nav-shape <INTEGER INTEGER>...
      --continuous                    In untriggered mode, send a single
                                      continuous acquisition by repeating the
                                      input file indefinitely (or `max_runs`
                                      times)
      --cached [NONE|MEM|MEMFD]
      --host TEXT                     Address to listen on (data, control, and
                                      trigger sockets)
      --data-port INTEGER
      --control-port INTEGER
      --trigger-port INTEGER
      --wait-trigger                  Wait for a SOFTTRIGGER command on the
                                      control port, or a trigger signal on the
                                      trigger socket
      --manual-trigger                Wait for a manual trigger by user input
                                      after ARM
      --garbage                       Send garbage before trigger. Implies --wait-
                                      trigger
      --max-runs INTEGER              Maximum number of runs through the input
                                      file in continuous mode
      --help                          Show this message and exit.


A suitable MIB dataset for testing can be downloaded at
https://zenodo.org/record/5113449.

See the :ref:`Merlin reference section <merlin reference>` for a description of
the acquisition parameters.

.. testcleanup::
    :skipif: not HAVE_MERLIN_TESTDATA

    ctx.close()
