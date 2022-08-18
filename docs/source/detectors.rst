Supported Detectors
===================

.. _`dectris detectors`:

DECTRIS EIGER2-based
--------------------

.. versionadded:: 0.2

LiberTEM-live has support for all DECTRIS detectors that support
the `SIMPLON API <https://media.dectris.com/210607-DECTRIS-SIMPLON-API-Manual_EIGER2-chip-based_detectros.pdf>`_,
including QUADRO and ARINA.

.. note::
    Python 3.7+ is required to use the DECTRIS-related features of
    LiberTEM-live, and we recommend to use Python 3.10. Currently, Windows and
    Linux on x86_64 are supported.

Installation
............

There are a few extra dependencies needed for DECTRIS support. You can easily
install them using:

.. code-block:: shell

    (libertem) $ python -m pip install "libertem-live[dectris]"

Currently, at least LiberTEM 0.10 is required, and using LiberTEM master is
recommended for improved stability.

Usage example
.............

.. testsetup::
    :skipif: not HAVE_DECTRIS_TESTDATA

    from libertem_live.detectors.dectris.sim import DectrisSim
    server = DectrisSim(path=DECTRIS_TESTDATA_PATH, port=0, zmqport=0, verbose=False)
    server.start()
    server.wait_for_listen()
    DCU_API_PORT = server.port
    DCU_DATA_PORT = server.zmqport

.. testoutput::
    :hide:

    RustedReplay listening on tcp://127.0.0.1:...
    Waiting for arm

.. testcode::
    :skipif: not HAVE_DECTRIS_TESTDATA

    from libertem.executor.pipelined import PipelinedExecutor
    from libertem.viz.bqp import BQLive2DPlot
    from libertem_live.api import LiveContext
    from libertem.udf.sum import SumUDF

    executor = PipelinedExecutor(spec=PipelinedExecutor.make_spec(cpus=range(10), cudas=[]))
    ctx = LiveContext(executor=executor, plot_class=BQLive2DPlot)

    def trigger(aq):
        """
        You can trigger the scan here, if you have a microscope control API
        """
        # you can imagine something like:
        # trigger_scan(shape=tuple(aq.shape.nav), dwelltime=67e-6)
        pass

    # prepare for acquisition, setting up scan parameters etc.
    aq = ctx.prepare_acquisition(
        'dectris',
        api_host="127.0.0.1",
        api_port=DCU_API_PORT,
        data_host="127.0.0.1",
        data_port=DCU_DATA_PORT,
        nav_shape=(128, 128),
        trigger_mode="exte",
        trigger=trigger,
        frames_per_partition=512,
    )

    # run one or more UDFs on the live data stream:
    # (this can be run multiple times on the same `aq` object)
    ctx.run_udf(dataset=aq, udf=SumUDF(), plots=True)

.. testoutput::
    :hide:

    ...

.. testcleanup::
    :skipif: not HAVE_DECTRIS_TESTDATA

    ctx.close()
    server.stop()
    server.maybe_raise()

See the :ref:`DECTRIS reference section <dectris reference>` for a description of
the acquisition parameters.

.. _`merlin detector`:

Quantum Detectors Merlin
------------------------

No extra dependencies are needed for using Merlin detectors.

Supported are currently 1 bit, 6 bit and 12 bit :code:`COUNTERDEPTH` for both the "binary"
(:code:`FILEFORMAT 0`) and the "raw binary" format (:code:`FILEFORMAT 2`).

For testing, an acquisition with soft trigger (:code:`TRIGGERSTART 5`) is
recommended since internal trigger (:code:`TRIGGERSTART 0`) may cause issues
with finding the beginning of the data stream. For a real STEM acquisition a
hardware trigger setup that matches the given instrument is required. See the
MerlinEM User Manual from Quantum Detectors for details!

A simple simulator for testing live acquisition without the actual hardware is
included in LiberTEM-Live. It replays an MIB dataset and accepts the following
parameters:

.. code-block:: shell

    (libertem) $ libertem-live-mib-sim --help
    Usage: libertem-live-mib-sim [OPTIONS] PATH

    Options:
    --nav-shape <INTEGER INTEGER>...
    --continuous
    --cached [NONE|MEM|MEMFD]
    --host TEXT                     Address to listen on (data, control, and
                                    trigger sockets)
    --data-port INTEGER
    --control-port INTEGER
    --wait-trigger                  Wait for a SOFTTRIGGER command on the
                                    control port, or a trigger signal on the
                                    trigger socket
    --garbage                       Send garbage before trigger. Implies --wait-
                                    trigger
    --max-runs INTEGER
    --help                          Show this message and exit.

A suitable MIB dataset for testing can be downloaded at
https://zenodo.org/record/5113449.

See the :ref:`Merlin reference section <merlin reference>` for a description of
the acquisition parameters.
