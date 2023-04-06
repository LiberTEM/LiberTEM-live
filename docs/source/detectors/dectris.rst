.. _`dectris detectors`:

DECTRIS SIMPLON
===============

.. versionadded:: 0.2

LiberTEM-live has support for all DECTRIS detectors that support
the `SIMPLON API <https://media.dectris.com/210607-DECTRIS-SIMPLON-API-Manual_EIGER2-chip-based_detectros.pdf>`_,
including QUADRO and ARINA.

Hardware and Software Requirements
----------------------------------

You should have LiberTEM-live installed on a computer that has a fast (10Gbit+)
connection to the DECTRIS DCU (detector control unit). The exact requirements on
this computer depend on the computation you want to run and the framerate you
want to run it at. For reference, a 10-core system was sufficient for live
processing using the QUADRO at full speed; for ARINA, we used a 24-core AMD EPYC.

We have tested on both Linux and Windows - if you have a choice, Linux should be
preferred.

Usage examples
--------------

This section shortly gives examples how to connect LiberTEM-live to a dectris
DCU. Depending on you setup, you may want to actively
control the detector parameters, synchronization and triggering, which we call
:ref:`active mode <dectris active mode>`, or you may just want to listen for ongoing
acquisitions, and start a reconstruction once the detector is armed and
triggered, which we call the :ref:`passive mode <dectris passive mode>`.


See the :ref:`DECTRIS reference section <dectris reference>` for a description of
the acquisition parameters.

Common to both active and passive mode is the initialization, creating a
:code:`LiveContext` and connecting to the DCU:

.. testsetup::
    :skipif: not HAVE_DECTRIS_TESTDATA

    from libertem_live.detectors.dectris.sim import DectrisSim
    server = DectrisSim(path=DECTRIS_TESTDATA_PATH, port=0, zmqport=0, verbose=False)
    server.start()
    server.wait_for_listen()
    DCU_API_PORT = server.port
    DCU_DATA_PORT = server.zmqport

    class MockMic:
        def trigger_scan(self, width, height, dwelltime):
            pass

    microscope = MockMic()

.. testoutput::
    :hide:

    RustedReplay listening on tcp://127.0.0.1:...
    Waiting for arm


.. testcode::
    :skipif: not HAVE_DECTRIS_TESTDATA

    from libertem.viz.bqp import BQLive2DPlot
    from libertem_live.api import LiveContext
    from libertem.udf.sum import SumUDF

    ctx = LiveContext(plot_class=BQLive2DPlot)

    # connect to the DECTRIS DCU, and set up a shared memory area:
    conn = ctx.make_connection('dectris').open(
        api_host="127.0.0.1",
        api_port=DCU_API_PORT,
        data_host="127.0.0.1",
        data_port=DCU_DATA_PORT,
        buffer_size=2048,
        bytes_per_frame=64*512,
    )

.. _`dectris active mode`:

Active mode
...........

In active mode, the acquisition is controlled actively from the same
Python script or notebook that also controls the reconstruction
with LiberTEM-live. That means it will set detector settings, arm the detector
and has the possibility to integrate with microscope APIs to trigger the scan.

.. testcode::
    :skipif: not HAVE_DECTRIS_TESTDATA

    from libertem_live.api import Hooks

    class MyHooks(Hooks):
        def on_ready_for_data(self, env):
            """
            You can trigger the scan here, if you have a microscope control API
            """
            print("Triggering!")
            height, width = env.aq.shape.nav
            microscope.trigger_scan(width, height, dwelltime=10e-6)

    # prepare for acquisition, setting up scan parameters etc.
    aq = ctx.make_acquisition(
        conn=conn,
        nav_shape=(128, 128),
        hooks=MyHooks(),
        frames_per_partition=512,
        controller=conn.get_active_controller(trigger_mode='exte'),
    )

    # run one or more UDFs on the live data stream:
    ctx.run_udf(dataset=aq, udf=SumUDF())

.. testoutput::

    Triggering!

.. _`dectris passive mode`:

Passive mode
............

In passive mode, LiberTEM-live only controls a minimal set of detector
parameters. It enables streaming mode, and makes sure headers are
sent with the right detail level. Other detector parameters are supposed
to be set from the outside, for example using vendor software.
Instead of arming the detector, we wait for the detector to be armed,
and then start receiving and processing data.


.. testcode::
    :skipif: not HAVE_DECTRIS_TESTDATA

    # NOTE: this is the part that is usually done by an external software,
    # but we include it here to have a running example:
    ec = conn.get_api_client()
    ec.sendDetectorCommand('arm')

    # If the timeout is hit, pending_aq is None.
    # In a real situation, make sure to test for this,
    # for example by looping until a pending acquisition
    pending_aq = conn.wait_for_acquisition(timeout=10.0)

    # prepare for acquisition
    # note that we still have to set the nav_shape here, because
    # we don't get this from the detector - it's controlled by
    # the scan engine or the microscope.
    aq = ctx.make_acquisition(
        conn=conn,
        nav_shape=(128, 128),
        frames_per_partition=512,
        pending_aq=pending_aq,
    )

    # run one or more UDFs on the live data stream:
    ctx.run_udf(dataset=aq, udf=SumUDF())

.. testcleanup::
    :skipif: not HAVE_DECTRIS_TESTDATA

    ctx.close()
    conn.close()
    server.stop()
    server.maybe_raise()
