.. _`dectris detectors`:

DECTRIS SIMPLON
===============

.. versionadded:: 0.2

LiberTEM-live has support for all DECTRIS detectors that support
the `SIMPLON API <https://media.dectris.com/210607-DECTRIS-SIMPLON-API-Manual_EIGER2-chip-based_detectros.pdf>`_,
including QUADRO and ARINA.

Usage examples
--------------

This section shortly gives examples how to connect LiberTEM-live to a dectris
DCU (detector control unit). Depending on you setup, you may want to actively
control the detector parameters, synchronization and triggering, which we call
:ref:`active mode <dectris active mode>`, or you may just want to listen for ongoing
acquisitions, and start a reconstruction once the detector is armed and
triggered, which we call the :ref:`passive mode <dectris passive mode>`.


See the :ref:`DECTRIS reference section <dectris reference>` for a description of
the acquisition parameters.

.. _`dectris active mode`:

Active mode
...........

In active mode, the acquisition is controlled actively from the same
Python script or notebook that also controls the reconstruction
with LiberTEM-live. That means it will set detector settings, arm the detector
and has the possibility to integrate with microscope APIs to trigger the scan.

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

    executor = PipelinedExecutor(
        spec=PipelinedExecutor.make_spec(cpus=range(10), cudas=[])
    )
    ctx = LiveContext(executor=executor, plot_class=BQLive2DPlot)

    def trigger(aq):
        """
        You can trigger the scan here, if you have a microscope control API
        """
        # you can imagine something like:
        # trigger_scan(shape=tuple(aq.shape.nav), dwelltime=67e-6)
        pass

    conn = ctx.connect(
        'dectris',
        api_host="127.0.0.1",
        api_port=DCU_API_PORT,
        data_host="127.0.0.1",
        data_port=DCU_DATA_PORT,
        num_slots=2000,
        bytes_per_frame=64*512,
    )

    # prepare for acquisition, setting up scan parameters etc.
    aq = ctx.prepare_acquisition(
        'dectris',
        conn=conn,
        nav_shape=(128, 128),
        trigger=trigger,
        frames_per_partition=512,
        controller=conn.get_active_controller(trigger_mode='exte'),
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

.. _`dectris passive mode`:

Passive mode
............

In passive mode, LiberTEM-live only controls a minimal set of detector
parameters. It enables streaming mode, and makes sure headers are
sent with the right detail level. Other detector parameters are supposed
to be set from the outside, for example using vendor software.
Instead of arming the detector, we wait for the detector to be armed,
and then start receiving and processing data.


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

    executor = PipelinedExecutor(
        spec=PipelinedExecutor.make_spec(cpus=range(10), cudas=[])
    )
    ctx = LiveContext(executor=executor, plot_class=BQLive2DPlot)

    conn = ctx.connect(
        'dectris',
        api_host="127.0.0.1",
        api_port=DCU_API_PORT,
        data_host="127.0.0.1",
        data_port=DCU_DATA_PORT,
        num_slots=2000,
        bytes_per_frame=64*512,
    )

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
    aq = ctx.prepare_from_pending(
        conn=conn,
        nav_shape=(128, 128),
        frames_per_partition=512,
        pending_acquisition=pending_aq,
    )

    # run one or more UDFs on the live data stream:
    ctx.run_udf(dataset=aq, udf=SumUDF(), plots=True)

.. testoutput::
    :hide:

    ...

.. testcleanup::
    :skipif: not HAVE_DECTRIS_TESTDATA

    ctx.close()
    conn.close()
    server.stop()
    server.maybe_raise()

Performance tuning
------------------

As different DECTRIS detectors can have quite different characteristics,
even in different configurations, it's important to properly tune the parameters
to match your situation.

Especially you observe that the reconstruction doesn't keep up with the data
rate of the detector, it can pay off to tweak the parameters.

For example, the DECTRIS ARINA detector, with binning active, will output frames
at up to 120kHz, but each frame will only be 96x96 pixels large. On the other end
of the spectrum, the QUADRO will output frames at 4500Hz, but each frame will be
512x512 pixels large. That means there's about a factor 30 difference between the
two situations, and right now that means some parameters need to be adjusted.

These numbers then directly influence how much data needs to be handled. As in most
situations compression will be used, we can't know the exact thoughput in advance,
so some guesswork and/or testing is involved in choosing the right parameters.

Connecting to the detector system currently looks like this:

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

    from libertem_live.api import LiveContext
    ctx = LiveContext()
    conn = ctx.connect(
        'dectris',
        api_host="127.0.0.1",
        api_port=DCU_API_PORT,
        data_host="127.0.0.1",
        data_port=DCU_DATA_PORT,
        num_slots=2000,
        bytes_per_frame=64*512,
        frame_stack_size=24,
    )

.. testoutput::
    :hide:

    ...

.. testcleanup::
    :skipif: not HAVE_DECTRIS_TESTDATA

    ctx.close()
    conn.close()
    server.stop()
    server.maybe_raise()

You may need to observe your system a bit before being able to tune it. Run a
monitoring utility like :code:`htop`, and try to identify the individual processes.

- If the worker processes show low CPU utilization, but the main Python process
  is at 100% or more, try to *decrease* the :code:`bytes_per_frame` parameter.
  If :code:`bytes_per_frame` is too large, it may be that too many frames
  end up in a single frame stack, meaning it has to be split up at partition
  boundaries. This mostly happens when the value of :code:`bytes_per_frame`
  is way off.
- In the same situation, it can also help to increase the :code:`frames_per_partition`
  parameter of the acquisition, as this decreases the number of updates that
  get sent to the main process.
- If the workers processes are at 100% utilization, it may be that the workload
  is too much for the given hardware. That means either optimizing the code
  of the UDFs you are trying to run, or maybe adding additional resources.
  In some cases it can help to run the UDF on your GPU, in addition to having
  CPU workers.
- It may make sense to actually *reduce* the number of workers a bit below the number
  of cores of your system, as there may be some other processes competing for
  CPU time. The workers might compete with the receiving thread(s) that run
  in the background, and keeping some resources for other system processes
  might make sense.
