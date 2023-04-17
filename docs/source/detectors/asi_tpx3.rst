
.. _`asi tpx3`:

Amsterdam Scientific Instruments CheeTah TPX3
=============================================

.. versionadded:: 0.2

    This detector is experimentally supported for now, and needs
    matching Serval and Accos versions running.

At this time, only the passive mode is implemented, which relies
on configuring the detector and acquisition from the outside.
Adding active mode support is planned for the future.

Example
~~~~~~~

.. testsetup::
    :skipif: not HAVE_TPX3_TESTDATA

    from libertem_live.detectors.asi_tpx3.sim import TpxCameraSim
    from libertem_live.api import LiveContext

    server = TpxCameraSim(
        paths=[TPX3_TESTDATA_PATH], port=0, cached='MEM', sleep=0.1,
    )
    server.start()
    server.wait_for_listen()
    TPX3_DATA_PORT = server.server_t.port

    ctx = LiveContext()


.. testcode::
    :skipif: not HAVE_TPX3_TESTDATA

    from libertem.viz.bqp import BQLive2DPlot
    from libertem_live.api import LiveContext
    from libertem.udf.sum import SumUDF

    ctx = LiveContext(plot_class=BQLive2DPlot)

    with ctx.make_connection('asi_tpx3').open(
        data_host="127.0.0.1",
        data_port=TPX3_DATA_PORT,
        chunks_per_stack=16,
        bytes_per_chunk=1500000,
        buffer_size=2048,  # MiB
    ) as conn:

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
            frames_per_partition=512,
            pending_aq=pending_aq,
        )

        # run one or more UDFs on the live data stream:
        ctx.run_udf(dataset=aq, udf=SumUDF())


.. testcleanup::
    :skipif: not HAVE_TPX3_TESTDATA

    ctx.close()
    conn.close()
    server.stop()
    server.maybe_raise()
