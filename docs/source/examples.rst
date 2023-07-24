
.. _`examples`:

Examples
========

Continuous live preview
~~~~~~~~~~~~~~~~~~~~~~~

A common use case is generating a continuous live preview.

The general pattern looks like this:

.. code-block:: python

    while True:
        pending_aq = conn.wait_for_acquisition()
        aq = ctx.make_acquisition(conn=conn, pending_aq=pending_aq)
        res = ctx.run_udf(dataset=aq, udf=SumUDF())

If you take setup, cancellation, timeouts etc. into account, it could look like
this:

.. code-block:: python

    from libertem.viz.bqp import BQLive2DPlot
    from libertem_live.api import LiveContext
    from libertem.udf.sum import SumUDF
    from libertem.exceptions import UDFRunCancelled

    ctx = LiveContext(plot_class=BQLive2DPlot)

    # connect to the DECTRIS DCU, and set up a shared memory area:
    with ctx.make_connection('dectris').open(
        api_host="127.0.0.1",
        api_port=DCU_API_PORT,
        data_host="127.0.0.1",
        data_port=DCU_DATA_PORT,
        buffer_size=2048,
        bytes_per_frame=64*512,
    ) as conn:
        while True:
            pending_aq = conn.wait_for_acquisition(timeout=10.0)
            if pending_aq is None:
                continue
            aq = ctx.make_acquisition(
                conn=conn,
                nav_shape=(128, 128),
                frames_per_partition=512,
                pending_aq=pending_aq,
            )
            try:
                res = ctx.run_udf(dataset=aq, udf=SumUDF(), plots=True)
                # do something with the result here
            except UDFRunCancelled:
                # this acquisiiton was cancelled, wait for the next one:
                continue

Example notebooks
~~~~~~~~~~~~~~~~~

This example is closer to a real-world application based on the Merlin
simulator:

.. toctree::

    merlin-example

This example shows how to access the detector frame stream without using LiberTEM:

.. toctree::

    merlin-lowlevel-example

This example shows a typical use with a DECTRIS detector, using the active API:

.. toctree::

    dectris-acquisition-example
