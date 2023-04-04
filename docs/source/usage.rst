.. _`usage`:

Usage
=====

Currently, LiberTEM-live can be used within Jupyter notebooks or other
applications that allow Python scripting. Support for live data processing in
the web GUI is not implemented at this time.

Usage is deliberately similar to the LiberTEM :ref:`libertem:api documentation`.
It differs in only a few aspects:

* It uses the :class:`libertem_live.api.LiveContext` that extends the
  :class:`libertem.api.Context` with methods to prepare and run live
  acquisitions on top of loading offline datasets.
* The concept of a LiberTEM :class:`~libertem.io.dataset.base.DataSet` is
  adapted to acquiring data from a detector instead of reading
  from a file.

We currently support the following detectors:

* :ref:`Merlin Medipix <merlin detector>`, including 2x2 quad configuration
* :ref:`DECTRIS EIGER2-based detectors <dectris detectors>`, like ARINA or QUADRO

Computations on the live stream use the LiberTEM user-defined functions (UDF) interface.
There are some useful UDFs shipped with both LiberTEM and LiberTEM-live, or you can
implement your own, custom reconstruction methods.

Common setup code
-----------------

As with LiberTEM itself, we have a main entry point for all interaction,
which is the :class:`~libertem_live.api.LiveContext`:

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

    from libertem_live.api import LiveContext

    ctx = LiveContext()

This creates the appropriate resources for computation, in other words, it
starts worker processes and prepares them for receiving data.

The next step is to prepare a connection to the detector system; in most cases
you'll specify network hostnames, IP addresses and/or ports here.

.. code::

    conn = ctx.make_connection('your_detector_type').open(
        key=value,
        ...
    )

For example, for DECTRIS SIMPLON based detectors, creating a connection looks
like this:

.. testcode::

    conn = ctx.make_connection('dectris').open(
        api_host="127.0.0.1",
        api_port=DCU_API_PORT,
        data_host="127.0.0.1",
        data_port=DCU_DATA_PORT,
    )

The connection is usually persistent, so it's important to clean up after yourself:

.. testcode::

    conn.close()

Or use the context manager based interface instead, which automatically cleans up
after the :code:`with`-block:

.. testcode::

    with ctx.make_connection('dectris').open(
        api_host="127.0.0.1",
        api_port=DCU_API_PORT,  # 80 by default
        data_host="127.0.0.1",
        data_port=DCU_DATA_PORT,  # 9999 by default
    ) as conn:
        # your code using the connection here
        pass
    # `conn` is closed here

.. _`passive mode`:

Passive mode
------------

.. versionadded:: 0.2

Possibly the easiest way of using LiberTEM-live is by passively listening
to events on the detector, and starting a reconstruction once the data
starts to arrive. Configuration, arming and triggering is assumed
to be done by an external program, for example from the detector vendor.

See below for the description
of the :ref:`active mode <active mode>`, where the detector is configured and the
acquisition is actively controlled via LiberTEM-live.

In passive mode, you usually use the :meth:`~libertem_live.detectors.base.connection.DetectorConnection.wait_for_acquisition`
to wait for an acquisition to start:

.. testcode::
    :skipif: not HAVE_DECTRIS_TESTDATA

    from libertem.udf.sum import SumUDF

    with ctx.make_connection('dectris').open(
        api_host="127.0.0.1",
        api_port=DCU_API_PORT,
        data_host="127.0.0.1",
        data_port=DCU_DATA_PORT,
    ) as conn:
        # if the timeout, specified in seconds as float here, is hit,
        # `pending_aq` will be `None`. This is useful if you need to
        # regularly do some other work in your code between acquisitions.
        pending_aq = conn.wait_for_acquisition(timeout=10.0)

        aq = ctx.make_acquisition(
            conn=conn,
            pending_aq=pending_aq,
            nav_shape=(128, 128),
        )

        # run one or more UDFs on the live data stream:
        ctx.run_udf(dataset=aq, udf=SumUDF(), plots=True)


This mode works with all detectors in the same way, the only difference
will be the connection parameters.

.. _`active mode`:

Active mode
-----------

.. versionchanged:: 0.2

    The API has changed in 0.2 to seamlessly support different detectors,
    and to allow connecting independently of the acquisition object.

Passive mode is a good way to use LiberTEM-live, if you already have configuration,
arming and triggering set up externally. If you want to integrate this more tightly,
and control everything from one place, you can use active mode instead.

In active mode, the acquisition is actively controlled by LiberTEM-live.
That includes setting detector settings, up to arming the detector.
Depending on your setup, you can also integrate configuration of your
microscope, STEM settings, control your scan engine and start a STEM scan etc.


.. testcode::
    :skipif: not HAVE_DECTRIS_TESTDATA

    from libertem.udf.sum import SumUDF

    with ctx.make_connection('dectris').open(
        api_host="127.0.0.1",
        api_port=DCU_API_PORT,
        data_host="127.0.0.1",
        data_port=DCU_DATA_PORT,
    ) as conn:
        # NOTE: we are no longer passing `pending_aq`, like in the passive mode.
        # Instead we pass a controller object:
        aq = ctx.make_acquisition(
            conn=conn,
            nav_shape=(128, 128),
            controller=conn.get_active_controller(
                # NOTE: parameters here are detector specific
                trigger_mode='exte',
                frame_time=55e-6,
            ),
        )

        # run one or more UDFs on the live data stream:
        ctx.run_udf(dataset=aq, udf=SumUDF(), plots=True)


Hooks
-----

.. versionchanged:: 0.2
    This is a replacement for the previously used :code:`trigger` function,
    and should be an equivalent replacement. The new hooks API is more open
    for future improvements while being backwards-compatible.

In order to integrate LiberTEM-live into your experimental setup,
we provide a way to hook into different points at the lifecycle of
an acquisition. Right now, the most important hook is
:meth:`~libertem_live.api.Hooks.on_ready_for_data`.

This hook is called in :ref:`active mode <active mode>`, when the LiberTEM is
ready to receive data. Depending on the setup and the detector, you can then trigger
a STEM scan, and possibly control other devices, such as signal generators, in-situ
holders with heating etc.

.. testsetup::
    conn = ctx.make_connection('dectris').open(
        api_host="127.0.0.1",
        api_port=DCU_API_PORT,
        data_host="127.0.0.1",
        data_port=DCU_DATA_PORT,
    )

    from libertem.udf.sum import SumUDF


.. testcode::

    from libertem_live.api import Hooks

    class MyHooks(Hooks):
        def on_ready_for_data(self, env):
            """
            You can trigger the scan here, if you have a microscope control API
            """
            print("Triggering!")
            height, width = env.aq.shape.nav
            microscope.trigger_scan(width, height, dwelltime=10e-6)

    aq = ctx.make_acquisition(
        conn=conn,
        nav_shape=(128, 128),
        hooks=MyHooks(),
    )

    # run one or more UDFs on the live data stream:
    ctx.run_udf(dataset=aq, udf=SumUDF(), plots=True)

:meth:`~libertem_live.api.Hooks.on_ready_for_data` is not called for passive
acquisitions, as we cannot accurately synchronize to the beginning of the acquisition
in this case. Also, you will probably have different code to execute based on
active or passive configuration.


Included UDFs
-------------

In addition to :ref:`the UDFs included with LiberTEM <libertem:utilify udfs>`,
we ship :ref:`a few additional UDFs with LiberTEM-live <utility udfs>` that are mostly
useful for live processing.

.. testcleanup::

    # close the context when done to free up resources:
    ctx.close()

    conn.close()
