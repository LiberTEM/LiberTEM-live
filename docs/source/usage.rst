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
* :ref:`Amsterdam Scientific Instruments TPX3 <asi tpx3>` (experimental)

Computations on the live stream use the LiberTEM user-defined functions (UDF) interface.
There are some useful UDFs shipped with both LiberTEM and LiberTEM-live, or you can
implement your own, custom processing methods.

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
    :skipif: not HAVE_DECTRIS_TESTDATA
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
    :skipif: not HAVE_DECTRIS_TESTDATA

    conn = ctx.make_connection('dectris').open(
        api_host="127.0.0.1",
        api_port=DCU_API_PORT,
        data_host="127.0.0.1",
        data_port=DCU_DATA_PORT,
    )

The connection is usually persistent, so it's important to clean up after yourself:

.. testcode::
    :skipif: not HAVE_DECTRIS_TESTDATA

    conn.close()

Or use the context manager based interface instead, which automatically cleans up
after the :code:`with`-block:

.. testcode::
    :skipif: not HAVE_DECTRIS_TESTDATA

    with ctx.make_connection('dectris').open(
        api_host="127.0.0.1",
        api_port=DCU_API_PORT,  # 80 by default
        data_host="127.0.0.1",
        data_port=DCU_DATA_PORT,  # 9999 by default
    ) as conn:
        # your code using the connection here
        pass
    # `conn` is closed here

Coordinating live processing
----------------------------

As a general design goal, LiberTEM should behave similarly between offline and
live processing. Once created, live acquisition objects can be used very
similarly to offline datasets. However, the creation process is different: In
offline processing, most relevant parameters are pre-determined by an existing
dataset, and most datasets share very similar user-controlled parameters.
Datasets backed by files can be read at any time and in any sequence.

In contrast, parameters and actions for live processing are dynamic and have to
be coordinated correctly in a sequence between microscope, scan engine, detector
and LiberTEM processing so that the setup generates the data that LiberTEM
expects to receive. Data can only be read sequentially and has to be consumed in
a short time window to prevent dropping frames. Furthermore, the parameters and
actions can be rather different between different setups and may have to be
customized to a higher degree than offline datasets.

Live acquisitions are therefore created in a multi-step procedure to separate
concerns of detector interface, detector parameters, hooks for synchronization
and customization, and generic LiberTEM parameters. Both an "active mode" where
LiberTEM sets parameters and initiates an acquisition, and a "passive mode"
where LiberTEM reads parameters and waits for an acquisition are available.

.. _`passive mode`:

Passive mode
------------

.. versionadded:: 0.2

Possibly the easiest way of using LiberTEM-live is by passively listening
to events on the detector, and starting processing once the data
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
        # NOTE: this is the part that is usually done by an external software,
        # but we include it here to have a running example:
        ec = conn.get_api_client()
        ec.sendDetectorCommand('arm')

        # if the timeout, specified in seconds as float here, is hit,
        # `pending_aq` will be `None`. This is useful if you need to
        # regularly do some other work in your code between acquisitions.
        pending_aq = conn.wait_for_acquisition(timeout=10.0)

        if pending_aq is not None:
            aq = ctx.make_acquisition(
                conn=conn,
                pending_aq=pending_aq,
                nav_shape=(128, 128),
            )

            # run one or more UDFs on the live data stream:
            ctx.run_udf(dataset=aq, udf=SumUDF())


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
                # NOTE: parameters here are detector-specific
                trigger_mode='exte',
                frame_time=1e-3,
            ),
        )

        # run one or more UDFs on the live data stream:
        ctx.run_udf(dataset=aq, udf=SumUDF())


Hooks
-----

.. versionchanged:: 0.2
    This is a replacement for the previously used :code:`trigger` function,
    and should be an equivalent replacement. The new hooks API is more open
    for future improvements while being backwards-compatible.

In order to integrate LiberTEM-live into your experimental setup,
we provide a way to hook into different points during the lifecycle of
an acquisition.

`on_ready_for_data`
...................

Right now, the most important hook is
:meth:`~libertem_live.hooks.Hooks.on_ready_for_data`.

This hook is called in :ref:`active mode <active mode>`, when LiberTEM is
ready to receive data. Depending on the setup and the detector, you can then trigger
a STEM scan, and possibly control other devices, such as signal generators, in-situ
holders with heating etc.

.. testsetup::
    :skipif: not HAVE_DECTRIS_TESTDATA

    from libertem_live.api import LiveContext
    from libertem.udf.sum import SumUDF

    ctx = LiveContext()

    conn = ctx.make_connection('dectris').open(
        api_host="127.0.0.1",
        api_port=DCU_API_PORT,
        data_host="127.0.0.1",
        data_port=DCU_DATA_PORT,
    )


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

    with conn:
        aq = ctx.make_acquisition(
            conn=conn,
            nav_shape=(128, 128),
            hooks=MyHooks(),
        )

        # run one or more UDFs on the live data stream:
        ctx.run_udf(dataset=aq, udf=SumUDF())

.. testoutput::
    :skipif: not HAVE_DECTRIS_TESTDATA

    Triggering!

:meth:`~libertem_live.hooks.Hooks.on_ready_for_data` is not called for passive
acquisitions, as we cannot accurately synchronize to the beginning of the acquisition
in this case. Also, you will probably have different code to execute based on
active or passive configuration.

`on_determine_nav_shape`
........................

Another hook is :meth:`~libertem_live.hooks.Hooks.on_determine_nav_shape`.
In passive mode, the :code:`nav_shape` is needed to make an acquisition instance.
As the scanning parameters can change over time, we now have added the possibility
to leave out the :code:`nav_shape` parameter, or set it to :code:`None`, which means
it will automatically be determined. As this automatism can fail, for example if you are
only performing a 1D scan (line scan or generic "time series"), it is now also
possible to override this with the :meth:`~libertem_live.hooks.Hooks.on_determine_nav_shape`
method.

In active mode, this hook method is not called, as the full :code:`nav_shape`
is passed to :meth:`~libertem_live.api.LiveContext.make_acquisition`.

.. testsetup::
    :skipif: not HAVE_DECTRIS_TESTDATA

    from libertem_live.api import LiveContext
    from libertem.udf.sum import SumUDF

    ctx = LiveContext()

    conn = ctx.make_connection('dectris').open(
        api_host="127.0.0.1",
        api_port=DCU_API_PORT,
        data_host="127.0.0.1",
        data_port=DCU_DATA_PORT,
    )


.. testcode::
    :skipif: not HAVE_DECTRIS_TESTDATA

    from libertem_live.api import Hooks

    class MyHooks(Hooks):
        def on_determine_nav_shape(self, env):
            print(f"We have {env.nimages} images")
            # We return the actual nav shape we want. It should match the
            # number of images.
            if env.nimages == 16384:
                return (64, 256)
            else:
                raise RuntimeError(f"Expected 16384 images, got {env.nimages}")

    with conn:
        # NOTE: this is the part that is usually done by an external software,
        # but we include it here to have a running example:
        ec = conn.get_api_client()
        ec.sendDetectorCommand('arm')

        pending_aq = conn.wait_for_acquisition(timeout=10.0)
        aq = ctx.make_acquisition(
            conn=conn,
            pending_aq=pending_aq,
            hooks=MyHooks(),
        )


.. testoutput::
    :skipif: not HAVE_DECTRIS_TESTDATA

    We have 16384 images


See :class:`~libertem_live.hooks.DetermineNavShapeEnv` for details on the passed
:code:`env` parameter.

.. note::

    If you don't override this hook, LiberTEM-live tries to determine or guess the
    :code:`nav_shape` based on the following method:

    #. If a concrete tuple of integers is passed into :meth:`~libertem_live.api.LiveContext.make_acquisition`,
       this tuple is used as-is.
    #. The :code:`nav_shape` can contain placeholders, i.e. values of :code:`-1`. These are handled
       similarly as numpy does for reshaping arrays, so if you give :code:`(-1, 64)` for an acquisition of 16384 images,
       the final shape will be :code:`(256, 64)`. For :code:`(4, -1, -1)`, it would be :code:`(4, 64, 64)`,
       so two placeholders are filled with a square shape. Up to two placeholders are allowed.
    #. If no :code:`nav_shape` is given, it is either determined by asking the detector API,
       or, if this is not available, it is assumed to be a 2D square.


Live visualization
------------------

The easiest way to get a live visualization going in a Jupyter notebook
is to pass :code:`plots=True` to :meth:`libertem:libertem.api.Context.run_udf`,
which will automatically add a live-updating plot to the notebook cell output.

In some cases, updating the plot can become a bottleneck - one way to
circumvent this is to use `bqplot` for visualization. Please see :ref:`the examples <examples>`
for usage.

Included UDFs
-------------

In addition to :ref:`the UDFs included with LiberTEM <libertem:utilify udfs>`,
we ship :ref:`a few additional UDFs with LiberTEM-live <utility udfs>` that are mostly
useful for live processing.

.. _`recording`:

Recording data
--------------

The :class:`~libertem_live.udf.record.RecordUDF` allows to record the input data
as NPY file.

.. testsetup::
    :skipif: not HAVE_DECTRIS_TESTDATA

    import os
    from tempfile import TemporaryDirectory

    d = TemporaryDirectory()
    filename = os.path.join(d.name, 'numpyfile.npy')

.. testcode::
    :skipif: not HAVE_DECTRIS_TESTDATA

    from libertem_live.udf.record import RecordUDF

    conn = ctx.make_connection('dectris').open(
        api_host="127.0.0.1",
        api_port=DCU_API_PORT,
        data_host="127.0.0.1",
        data_port=DCU_DATA_PORT,
    )

    aq = ctx.make_acquisition(
        conn=conn,
        nav_shape=(128, 128),
    )

    ctx.run_udf(dataset=aq, udf=RecordUDF(filename))

.. testcleanup::
    :skipif: not HAVE_DECTRIS_TESTDATA

    # close the context when done to free up resources:
    ctx.close()

    conn.close()
