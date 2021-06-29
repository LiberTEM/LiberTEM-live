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

Simulating a detector
---------------------

For basic testing, the
:class:`~libertem_live.detectors.memory.MemoryAcquisition` can be used. It
implements the additional functionality of an acquisition on top of the
:class:`libertem.io.dataset.memory.MemoryDataSet`. A more advanced data source
that emulates the Merlin data and control socket is available as well. Please
note that this emulation is still very basic for the time being.

This command runs an emulation server on the default ports 6341 for control and
6342 for data which replays the provided MIB dataset:

.. code-block:: shell

    (libertem) $ libertem-live-mib-sim "Ptycho01/20200518 165148/default.hdr"

See :ref:`merlin detector` for all available command line arguments. For
running the example notebooks, you need to use at least the
:code:`--wait-trigger` parameter.

Start a :code:`LiveContext`
---------------------------

A :class:`~libertem_live.api.LiveContext` requires using a compatible executor
since task scheduling has to be coordinated with the incoming detector data. It
starts a suitable executor by default when it is instantiated, currently the
:class:`~libertem.executor.inline.InlineJobExecutor`.

.. testcode::

    from libertem_live.api import LiveContext

    ctx = LiveContext()

.. _`trigger`:

Define a callback function to trigger an acquisition
----------------------------------------------------

This callback function will be included in an acquisition object to be called
when LiberTEM-live is ready and waiting for data. It should trigger the start of
the acquisition, for example by starting a scan. If a scan is triggered before
the acquisition system is ready to receive, data may be lost depending on the
timimg and architecture.

Setting up the devices before the scan with a configuration that generates the
expected data as well as cleanup after a scan is finished should be handled by
the user.

.. testcode::

    def trigger(acquisition):
        print("Triggering!")
        height, width = acquisition.shape.nav
        # microscope.trigger_scan()

Prepare an acquisition
----------------------

The API of an acquisition is deliberately similar to a LiberTEM offline
:class:`~libertem.io.dataset.base.dataset.DataSet` to make the internals of
LiberTEM work the same. However, the behavior and usage are different. An
offline dataset represents immutable data on a storage medium that can be
accessed at any time and be shared between all processes that have access to the
underlying file system. As a consequence,
:class:`~libertem.io.dataset.base.dataset.DataSet` objects can be long-lived and
re-used easily.

:class:`~libertem_live.detector.base.acquisition.AcquisitionMixin` objects, in
contrast, are just a vehicle to communicate to the LiberTEM internals what data
to expect. What data will actually arrive depends entirely on the settings of
the acquisition system. Since doing a 4D STEM acquisition requires coordinating
at least camera, scan engine and microscope, it is usually customized for each
setup. Furthermore, interactive user adjustments such as focusing and navigating
are usually required. That makes generalizing a live acquisition harder than an
offline dataset: It doesn't represent an immutable state on storage like an
offline dataset, but a desired action of an acquisition system that usually has
a unique configuration and depends on a state that is dynamic, complex and
diverse in nature.

For that reason, the functionality of a
:class:`~libertem_live.detector.base.acquisition.AcquisitionMixin` object is
strictly limited to reading data from the camera, expecting data in the shape
and kind that was specified. Controlling the settings of the acquisition system
is the responsibility of the user. The example notebook uses a convenience
function that accepts an acquisition object as a parameter and adjusts the
settings of the acquisition system to match the scan shape of the acquisition.
This function is called directly before running an acquisition, together with
other setup functions.

This example just creates a memory acquisition:

.. testcode::

    import numpy as np

    # We use a memory-based acquisition to make this example runnable
    # without a real detector or detector simulation.
    data = np.random.random((23, 42, 51, 67))

    aq = ctx.prepare_acquisition(
        'memory',
        trigger=trigger,
        data=data,
    )

Run an acquisition
------------------

This first initializes the acquisition to the point that it can receive data, for
example by connecting to the data socket in the case of the Merlin detector. Then it
calls the user-provided :code:`trigger` callback function that should set off
the acquisition. After that, it reads the data from the camera and feeds it into
the provided UDFs. Finally, it closes the camera connection, if applicable.

.. testcode::

    from libertem_live.udf.monitor import SignalMonitorUDF

    res = ctx.run_udf(dataset=aq, udf=SignalMonitorUDF(), plots=True)

.. testoutput::

    Triggering!

Examples
--------

This example is closer to a real-world application based on the Merlin
simulator:

.. toctree::

    merlin
