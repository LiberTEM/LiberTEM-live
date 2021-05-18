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
* It requires using a compatible executor since task scheduling has to be
  coordinated with the incoming detector data. The
  :class:`~libertem_live.api.LiveContext` starts a suitable executor by default
  when it is instantiated, currently the
  :class:`~libertem.executor.inline.InlineJobExecutor`.
* The concept of a LiberTEM :class:`~libertem.io.dataset.base.DataSet` is
  extended with provisions for controlling, starting and stopping data
  acquisition from a detector instead of reading from a file. That includes
  running user-provided code to send the necessary control commands to the
  microscope and other devices to perform an acquisition.

Simulating a detector
---------------------

For basic testing, the
:class:`~libertem_live.detectors.memory.MemoryLiveDataSet` can be used. It
implements the additional functionality of a live dataset on top of the
:class:`libertem.io.dataset.memory.MemoryDataSet`. A more advanced data source
that emulates the Merlin data and control socket is available as well. Please
note that this emulation is still very basic for the time being.

This command runs an emulation server on the default ports 6341 for control and
6342 for data which replays the provided MIB dataset:

.. code-block:: shell
    
    (libertem) $ libertem-live-mib-sim "Ptycho01/20200518 165148/default.hdr"

See :ref:`merlin detector` for all available command line arguments.

Start a LiveContext
-------------------

.. testcode::

    from libertem_live.api import LiveContext

    ctx = LiveContext()

.. _`enter exit`:

Define routines to start and stop an acquisition
-------------------------------------------------

These functions will be included in a live dataset object to be called when an
acquisition is started resp. after it is finished.

:code:`on_enter()`: can be used to execute setup commands, for example to
configure camera, scan system and triggers for the desired scan, and to initiate
the scan.

:code:`on_exit()`: can be used to bring the setup back to the default state, for
example by disarming triggers.

.. testcode::

    def on_enter(meta):
        print("Calling on_enter")
        print("Dataset shape:", meta.dataset_shape)
    
    def on_exit(meta):
        print("Calling on_exit")
 
Prepare an acquisition
----------------------

The acquisition definition is deliberately similar to a LiberTEM dataset.
The :code:`on_enter()` and :code:`on_exit()` functions are included in the
acquisition object and are called when the acquisition is run.

.. testcode::

    import numpy as np

    # We use a memory-based acquisition to make this example runnable
    # without a real detector or detector simulation.
    data = np.random.random((23, 42, 51, 67))

    ds = ctx.prepare_acquisition(
        'memory',
        on_enter=on_enter,
        on_exit=on_exit,
        data=data,
    )

Run an acquisition
------------------

This first calls the user-provided :code:`on_enter` function. After that, it reads the data from the camera and feeds
it into the provided UDFs. Finally, it closes the camera connection and calls
the user-provided :code:`on_exit` function.

The :code:`on_enter` and :code:`on_exit` functions are called with a
:class:`~libertem_live.detectors.base.meta.LiveMeta` object that contains meta
information about the acquisition. See the documentation of
:class:`~libertem_live.detectors.base.meta.LiveMeta` for an overview over the
available information.

.. testcode::

    from libertem_live.udf.monitor import SignalMonitorUDF

    res = ctx.run_udf(dataset=ds, udf=SignalMonitorUDF(), plots=True)

.. testoutput::

    Calling on_enter
    Dataset shape: (23, 42, 51, 67)
    Calling on_exit

Examples
--------

This example is closer to a real-world application based on the Merlin
simulator:

.. toctree::

    merlin
