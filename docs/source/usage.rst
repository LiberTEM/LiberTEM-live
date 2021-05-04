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

See :ref:`merlin` for all available command line arguments.

Start a LiveContext
-------------------

.. testcode::

    from libertem_live.api import LiveContext

    ctx = LiveContext()

Define a routine to start and stop an acquisition
-------------------------------------------------

TODO to be discussed!

.. testcode::

    from contextlib import contextmanager

    @contextmanager
    def mem_setup(dataset, udfs):
        print("memory prep")
        # additional setup commands go here
        print("running acquisition")
        with dataset.start_acquisition():
            yield
        print("camera teardown")
        # additional cleanup commands go here
 
Prepare an acquisition
----------------------

The acquisition definition is deliberately similar to a LiberTEM dataset.
It splices the setup context manager into the general camera interface. 

.. testcode::

    import numpy as np

    data = np.random.random((23, 42, 51, 67))

    ds = ctx.prepare_acquisition(
        'memory',
        mem_setup,
        data=data,
        num_partitions=23
    )

Run an acquisition
------------------

This first initializes the acquisition system, for example by connecting to the
camera control interface. Then it enters the user-provided context manager to
execute additional setup commands. After that, it reads the data from the camera
and feeds it into the provided UDFs. Finally, the control flow leaves the
user-provided context manager and all connections to the camera are closed.

.. testcode::

    from libertem_live.udf.monitor import SignalMonitorUDF

    ctx.run_udf(dataset=ds, udf=SignalMonitorUDF(), plots=True)

Examples
--------

This example is closer to a real-world application based on the Merlin
simulator:

.. toctree::

    merlin
