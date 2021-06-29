Reference
=========

Live Context
~~~~~~~~~~~~

.. automodule:: libertem_live.api
   :members:

.. _`detector reference`:

Detectors
~~~~~~~~~

.. _`merlin detector`:

Quantum Detectors Merlin
........................

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

.. automodule:: libertem_live.detectors.merlin
    :members:
    :undoc-members:

.. _`memory detector`:

Memory
......

.. automodule:: libertem_live.detectors.memory
   :members:

UDFs
~~~~

.. automodule:: libertem_live.udf.monitor
   :members:
