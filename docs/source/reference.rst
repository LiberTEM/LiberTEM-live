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

The simulator for the Quantum Detectors Merlin camera accepts the following
parameters:

.. code-block:: shell
    
    (libertem) $ libertem-live-mib-sim --help
    Usage: libertem-live-mib-sim [OPTIONS] PATH

    Options:
    --continuous
    --cached [NONE|MEM|MEMFD]
    --data-port INTEGER
    --control-port INTEGER
    --max-runs INTEGER
    --help                     Show this message and exit.

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