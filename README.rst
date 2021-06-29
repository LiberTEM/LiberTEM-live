|gitter|_ |azure|_ |github|_

.. |gitter| image:: https://badges.gitter.im/Join%20Chat.svg
.. _gitter: https://gitter.im/LiberTEM/Lobby

.. |azure| image:: https://dev.azure.com/LiberTEM/LiberTEM-live/_apis/build/status/LiberTEM.LiberTEM-live?branchName=master
.. _azure: https://dev.azure.com/LiberTEM/LiberTEM-live/_build/latest?definitionId=5&branchName=master

.. |github| image:: https://img.shields.io/badge/GitHub-GPL--3.0-informational
.. _github: https://github.com/LiberTEM/LiberTEM-live/

LiberTEM-live is an extension module for `LiberTEM
<https://libertem.github.io/LiberTEM/>`_ :cite:`Clausen2018` that allows live
data processing.

.. note::
  LiberTEM-live is still experimental and under active development, including
  the overall architecture. New releases can include changes that break
  backwards compatibility until the code and architecture are proven in
  practical application and stabilized sufficiently.

  That being said, we encourage early experimental use, are happy to support
  real-world application and appreciate feedback! You can contact us by
  `creating or commenting on an Issue on GitHub
  <https://github.com/LiberTEM/LiberTEM-live/issues>`_ or in the `LiberTEM
  Gitter chat <https://gitter.im/LiberTEM/Lobby>`_.

LiberTEM `user-defined functions (UDFs)
<https://libertem.github.io/LiberTEM/udf.html>`_ are designed to work without
modification on both offline data and live data streams. That means all
`LiberTEM applications <https://libertem.github.io/LiberTEM/applications.html>`_
and `LiberTEM-based modules
<https://libertem.github.io/LiberTEM/packages.html>`_ can work with all
supported detectors in LiberTEM-live.

Installation
------------

The short version to install into an existing LiberTEM environment:

.. code-block:: shell

    (libertem) $ python -m pip install "libertem-live"

See the `LiberTEM installation instructions
<https://libertem.github.io/LiberTEM/install.html>`_ for more details on
installing LiberTEM.

Detectors
---------

* `Quantum Detectors Merlin
  <https://libertem.github.io/LiberTEM-live/reference.html#quantum-detectors-merlin>`_

Support for the Gatan K2 IS is currently under development.

License
-------

LiberTEM-live is licensed under GPLv3. The I/O parts are also available under
the MIT license, please see LICENSE files in the subdirectories for details.
