Changelog
=========

.. _continuous:

0.3.0.dev0
##########

.. toctree::
  :glob:

  changelog/*/*

.. _latest:

.. _`v0-2-1`:

0.2.1 / 2023-05-09
##################

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.7915819.svg
   :target: https://doi.org/10.5281/zenodo.7915819

Updated the packaging to include the tests in the sdist (:pr:`99`),
and add tests that don't require "real" test data (:pr:`102`, :pr:`103`).

.. _`v0-2-0`:

0.2.0 / 2023-04-21
##################

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.7853129.svg
   :target: https://doi.org/10.5281/zenodo.7853129

This release updates LiberTEM-live to use the new features of
LiberTEM v0.11 and the pipelined executor (:pr:`51`), and adds support for streaming data from two new detectors.
It introduces a new, more convenient API for connecting to detectors and running
UDFs in an active or passive way. It also includes enhancements for
Merlin Medipix support.

Certain parts of LiberTEM-live are now implemented in Rust with Python bindings,
these now live in `the LiberTEM-rs repository
<https://github.com/LiberTEM/LiberTEM-rs/>`_. Among other things, this includes
receiving data directly into a shared memory area.

This release also drops support for Python 3.6, which is no longer supported
upstream.

New features
------------

* Support for DECTRIS detectors, like QUADRO or ARINA (:pr:`51`, :pr:`74`).
* Initial support for ASI TPX3 detectors (:pr:`81`).
* Support for 2x2 layouts of Merlin Medipix, including 1/6/12bit raw data
  formats (:pr:`36`, :issue:`34`).
* :ref:`recording` with :class:`~libertem_live.udf.record.RecordUDF` (:pr:`51`).
* Updated user-facing APIs (:pr:`74`).
   * Most operations are now reachable just by importing the :class:`libertem_live.api.LiveContext`,
     no need to manually import detector-specific classes etc.
   * Introduce :class:`libertem_live.hooks.Hooks` as a replacement
     for the trigger function that is open for future enhancements.
   * Generic interface :meth:`libertem_live.api.LiveContext.make_acquisition` for
     creating acquisition objects; the detector specific parameters are now mostly
     specified in :meth:`libertem_live.api.LiveContext.make_connection`. Connecting
     and creating an acquisition object is now a two-step process.
   * Support for a passive acquisition workflow using
     :meth:`libertem_live.detectors.base.connection.DetectorConnection.wait_for_acquisition`.
* Automatically determine :code:`nav_shape` if possible (:pr:`74`).
   * This is only possible in passive mode, as we have to know
     the :code:`nav_shape` in active mode.
   * If the automatic mode doesn't work for you, it's still possible to
     manually specify the shape, or to override :meth:`libertem_live.hooks.Hooks.on_determine_nav_shape`
     which takes precedence over the automatic detection.
   * It's now possible to have placeholders in the :code:`nav_shape`,
     which will be filled with the remainder of the shape. For example,
     It's possible to specify :code:`(-1, -1)` and get a 2D shape, or
     something like :code:`(128, -1)` to fill the last dimension automatically,
     based on the number of frames in the acquisition.

Bugfixes
--------

* Fix for the low level Merlin Medipix API, adding async and iterator support (:pr:`30`).
* Fix flakyness in the Merlin Medipix receiver, which could misbehave at the end
  of an acquisition (:pr:`31`).


Obsolescence
------------

* Drop Python 3.6 support (:pr:`68`).
* All acquisition objects now take the same parameter set, meaning scripts
  using the early Merlin Medipix support need to be adjusted to the new API (:pr:`27` and others).

.. _`v0-1-0`:

0.1.0 / 2021-06-29
##################

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4916316.svg
   :target: https://doi.org/10.5281/zenodo.4916316

Initial release with support for Quantum Detectors Merlin.
