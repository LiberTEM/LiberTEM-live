[Feature] Update user-facing APIs
=================================

* Most operations are now reachable just by importing the :class:`libertem_live.api.LiveContext`,
  no need to manually import detector-specific classes etc.
* Introduce :class:`libertem_live.hooks.Hooks` as a replacement
  for the trigger function that is open for future enhancements.
* Generic interface :meth:`libertem_live.api.LiveContext.make_acquisition` for
  creating acquisition objects; the detector specific parameters are now mostly
  specified in :meth:`libertem_live.api.LiveContext.make_connection`. Connecting
  and creating an acquisition object is now a two-step process.
* Support for a passive acquisition workflow using
  :meth:`libertem_live.detectors.base.connection.DetectorConnection.wait_for_acquisition`
  (:pr:`74`).
