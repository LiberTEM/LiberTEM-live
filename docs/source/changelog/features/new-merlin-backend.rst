[Feature] New data backend for QD Merlin Medipix
================================================

* For better performance and stability, the new QD Merlin Medipix
  backend is re-written in rust, see https://github.com/LiberTEM/LiberTEM-rs/pull/65
  for the rust side. Both the Python and rust sides have been refactored to be
  more generic, such that adding support for new (frame-based) detectors is easier
  (:pr:`161`, :issue:`160`).
