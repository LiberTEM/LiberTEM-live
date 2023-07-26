[Feature] Properly cancel UDF runs when data stream is interrupted
==================================================================

* Raise a :class:`~libertem.exceptions.UDFRunCancelled` exception
  when we receive less data than expected, which the user can
  gracefully handle (:pr:`121`). Needs LiberTEM v0.12+; older versions
  won't be able to continue as gracefully.
