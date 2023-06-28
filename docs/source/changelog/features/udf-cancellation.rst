[Feature] Properly cancel UDF runs when data stream is interrupted
==================================================================

* Raise a :class:`~libertem.common.executor.JobCancelledError`
  when we receive less data than expected, which the user can
  gracefully handle (:pr:`121`).
