[Bugfix] Fix :code:`ReaderPool` flakyness
=========================================

* At the end of the acquisition, it could happen that
  :code:`ReaderPoolImpl.get_result` returns too early, before the result queue
  is drained (:pr:`31`)
