[Bugfix] Support reading Merlin results async and as an iterator
================================================================

* Make sure the data socket stays connected while we consume from the result generator resp. await the result (:pr:`30`).
