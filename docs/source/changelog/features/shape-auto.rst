[Feature] Automatically determine :code:`nav_shape` if possible
===============================================================

* This is only possible in passive mode, as we have to know
  the :code:`nav_shape` in active mode.
* If the automatic mode doesn't work for you, it's still possible to
  manually specify the shape, or to override :meth:`libertem_live.hooks.Hooks.on_determine_nav_shape`
  which takes precedence over the automatic detection.
* It's now possible to have placeholders in the :code:`nav_shape`,
  which will be filled with the remainder of the shape. For example,
  It's possible to specify :code:`(-1, -1)` and get a 2D shape, or
  something like :code:`(128, -1)` to fill the last dimension automatically,
  based on the number of frames in the acquisition (:pr:`74`).
