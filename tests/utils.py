import contextlib
from unittest.mock import MagicMock

from libertem_live.detectors.k2is.state import Event, Store


@contextlib.contextmanager
def expect_event(store: Store, event: Event):
    """
    When executing the with-block, :code:`event` should be emitted
    to the given :code:`store`.

    Parameters
    ----------
    store : Store
        [description]
    event : Event
        [description]
    """
    cb = MagicMock()
    store.listen(event.typ, cb)
    try:
        yield
        assert cb.called
        args = cb.call_args[0]

        call_state, call_new_state, call_event, call_effects = args

        assert call_event == event
        assert call_effects == store
        assert call_new_state == store.state
    finally:
        store.remove_callback(cb)
