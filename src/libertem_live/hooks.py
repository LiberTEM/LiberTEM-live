from typing import Tuple, NamedTuple, TYPE_CHECKING, Optional


if TYPE_CHECKING:
    from libertem_live.detectors.base.acquisition import AcquisitionProtocol


class ReadyForDataEnv(NamedTuple):
    """
    Parameter object used in :meth:`~libertem_live.hooks.Hooks.on_ready_for_data`
    """

    aq: "AcquisitionProtocol"
    """
    The acquisition object which will receive the data.
    """


class DetermineNavShapeEnv(NamedTuple):
    """
    Parameter object used in :meth:`~libertem_live.hooks.Hooks.on_determine_nav_shape`
    """

    nimages: int
    """
    The total number of images in the acquisition.
    """

    shape_hint: Optional[Tuple[int, ...]]
    """
    Shape that was passed into :meth:`~libertem_live.api.LiveContext.make_acquisition`, can contain
    placeholders, i.e. :code:`(-1, 256)` or :code:`(-1, -1)`.
    """


class Hooks:
    """
    Interface for defining actions to perform in reaction to events.
    This is setup- and possibly experiment-specific and can, for example, encode
    how the microscope, scan engine and detector are synchronized.

    By implementing methods of this interface, you can "hook into" the lifecycle of
    an acquisition at different important events.

    Each hook method gets a specific environment object as a parameter, that
    includes all necessary context information from current state of the
    acquisition.
    """

    # FIXME: additional lifecycle methods?
    # - on_setup - before arming?
    # - on_acquisition_done - when all data has been received and processed

    def on_ready_for_data(self, env: ReadyForDataEnv):
        """
        This hook is called whenever we are ready for data, i.e. the detector is
        armed.

        Usually, at this point, the next step is triggering the microscope or
        scan generator to start a scan.

        It is only called in active mode, where we control when the detector is
        armed - in passive mode, the data is possibly already being received and
        we don't control when this happens at all.
        """
        pass

    def on_determine_nav_shape(
        self,
        env: DetermineNavShapeEnv,
    ) -> Optional[Tuple[int, ...]]:
        """
        This hook is called to determine the N-D nav shape for the acquisition.
        This is needed as many detectors only know about the 1D shape, i.e. the
        number of images to acquire. For some, the controller may be able to
        determine the nav shape, but for others, this is specific to the setup
        or experiment.

        The user can give a shape hint when creating the acquisition object,
        for example by passing :code:`nav_shape=(-1, 256)` to
        :meth:`~libertem_live.api.LiveContext.make_acquisition`.

        If the default behavior is not working as intended, the user can override
        this method for their specific setup. Example: ask the microscope via its
        API what the current scan settings are.

        Only called in passive mode, as we need a concrete `nav_shape` as user
        input in active mode.

        Order of operations:

        * If the `nav_shape` passed into
          :meth:`libertem_live.api.LiveContext.make_acquisition` is concrete, i.e.
          is not `None` and doesn't contain any `-1` values, use that
        * If this hook is implemented and returns a tuple, use that
        * If the controller for the specific detector type can give us
          the concrete `nav_shape`, use that
        * If none of the above succeed, try to make a square out of the
          number of frames
        * If all above fails, raise a `RuntimeError`
        """
        pass
