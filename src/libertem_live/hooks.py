import math
from typing import Tuple, Optional, TYPE_CHECKING

from libertem_live.detectors.base.controller import AcquisitionController

if TYPE_CHECKING:
    from libertem_live.detectors.base.acquisition import AcquisitionMixin


class Hooks:
    """
    Interface for defining actions to perform in reaction to events.
    This is setup- and possibly experiment-specific and can, for example, encode
    how the microscope, scan engine and detector are synchronized.

    By implementing methods of this interface, you can "hook into" the lifecycle of
    an acquisition at different important events.
    """

    # FIXME: additional lifecycle methods?
    # - on_setup - before arming?
    # - on_acquisition_done - when all data has been received and processed

    # FIXME: type hint for aq: intersection type?
    def on_ready_for_data(self, aq: "AcquisitionMixin"):
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

    def determine_nav_shape(
        self,
        nimages: int,
        controller: AcquisitionController,
        shape_hint: Optional[Tuple[int, ...]] = None,
    ) -> Tuple[int, ...]:
        """
        This hook is called to determine the N-D nav shape for the acquisition.
        This is needed as many detectors only know about the 1D shape, i.e. the
        number of images to acquire. For some, the controller may be able to
        determine the nav shape, but for others, this is specific to the setup
        or experiment.

        The user can give a shape hint when creating the acquisition object,
        for example by passing :code:`nav_shape=(-1, 256)` to `make_acquisition`.

        If the default behavior is not working as intended, the user can override
        this method for their specific setup. Example: ask the microscope via its
        API what the current scan settings are.

        Only called in passive mode, as we need a concrete `nav_shape` as user
        input in active mode.
        """
        # TODO: implement the following behavior:
        #
        # Order of operations to determine `nav_shape`:
        # - If a concrete `nav_shape` is given as `shape_hint`, use that
        # - If a `nav_shape` with placeholders, i.e. `-1` entries, is given,
        #   use the number of images to fill these placeholders
        # - If no `nav_shape` is give, ask the controller
        # - If the controller doesn't know, try to make a 2D square
        # - If all above fails, raise an Exception

        # right now, only perform the last step:
        side = int(math.sqrt(nimages))
        if side * side != nimages:
            raise RuntimeError(
                "Can't handle non-square scans by default, please override"
                " `Hooks.determine_nav_shape` or pass in a concrete nav_shape"
            )
        return (side, side)