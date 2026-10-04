"""The whole of what is shown, as a caller saves or shows it.

A :class:`Scene` frames a component on its box and gives it a size in
pixels. How it is written is left to :mod:`nuc2d._svg`.
"""

import os
from typing import TYPE_CHECKING, Any, final

from . import _svg
from ._validation import check_finite_positive
from ._component import Component, Placement


@final
class Scene:
    """The whole of what is shown: a component, framed and sized.

    A component is a part, to be placed among others; a scene is what
    they add up to, and is what is saved or shown.

    Parameters
    ----------
    component : Component
        What the scene shows.
    width_px : float, optional
        Width of the scene in pixels. Given alone, the height follows
        from the component's proportions.
    height_px : float, optional
        Height of the scene in pixels. Given alone, the width follows
        from the component's proportions. Giving neither draws the
        component at its own size, one unit of its box to a pixel. Giving
        both keeps the component's proportions and centers it, rather
        than stretching it to fit.

    Raises
    ------
    TypeError
        If ``component`` is not a :class:`Component`, or ``width_px`` or
        ``height_px`` is neither a number nor None.
    ValueError
        If the component has no width or no height, or ``width_px`` or
        ``height_px`` is a number that is not positive and finite.

    Notes
    -----
    In Jupyter, a scene that ends a cell is displayed as it is.

    A scene cannot be changed once it is made: assigning to an attribute
    raises AttributeError.
    """

    # Only these can be held, so that an attribute assigned by mistake,
    # such as scene.width_px = 800, raises rather than changing nothing.
    __slots__ = ("_component", "_width_px", "_height_px")

    def __init__(
        self,
        component: Component,
        *,
        width_px: float | None = None,
        height_px: float | None = None,
    ) -> None:
        if not isinstance(component, Component):
            # A placement says where a component goes among others. A
            # scene is framed on the box of what it shows, so where that
            # was placed makes no difference to it.
            hint = (
                " Pass the component itself, placement.component; to show "
                "several placed components, make one of them with "
                "Component.from_placements."
                if isinstance(component, Placement)
                else ""
            )
            raise TypeError(
                f"Scene takes a Component; got {type(component).__name__}."
                + hint
            )
        # A scene is framed on the component's box, so it has no area to
        # show without both a width and a height. An empty box has neither.
        bbox = component.bbox
        if bbox.width <= 0 or bbox.height <= 0:
            raise ValueError(
                "A scene needs a component wider and taller than 0; this "
                f"one is {bbox.width!r} wide and {bbox.height!r} tall."
            )
        if width_px is not None:
            width_px = check_finite_positive("width_px", width_px)
        if height_px is not None:
            height_px = check_finite_positive("height_px", height_px)

        aspect_ratio = bbox.width / bbox.height
        if height_px is None:
            if width_px is None:
                width_px, height_px = bbox.width, bbox.height
            else:
                height_px = width_px / aspect_ratio
        elif width_px is None:
            width_px = height_px * aspect_ratio

        self._component = component
        self._width_px = width_px
        self._height_px = height_px

    def to_svg(self) -> str:
        """Return the scene as an SVG document."""
        return _svg.document_string(
            self._component,
            width_px=self._width_px,
            height_px=self._height_px,
        )

    def save_svg(self, path: str | os.PathLike[str]) -> None:
        """Write the scene to ``path`` as an SVG file."""
        _svg.save_document(
            path,
            self._component,
            width_px=self._width_px,
            height_px=self._height_px,
        )

    def _repr_svg_(self) -> str:
        """Let Jupyter display the scene."""
        return self.to_svg()

    # What draw_svg returned up to 1.x was an svgwrite.Drawing, and
    # these are the two of its methods every caller wrote. Asking for either
    # still fails, but says what replaced it.
    _RENAMED = {"tostring": "to_svg", "saveas": "save_svg"}

    # Out of a type checker's sight: one that sees __getattr__ takes any
    # name as an attribute of a scene, so that scene.savesvg would pass.
    if not TYPE_CHECKING:

        def __getattr__(self, name: str) -> Any:
            if name in self._RENAMED:
                raise AttributeError(
                    f"{type(self).__name__!r} object has no attribute "
                    f"{name!r}; nuc2d 2.0 renamed it "
                    f"{self._RENAMED[name]!r}."
                )
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            )
