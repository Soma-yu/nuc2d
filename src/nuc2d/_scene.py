"""The whole of what is shown, as a caller saves or shows it.

A :class:`Scene` frames a component on its box and gives it a size in
pixels. How it is written is left to :mod:`nuc2d._svg`.
"""

import os
from typing import Any

from . import _svg
from ._validation import check_finite_positive
from ._component import Component, Placement


class Scene:
    """The whole of what is shown: a component, framed and sized.

    A component is a part, to be placed among others; a scene is what
    they add up to, and is what is saved or shown.

    Parameters
    ----------
    component : Component
        What the scene shows. The scene is framed on exactly its box.
    width_px : float, optional
        Width of the scene in pixels. Given alone, the height follows
        from the component's proportions.
    height_px : float, optional
        Height of the scene in pixels. Given alone, the width follows
        from the component's proportions. Giving neither sets the height
        to 500. Giving both keeps the component's proportions and centres
        it, rather than stretching it to fit.

    Raises
    ------
    TypeError
        If ``component`` is not a :class:`Component`, or a size given is
        not a number.
    ValueError
        If the component has no width or no height, or a size given is
        not a positive finite number.

    Notes
    -----
    In Jupyter, a scene that ends a cell is displayed as it is.
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
            height_px = 500.0 if width_px is None else width_px / aspect_ratio
        if width_px is None:
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

    def __getattr__(self, name: str) -> Any:
        if name in self._RENAMED:
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}; "
                f"nuc2d 2.0 renamed it {self._RENAMED[name]!r}."
            )
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )
