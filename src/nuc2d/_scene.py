"""The whole of what is shown, as a caller saves or shows it.

A :class:`Scene` frames a component on its box and gives it a size in
pixels. How it is written is left to :mod:`nuc2d._svg`.
"""

import math
import os
from typing import Any

from . import _svg
from ._component import Component


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
    ValueError
        If the component is empty, or a size given is not a positive
        finite number.

    Notes
    -----
    In Jupyter, a scene that ends a cell is displayed as it is.
    """

    def __init__(
        self,
        component: Component,
        *,
        width_px: float | None = None,
        height_px: float | None = None,
    ) -> None:
        bbox = component.bbox
        if bbox.is_empty:
            raise ValueError("Nothing to draw: the component is empty.")
        for name, size in (("width_px", width_px), ("height_px", height_px)):
            if size is not None and not (math.isfinite(size) and size > 0):
                raise ValueError(
                    f"{name} must be a positive finite number; got {size!r}."
                )

        aspect_ratio = bbox.width / bbox.height if bbox.height > 0 else 1.0
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

    def save_svg(self, filename: str | os.PathLike[str]) -> None:
        """Write the scene to ``filename`` as an SVG file."""
        _svg.save_document(
            filename,
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
