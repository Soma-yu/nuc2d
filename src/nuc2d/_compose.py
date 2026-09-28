"""Placing drawn parts, and framing the result as a whole.

A :class:`Component` is anything that can be placed: a structure, a
colorbar, or a group of them. :class:`Placement` says where one goes, and
:func:`compose` gathers placed components into a new one. A
:class:`Scene` is the whole: a component framed on its box and sized, as
a caller saves or shows it.

This module is the geometry of placing, and the checks on what a caller
asks for. It does not know how anything is written: a component's
content, and the definitions that content refers to, are opaque here,
and every operation on them goes through :mod:`nuc2d._svg`. Nothing here
takes a drawing to draw into, either, because a component carries its
own definitions and they are written once, when the scene is.
"""

import math
import numbers
import os
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from functools import reduce
from typing import Any, Literal, Union

from . import _svg
from ._geometry import BBox


AnchorName = Literal[
    "upper left", "upper center", "upper right",
    "center left", "center", "center right",
    "lower left", "lower center", "lower right",
]
"""A point of a box, named as matplotlib names a legend's ``loc``."""

Anchor = Union[AnchorName, tuple[float, float]]
"""A point of a box: one of its names, or fractions of the box's width and
height from its upper left corner. ``(0, 0)`` is the upper left, ``(1, 1)``
the lower right, and values outside ``[0, 1]`` fall outside the box."""

_ANCHOR_FRACTIONS: dict[str, tuple[float, float]] = {
    "upper left": (0.0, 0.0),
    "upper center": (0.5, 0.0),
    "upper right": (1.0, 0.0),
    "center left": (0.0, 0.5),
    "center": (0.5, 0.5),
    "center right": (1.0, 0.5),
    "lower left": (0.0, 1.0),
    "lower center": (0.5, 1.0),
    "lower right": (1.0, 1.0),
}


def _fractions(anchor: object) -> tuple[float, float]:
    """Return an anchor as fractions of a box, or raise if it is not one."""
    if isinstance(anchor, str):
        try:
            return _ANCHOR_FRACTIONS[anchor]
        except KeyError:
            names = ", ".join(map(repr, _ANCHOR_FRACTIONS))
            raise ValueError(
                f"anchor must be one of {names}, or a pair of fractions of "
                f"the box such as (0.5, 0.0); got {anchor!r}."
            ) from None
    if isinstance(anchor, tuple) and len(anchor) == 2:
        fx, fy = anchor
        if all(
            isinstance(f, numbers.Real) and not isinstance(f, bool)
            and math.isfinite(f)
            for f in (fx, fy)
        ):
            return float(fx), float(fy)
    raise ValueError(
        "anchor must be a name such as 'upper left', or a pair of finite "
        f"numbers such as (0.5, 0.0); got {anchor!r}."
    )


def _anchor_point(bbox: BBox, anchor: Anchor) -> tuple[float, float]:
    """Return the point of a non-empty box that ``anchor`` names."""
    fx, fy = _fractions(anchor)
    return (bbox.xmin + fx * bbox.width, bbox.ymin + fy * bbox.height)


@dataclass(frozen=True, kw_only=True)
class Component:
    """Something that can be placed: a structure, a colorbar, or a group.

    Components are made by nuc2d's drawing functions and by
    :func:`compose`, not built by hand.

    Attributes
    ----------
    bbox : BBox
        Extent of the component in its own coordinate system. Its
        ``width`` and ``height`` give its size. Where the box sits is
        otherwise arbitrary; :class:`Placement` positions a component by a
        point of this box, not by its coordinates.

    Notes
    -----
    What the component is drawn with is private, and may change in any
    release.
    """

    bbox: BBox
    _content: Any = field(repr=False)
    _definitions: tuple[Any, ...] = field(default=(), repr=False)


@dataclass(frozen=True, kw_only=True)
class Placement:
    """Where a component goes, and at what size.

    The component is scaled about its ``anchor`` point, and moved so that
    this point lands on ``(x, y)``.

    Parameters
    ----------
    component : Component
        Component being placed.
    x : float, default=0.0
        Where the anchor point lands along the x-axis.
    y : float, default=0.0
        Where the anchor point lands along the y-axis. The y-axis points
        down, as in SVG.
    anchor : str or tuple of float, default="upper left"
        The point of the component's box that is placed at ``(x, y)`` and
        stays fixed while it is scaled. One of ``"upper left"``,
        ``"upper center"``, ``"upper right"``, ``"center left"``,
        ``"center"``, ``"center right"``, ``"lower left"``,
        ``"lower center"`` and ``"lower right"``; or a pair of fractions of
        the box's width and height from its upper left corner, so that
        ``(0.5, 0.0)`` is the middle of the top edge.
    scale : float, default=1.0
        Uniform scaling factor. Must be positive.
    z_index : int, default=0
        Drawing order. Components with smaller values are drawn first, so
        later ones cover them.

    Attributes
    ----------
    bbox : BBox
        Extent of the component once placed. Its ``width`` and ``height``
        give the size of the component once scaled.

    Raises
    ------
    ValueError
        If ``anchor`` is neither a name above nor a pair of finite
        numbers, or ``scale`` is not a positive finite number.
    """

    component: Component
    x: float = 0.0
    y: float = 0.0
    anchor: Anchor = "upper left"
    scale: float = 1.0
    z_index: int = 0

    def __post_init__(self) -> None:
        # Refuse a bad placement where it is written, not when composing.
        _fractions(self.anchor)
        if not (math.isfinite(self.scale) and self.scale > 0):
            raise ValueError(
                f"scale must be a positive finite number; got {self.scale!r}."
            )

    @property
    def bbox(self) -> BBox:
        """Extent of the component once placed."""
        # The component's own box, as distinct from the placed one returned.
        component_bbox = self.component.bbox
        if component_bbox.is_empty:
            return component_bbox
        fx, fy = _fractions(self.anchor)
        width = component_bbox.width * self.scale
        height = component_bbox.height * self.scale
        left, top = self.x - fx * width, self.y - fy * height
        return BBox(left, top, left + width, top + height)

    def _placed_content(self) -> Any:
        """The component's content, scaled and moved to where it goes."""
        component_bbox = self.component.bbox
        if component_bbox.is_empty:
            dx, dy = self.x, self.y
        else:
            # Scaling about the anchor and then moving the anchor to (x, y)
            # is one scaling about the origin followed by this translation.
            ax, ay = _anchor_point(component_bbox, self.anchor)
            dx, dy = self.x - self.scale * ax, self.y - self.scale * ay
        return _svg.transformed(
            self.component._content, dx=dx, dy=dy, scale=self.scale
        )


def fit(
    component: Component,
    slot: BBox,
    *,
    anchor: Anchor = "center",
    z_index: int = 0,
) -> Placement:
    """Place a component at the largest size that fits inside ``slot``.

    The component keeps its proportions and is aligned inside the slot by
    ``anchor``. The placement's box is the slot itself rather than the
    component's own, so that a row of slots stays a row of equal cells
    whatever shape each component is.
    """
    bbox = component.bbox
    if bbox.is_empty or bbox.width <= 0 or bbox.height <= 0:
        raise ValueError("Cannot fit a component that has no area.")
    if slot.is_empty or slot.width <= 0 or slot.height <= 0:
        raise ValueError("Cannot fit a component into a slot that has no area.")

    scale = min(slot.width / bbox.width, slot.height / bbox.height)
    fx, fy = _fractions(anchor)
    # The space the slot leaves over, in the component's own units, goes
    # before and after the component in the proportions anchor gives.
    spare_x = slot.width / scale - bbox.width
    spare_y = slot.height / scale - bbox.height
    padded = BBox(
        bbox.xmin - fx * spare_x,
        bbox.ymin - fy * spare_y,
        bbox.xmax + (1 - fx) * spare_x,
        bbox.ymax + (1 - fy) * spare_y,
    )
    return Placement(
        component=replace(component, bbox=padded),
        x=slot.xmin,
        y=slot.ymin,
        scale=scale,
        z_index=z_index,
    )


def compose(placements: Sequence[Placement]) -> Component:
    """Gather placed components into a single component.

    Parameters
    ----------
    placements : Sequence[Placement]
        Components to gather, each with where it goes.

    Returns
    -------
    Component
        A component holding every one placed, whose box encloses them
        all. An empty sequence gives a component with an empty box.

    Raises
    ------
    TypeError
        If an item is not a :class:`Placement`, such as a component that
        was not wrapped in one.

    Notes
    -----
    Components are drawn in ascending order of ``z_index``, and in the
    order given where it is equal. No padding is added between them.
    """
    for item in placements:
        if not isinstance(item, Placement):
            raise TypeError(
                "compose takes Placements; wrap each component as "
                f"Placement(component=...). Got {type(item).__name__}."
            )

    in_order = sorted(placements, key=lambda p: p.z_index)
    return Component(
        bbox=reduce(BBox.union, (p.bbox for p in placements), BBox.empty()),
        _content=_svg.gathered([p._placed_content() for p in in_order]),
        _definitions=_svg.merged_definitions(
            p.component._definitions for p in in_order
        ),
    )


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
        if width_px is None and height_px is None:
            height_px = 500.0
        if width_px is None:
            assert height_px is not None
            width_px = height_px * aspect_ratio
        elif height_px is None:
            height_px = width_px / aspect_ratio

        self._component = component
        self._width_px = width_px
        self._height_px = height_px

    def to_svg(self) -> str:
        """Return the scene as an SVG document."""
        return _svg.document_string(
            self._component._content,
            self._component._definitions,
            viewbox=self._component.bbox.to_viewbox(),
            width_px=self._width_px,
            height_px=self._height_px,
        )

    def save_svg(self, filename: str | os.PathLike[str]) -> None:
        """Write the scene to ``filename`` as an SVG file."""
        _svg.save_document(
            filename,
            self._component._content,
            self._component._definitions,
            viewbox=self._component.bbox.to_viewbox(),
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
