"""Components, and where each one is placed.

A :class:`Component` is anything that can be placed: a structure, a
colorbar, a line of text, or several components placed together.
:class:`Placement` says where one goes and at what size, and
:meth:`Component.from_placements` makes one component of several placed
ones.

This module is the geometry of placing, and the checks on what a caller
asks for. It does not know how anything is written, and imports nothing
that does. A drawn component holds its graphics, which only the module
that drew them can read, and one made of others holds only the
placements of its children; how the whole is written is worked out from
them when a scene is. That keeps the dependency one way, so that the
module that writes SVG can make components itself.

The functions at the end of this module are for that module. It reads a
component through them, so that the private fields are touched here and
nowhere else.
"""

import math
import numbers
from dataclasses import dataclass
from typing import Any, Literal, Union

from ._validation import check_finite, check_finite_positive
from ._geometry import BBox, bbox_around, is_empty_bbox


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


class Component:
    """Something that can be placed: a drawing, or several put together.

    Components are made by nuc2d's drawing functions, which draw a
    structure, a colorbar or a line of text, and by
    :meth:`from_placements`, which puts placed components together.
    Calling the class itself raises TypeError.

    Attributes
    ----------
    bbox : BBox
        Extent of the component in its own coordinate system. Its
        ``width`` and ``height`` give its size. Where the box sits is
        otherwise arbitrary; :class:`Placement` positions a component by a
        point of this box, not by its coordinates.

    Notes
    -----
    A component cannot be changed once it is made. Two components are
    equal only if they are the same component, however alike they are
    drawn.

    What the component is drawn with is private, and may change in any
    release.
    """

    # A drawn component holds its graphics: what is drawn, as against its
    # box, the room it takes. Only the module that drew them can read
    # them. A component made of others holds the placements of its
    # children instead, in the order they are drawn.
    __slots__ = ("_bbox", "_graphics", "_child_placements")

    _bbox: BBox
    _graphics: Any
    _child_placements: tuple["Placement", ...]

    def __init__(self, *args: object, **kwargs: object) -> None:
        # What a component holds is private, so a caller has nothing to
        # build one from. Say how one is made instead.
        raise TypeError(
            "Components are not made by calling Component. Draw one with "
            "draw_structure, draw_colorbar or draw_text, or put placed "
            "ones together with Component.from_placements."
        )

    @property
    def bbox(self) -> BBox:
        """Extent of the component in its own coordinate system."""
        return self._bbox

    def __repr__(self) -> str:
        return f"<{type(self).__name__} bbox={self._bbox!r}>"

    @classmethod
    def from_placements(cls, placements: list["Placement"]) -> "Component":
        """Make one component of several placed components.

        Parameters
        ----------
        placements : list[Placement]
            Components to put together, each with where it goes.

        Returns
        -------
        Component
            A component holding every one placed, whose box encloses them
            all. An empty list gives a component with an empty box.

        Raises
        ------
        TypeError
            If ``placements`` is not a list, or an item in it is not a
            :class:`Placement`, such as a component that was not wrapped
            in one.

        Notes
        -----
        Components are drawn in the order given, so a later one covers an
        earlier one where they overlap. No padding is added between them.
        """
        if not isinstance(placements, list):
            raise TypeError(
                "placements must be a list of Placements; "
                f"got {type(placements).__name__}."
            )
        for item in placements:
            if not isinstance(item, Placement):
                raise TypeError(
                    "from_placements takes Placements; wrap each component as "
                    f"Placement(component=...). Got {type(item).__name__}."
                )
        # A copy of our own, which nothing else can change afterwards, so
        # that the box worked out here stays the box of what is held.
        child_placements = tuple(placements)
        return make_component(
            bbox_around(p.bbox for p in child_placements),
            child_placements=child_placements,
        )


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

    Attributes
    ----------
    bbox : BBox
        Extent of the component once placed. Its ``width`` and ``height``
        give the size of the component once scaled.

    Raises
    ------
    TypeError
        If ``component`` is not a :class:`Component`, or ``x``, ``y`` or
        ``scale`` is not a number.
    ValueError
        If ``x`` or ``y`` is not finite, ``anchor`` is neither a name above
        nor a pair of finite numbers, or ``scale`` is not a positive finite
        number.
    """

    component: Component
    x: float = 0.0
    y: float = 0.0
    anchor: Anchor = "upper left"
    scale: float = 1.0

    def __post_init__(self) -> None:
        # Refuse a bad placement where it is written, not when it is used.
        if not isinstance(self.component, Component):
            raise TypeError(
                "component must be a Component, such as draw_structure "
                f"returns; got {type(self.component).__name__}."
            )
        check_finite("x", self.x)
        check_finite("y", self.y)
        _fractions(self.anchor)
        check_finite_positive("scale", self.scale)

    @property
    def bbox(self) -> BBox:
        """Extent of the component once placed."""
        # The component's own box, as distinct from the placed one returned.
        component_bbox = self.component.bbox
        if is_empty_bbox(component_bbox):
            return component_bbox
        fx, fy = _fractions(self.anchor)
        width = component_bbox.width * self.scale
        height = component_bbox.height * self.scale
        left, top = self.x - fx * width, self.y - fy * height
        return BBox(left, top, left + width, top + height)


def fit(
    component: Component,
    slot: BBox,
    *,
    anchor: Anchor,
) -> Placement:
    """Place a component at the largest size that fits inside ``slot``.

    The component keeps its proportions and is aligned inside the slot by
    ``anchor``: the point of the component that it names lands on the
    point of the slot that it names. The placement's box is the slot
    itself rather than the component's own, so that a row of slots stays
    a row of equal cells whatever shape each component is.

    ``anchor`` has no default. Placing a component at a point is done by
    a corner, as ``Placement`` does, while aligning one inside a box is
    usually done by its centre, so neither default is obvious here.
    """
    bbox = component.bbox
    if is_empty_bbox(bbox) or bbox.width <= 0 or bbox.height <= 0:
        raise ValueError("Cannot fit a component that has no area.")
    if is_empty_bbox(slot) or slot.width <= 0 or slot.height <= 0:
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
        component=make_component(
            padded,
            graphics=component._graphics,
            child_placements=component._child_placements,
        ),
        x=slot.xmin,
        y=slot.ymin,
        scale=scale,
    )


# Making a component, and reading what it holds. The module that writes
# SVG makes and reads components through these, and this module makes its
# own through make_component as well.


def make_component(
    bbox: BBox,
    *,
    graphics: Any = None,
    child_placements: tuple[Placement, ...] = (),
) -> Component:
    """Make a component taking up ``bbox``.

    A drawn component is made from its ``graphics``, which are whatever
    the module that drew them needs in order to write them; nothing here
    looks inside. One made of others is made from the placements of its
    children, in the order they are drawn.

    Calling the class does not make a component, so every one is made
    here, and here is where what it holds is filled in.
    """
    component = object.__new__(Component)
    component._bbox = bbox
    component._graphics = graphics
    component._child_placements = child_placements
    return component


def graphics_of(component: Component) -> Any:
    """Return a component's graphics, or None if it is made of others."""
    return component._graphics


def child_placements_of(component: Component) -> tuple[Placement, ...]:
    """Return the placements of a component's children, in drawing order.

    A drawn component has none.
    """
    return component._child_placements


def transform_of(placement: Placement) -> tuple[float, float, float]:
    """Return ``(dx, dy, scale)`` putting a component where it is placed.

    The component's own coordinates are scaled about the origin by
    ``scale``, then moved by ``(dx, dy)``.
    """
    bbox = placement.component.bbox
    if is_empty_bbox(bbox):
        return placement.x, placement.y, placement.scale
    # Scaling about the anchor and then moving the anchor to (x, y) is one
    # scaling about the origin followed by this translation.
    ax, ay = _anchor_point(bbox, placement.anchor)
    return (
        placement.x - placement.scale * ax,
        placement.y - placement.scale * ay,
        placement.scale,
    )
