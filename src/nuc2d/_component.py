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
from dataclasses import dataclass
from typing import Any, Literal, Union, final

from ._validation import check_finite, check_finite_positive, exact_str
from ._geometry import BBox, bbox_around, is_empty_bbox, make_bbox


AnchorName = Literal[
    "upper left", "upper center", "upper right",
    "center left", "center", "center right",
    "lower left", "lower center", "lower right",
]
"""A point of a box, named as matplotlib names a legend's ``loc``."""

Anchor = Union[AnchorName, tuple[float, float]]
"""A point of a box: one of its names, or fractions of the box's width and
height from its upper left corner, each from 0 to 1. ``(0, 0)`` is the
upper left and ``(1, 1)`` the lower right."""

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


def _fraction(name: str, value: object) -> float:
    """Return one fraction of an anchor, or raise unless it is from 0 to 1.

    An anchor is a point of the box, and moving a component off it is what
    x and y are for. There is no tolerance: a fraction worked out to land
    a hair past 0 or 1 is refused with its value in the message, which says
    what happened, and a tolerance could be added later without breaking
    anything, while one given now could not be taken back.
    """
    fraction = check_finite(name, value)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(
            f"{name} must be a fraction of the box from 0 to 1; got "
            f"{value!r}. An anchor is a point of the box; to move the "
            "component, change x or y instead."
        )
    return fraction


def _fractions(anchor: object) -> tuple[float, float]:
    """Return an anchor as fractions of a box, or raise if it is not one."""
    if isinstance(anchor, str):
        name = exact_str(anchor)
        try:
            return _ANCHOR_FRACTIONS[name]
        except KeyError:
            names = ", ".join(map(repr, _ANCHOR_FRACTIONS))
            raise ValueError(
                f"anchor must be one of {names}, or a tuple of two fractions "
                f"of the box from 0 to 1, such as (0.5, 0.0); got {anchor!r}."
            ) from None
    elif isinstance(anchor, tuple) and len(anchor) == 2:
        fx, fy = anchor
        return _fraction("anchor[0]", fx), _fraction("anchor[1]", fy)
    elif isinstance(anchor, tuple):
        # A tuple of another length is another type, tuple[float] or
        # tuple[float, float, float], as it is to os.utime's pair of times.
        raise TypeError(
            "anchor must be a tuple of two fractions from 0 to 1, such as "
            f"(0.5, 0.0); got {anchor!r}."
        )
    else:
        raise TypeError(
            "anchor must be a name such as 'upper left', or a tuple of two "
            "fractions from 0 to 1, such as (0.5, 0.0); got "
            f"{type(anchor).__name__}."
        )


def _anchor_point(
    bbox: BBox, anchor: tuple[float, float]
) -> tuple[float, float]:
    """Return the point of a non-empty box at these fractions of it."""
    fx, fy = anchor
    return (bbox.xmin + fx * bbox.width, bbox.ymin + fy * bbox.height)


@final
class Component:
    """Something that can be placed: a drawing, or several put together.

    Components are made by :func:`draw_svg_as_component`, which draws a
    structure, and by :meth:`from_placements`, which puts placed
    components together. Calling the class itself raises TypeError.

    Attributes
    ----------
    bbox : BBox
        Extent of the component in its own coordinate system. Its
        ``width`` and ``height`` give its size. Where the box sits is
        otherwise arbitrary; :class:`Placement` positions a component by a
        point of this box, not by its coordinates.

    Notes
    -----
    A component cannot be changed once it is made: assigning to an
    attribute raises AttributeError.

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
            "draw_svg_as_component, or put placed ones together with "
            "Component.from_placements."
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
            all.

        Raises
        ------
        TypeError
            If ``placements`` is not a list, or an item in it is not a
            :class:`Placement`, such as a component that was not wrapped
            in one.
        ValueError
            If ``placements`` is empty, or the box around them would not
            have a finite width and height.

        Notes
        -----
        Components are drawn in the order given, so a later one covers an
        earlier one where they overlap.
        """
        if not isinstance(placements, list):
            raise TypeError(
                "placements must be a list of Placements; "
                f"got {type(placements).__name__}."
            )
        if not placements:
            raise ValueError(
                "placements is empty, so there is nothing to put together."
            )
        for item in placements:
            if not isinstance(item, Placement):
                raise TypeError(
                    "from_placements takes Placements; wrap each component as "
                    f"Placement(component). Got {type(item).__name__}."
                )
        # A copy of our own, which nothing else can change afterwards, so
        # that the box worked out here stays the box of what is held.
        child_placements = tuple(placements)
        bbox = bbox_around(p.bbox for p in child_placements)
        # Each placed box is finite, but placements far enough apart make
        # one around them wider or taller than the largest float.
        if not (math.isfinite(bbox.width) and math.isfinite(bbox.height)):
            raise ValueError(
                "The box around the placements would not be finite: "
                f"{bbox.width!r} wide and {bbox.height!r} tall."
            )
        return make_component(bbox, child_placements=child_placements)


@final
@dataclass(frozen=True, init=False, match_args=False)
class Placement:
    """Where a component goes, and at what size.

    The component is scaled about its ``anchor`` point, and moved so that
    this point lands on ``(x, y)``.

    Parameters
    ----------
    component : Component
        Component being placed.
    anchor : str or tuple[float, float], default="upper left"
        The point of the component's box that is placed at ``(x, y)`` and
        stays fixed while it is scaled. One of ``"upper left"``,
        ``"upper center"``, ``"upper right"``, ``"center left"``,
        ``"center"``, ``"center right"``, ``"lower left"``,
        ``"lower center"`` and ``"lower right"``; or a tuple of two
        fractions of the box's width and height from its upper left corner,
        each from 0 to 1, so that ``(0.5, 0.0)`` is the middle of the top
        edge.
    x : float, default=0.0
        Where the anchor point lands along the x-axis.
    y : float, default=0.0
        Where the anchor point lands along the y-axis. The y-axis points
        down, as in SVG.
    scale : float, default=1.0
        Uniform scaling factor. Must be positive.

    Attributes
    ----------
    anchor : tuple[float, float]
        The anchor as a tuple of two fractions of the box, however it was
        given: ``"upper left"`` is kept as ``(0.0, 0.0)``.
    bbox : BBox
        Extent of the component once placed. Its ``width`` and ``height``
        give the size of the component once scaled.

    Raises
    ------
    TypeError
        If ``component`` is not a :class:`Component`; ``anchor`` is
        neither a string nor a tuple of two, or holds something other than
        a number; or ``x``, ``y`` or ``scale`` is not a number.
    ValueError
        If ``anchor`` is a name not listed above, or holds a number that
        is not from 0 to 1; ``x`` or ``y`` is not finite; ``scale`` is not
        a positive finite number; or ``bbox`` would not be finite.

    Notes
    -----
    A placement cannot be changed once it is made: assigning to an
    attribute raises AttributeError.
    """

    component: Component
    anchor: tuple[float, float]
    x: float
    y: float
    scale: float

    # Written here rather than generated, which would give the argument and
    # the attribute one annotation: an anchor is given as a name or a pair,
    # and read as a pair. The component, which is what a placement places,
    # is taken by position too, as Scene takes it; the rest only by name.
    def __init__(
        self,
        component: Component,
        *,
        anchor: Anchor = "upper left",
        x: float = 0.0,
        y: float = 0.0,
        scale: float = 1.0,
    ) -> None:
        # Refuse a bad placement where it is written, not when it is used,
        # and keep what the checks return. The placement is frozen, so each
        # field is assigned past its __setattr__.
        if not isinstance(component, Component):
            raise TypeError(
                "component must be a Component, such as "
                "draw_svg_as_component returns; "
                f"got {type(component).__name__}."
            )
        object.__setattr__(self, "component", component)
        # A name is only a way of writing a pair of fractions, and is kept
        # as that pair, so that one point is one anchor however it is given.
        object.__setattr__(self, "anchor", _fractions(anchor))
        object.__setattr__(self, "x", check_finite("x", x))
        object.__setattr__(self, "y", check_finite("y", y))
        object.__setattr__(
            self, "scale", check_finite_positive("scale", scale)
        )
        # A box past the largest float, as a very large scale makes, is
        # infinite or NaN.
        placed = self.bbox
        corners = (placed.xmin, placed.ymin, placed.xmax, placed.ymax)
        if not is_empty_bbox(placed) and not all(map(math.isfinite, corners)):
            bbox = component.bbox
            raise ValueError(
                f"A component {bbox.width!r} wide and {bbox.height!r} tall, "
                f"placed at x={self.x!r} and y={self.y!r} with "
                f"scale={self.scale!r}, would not have a finite box."
            )

    @property
    def bbox(self) -> BBox:
        """Extent of the component once placed."""
        # The component's own box, as distinct from the placed one returned.
        component_bbox = self.component.bbox
        if is_empty_bbox(component_bbox):
            return component_bbox
        fx, fy = self.anchor
        width = component_bbox.width * self.scale
        height = component_bbox.height * self.scale
        left, top = self.x - fx * width, self.y - fy * height
        return make_bbox(left, top, left + width, top + height)


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
    usually done by its center, so neither default is obvious here.
    """
    # An empty box has no width or height, so this refuses one too.
    bbox = component.bbox
    if bbox.width <= 0 or bbox.height <= 0:
        raise ValueError("Cannot fit a component that has no area.")
    if slot.width <= 0 or slot.height <= 0:
        raise ValueError("Cannot fit a component into a slot that has no area.")

    scale = min(slot.width / bbox.width, slot.height / bbox.height)
    fx, fy = _fractions(anchor)
    # The space the slot leaves over, in the component's own units, goes
    # before and after the component in the proportions anchor gives.
    spare_x = slot.width / scale - bbox.width
    spare_y = slot.height / scale - bbox.height
    padded = make_bbox(
        bbox.xmin - fx * spare_x,
        bbox.ymin - fy * spare_y,
        bbox.xmax + (1 - fx) * spare_x,
        bbox.ymax + (1 - fy) * spare_y,
    )
    return Placement(
        make_component(
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
