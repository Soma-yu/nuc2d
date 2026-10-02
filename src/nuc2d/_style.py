"""Drawing style definitions for secondary structure rendering.

This module provides classes for configuring the visual appearance of
rendered secondary structure diagrams. Style parameters control
the appearance of graphical elements such as nucleotide nodes,
backbone and base-pair edges, their letters, and the colormap
probabilities are shown in.

The main class, StructureStyle, stores the parameters the renderer draws
with.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field

import matplotlib as mpl

from ._validation import (
    check_attribute,
    check_finite_non_negative,
    check_font_family,
    exact_str,
)


# The color keywords SVG 1.1 defines. They are CSS's named colors as they
# stood then; rebeccapurple came later and is not one of them.
_COLOR_KEYWORDS = frozenset(mpl.colors.CSS4_COLORS) - {"rebeccapurple"}
_HEX_COLOR = re.compile(r"#[0-9A-Fa-f]{6}")

# One or more non-negative numbers, such as 1 or 0.5, separated by commas.
_NUMBER = r"[0-9]+(?:\.[0-9]+)?"
_DASHARRAY = re.compile(rf"{_NUMBER}(?:,{_NUMBER})*")


def _check_color(name: str, value: object) -> str:
    """Raise unless ``value`` is written the way SVG writes a color.

    The color is returned.
    """
    if not isinstance(value, str):
        raise TypeError(
            f"{name} must be a string, such as 'black'; "
            f"got {type(value).__name__}."
        )
    color = exact_str(value)
    if not (
        color == "none"
        or color in _COLOR_KEYWORDS
        or _HEX_COLOR.fullmatch(color)
    ):
        raise ValueError(
            f"{name} must be an SVG color: a name such as 'black', "
            f"'#rrggbb', or 'none'; got {value!r}."
        )
    return color


def _check_dasharray(name: str, value: object) -> str:
    """Raise unless ``value`` is 'none' or a list of dash and gap lengths.

    The pattern is returned.
    """
    if not isinstance(value, str):
        raise TypeError(
            f"{name} must be a string, such as 'none' or '1,1'; "
            f"got {type(value).__name__}."
        )
    pattern = exact_str(value)
    if not (pattern == "none" or _DASHARRAY.fullmatch(pattern)):
        raise ValueError(
            f"{name} must be 'none', or non-negative numbers separated by "
            f"commas, such as '1,1'; got {value!r}."
        )
    return pattern


def _check_colormap(name: str, value: object) -> mpl.colors.Colormap:
    """Raise unless ``value`` is a matplotlib colormap.

    A name such as ``"turbo"`` is what matplotlib's own functions take,
    so it is the likeliest mistake, and the message names that colormap.
    The colormap is returned.
    """
    if isinstance(value, mpl.colors.Colormap):
        return value
    if isinstance(value, str):
        example, got = value, f"the string {value!r}"
    else:
        example, got = "turbo", type(value).__name__
    raise TypeError(
        f"{name} must be a matplotlib Colormap, such as "
        f"mpl.colormaps[{example!r}]; got {got}."
    )


# How each field is checked, whenever it is assigned. Every field is here,
# so that a name that is not is a misspelt one. A check returns what the
# field keeps.
_FIELD_CHECKS: dict[str, Callable[[str, object], object]] = {
    "backbone_color": _check_color,
    "backbone_width": check_finite_non_negative,
    "backbone_dasharray": _check_dasharray,
    "three_prime_arrow_length": check_finite_non_negative,
    "basepair_color": _check_color,
    "basepair_width": check_finite_non_negative,
    "basepair_dasharray": _check_dasharray,
    "node_color": _check_color,
    "node_radius": check_finite_non_negative,
    "node_font_size": check_finite_non_negative,
    "font_family": check_font_family,
    "colormap": _check_colormap,
}


@dataclass(kw_only=True)
class StructureStyle:
    """How a structure looks, and the colors its probabilities are shown in.

    A colorbar is the key to those colors, so it is drawn from the style
    of the structure it belongs to, and takes the colormap and the font
    family from it. How the colorbar itself is laid out is not a style
    setting.

    Where the nucleotides go is not a style setting either: a layout
    engine, such as :class:`RadialLayoutEngine`, places them, and the style
    says how they and the edges between them are drawn.

    Attributes
    ----------
    backbone_color : str
        Color of backbone edges. The arrow at each 3' terminus continues
        the backbone, so it is drawn in this color too.
    backbone_width : float
        Stroke width used for backbone edges.
    backbone_dasharray : str
        Dash pattern used for backbone edges, specified as an SVG
        ``stroke-dasharray`` value. The default, ``"none"``, draws them
        solid.
    three_prime_arrow_length : float
        Length of the arrow drawn at each 3' terminus. The arrowhead is
        measured in stroke widths, so its size follows
        ``backbone_width`` rather than this.

    basepair_color : str
        Color of base-pair edges.
    basepair_width : float
        Stroke width used for base-pair edges.
    basepair_dasharray : str
        Dash pattern used for base-pair edges, specified as an SVG
        ``stroke-dasharray`` value.

    node_color : str
        Color a node is drawn in when no base-pair probabilities are
        given. With ``basepair_probabilities``, a node takes its color from
        ``colormap`` instead.
    node_radius : float
        Radius of nucleotide nodes.
    node_font_size : float
        Font size of the base letter drawn inside a node.

    font_family : str
        Font family of the letters: those inside the nodes, and those of a
        colorbar drawn with this style. One family name, not a CSS
        list: ``"Arial"``, not ``"Arial, sans-serif"``. The font is looked
        up on the machine doing the drawing, and its metrics decide where
        the letters sit, so a drawing can differ between machines.
        Arial is recommended for consistent rendering in PowerPoint.
    colormap : mpl.colors.Colormap, default=mpl.colormaps["turbo"]
        Colormap a probability from 0 to 1 is shown in, on the nodes and
        on a colorbar drawn with this style. It is a matplotlib colormap
        itself, such as ``mpl.colormaps["turbo"]``, not the name of one.

    Raises
    ------
    ValueError
        If a color or a dash pattern is not written as described below, a
        size is negative or not finite, or ``font_family`` is not the name
        of one font family: empty or blank, holding a line break, a tab or
        another control character, or a list of families. Every field is
        checked when the style is made and whenever it is assigned, so a
        mistake is reported where it is written rather than when a drawing
        is made from it.
    TypeError
        If a color, a dash pattern or ``font_family`` is not a string, a
        size is not a number, or ``colormap`` is not a matplotlib
        colormap.
    AttributeError
        If an attribute that a style does not have is assigned, such as a
        misspelt one.

    Notes
    -----
    A color is written as SVG writes one: one of the color names SVG
    defines, such as ``"black"``; ``"#rrggbb"``; or ``"none"``.
    Matplotlib's own spellings, such as ``"tab:blue"`` or ``"C0"``, are not
    SVG and are refused.

    A dash pattern is ``"none"``, or one or more lengths separated by
    commas, such as ``"1,1"``: the length of a dash, of the gap after it,
    and so on, repeated along the edge. An odd number of lengths is taken
    twice, so ``"3"`` is ``"3,3"``. A length is a non-negative number such
    as ``1`` or ``0.5``, in the units of the drawing.

    Every size in the style, the two widths, ``three_prime_arrow_length``,
    ``node_radius`` and ``node_font_size``, is a finite number of at least
    0, in the units of the drawing.

    A structure's box leaves a margin around the centres of its outermost
    nucleotides, whatever the style. A ``node_radius``,
    ``node_font_size`` or ``three_prime_arrow_length`` large enough to
    draw past it is cut off at the edge of a scene.

    Two styles are equal when their fields are. Colors and dash patterns
    are compared as they are written, so ``"black"`` and ``"#000000"`` are
    not equal, and colormaps as matplotlib compares them.
    """
    backbone_color: str = "black"
    backbone_width: float = 2.0
    backbone_dasharray: str = "none"
    three_prime_arrow_length: float = 7.0

    basepair_color: str = "black"
    basepair_width: float = 1.5
    basepair_dasharray: str = "1,1"

    node_color: str = "black"
    node_radius: float = 4.2
    node_font_size: float = 6.5

    font_family: str = "Arial"
    colormap: mpl.colors.Colormap = field(
        default_factory=lambda: mpl.colormaps["turbo"]
    )

    def __setattr__(self, name: str, value: object) -> None:
        # The generated __init__ assigns every field through here as well,
        # so one check covers both a new style and a changed one.
        check_attribute(self, name, _FIELD_CHECKS)
        super().__setattr__(name, _FIELD_CHECKS[name](name, value))
