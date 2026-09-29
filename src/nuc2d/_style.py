"""Drawing style definitions for secondary structure rendering.

This module provides classes for configuring the visual appearance of
rendered secondary structure diagrams. Style parameters control
the appearance of graphical elements such as nucleotide nodes,
backbone and base-pair edges, their letters, and the colormap
probabilities are shown in.

The main class, DrawingStyle, stores the parameters the renderer draws
with.
"""

import re
from dataclasses import dataclass, field

import matplotlib as mpl


# The colour keywords SVG 1.1 defines. They are CSS's named colours as they
# stood then; rebeccapurple came later and is not one of them.
_COLOR_KEYWORDS = frozenset(mpl.colors.CSS4_COLORS) - {"rebeccapurple"}
_HEX_COLOR = re.compile(r"#[0-9A-Fa-f]{3}(?:[0-9A-Fa-f]{3})?")
_RGB_INTEGER = re.compile(r"rgb\( *[0-9]+ *, *[0-9]+ *, *[0-9]+ *\)")
_RGB_PERCENTAGE = re.compile(
    r"rgb\( *[0-9]+(?:\.[0-9]*)?% *, *[0-9]+(?:\.[0-9]*)?% *,"
    r" *[0-9]+(?:\.[0-9]*)?% *\)"
)

# One or more non-negative numbers, separated by a comma or by spaces. SVG
# also allows a comma with spaces around it, but the renderer's own check
# refuses that, and accepting it here would only move the error there.
_NUMBER = r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)"
_DASHARRAY = re.compile(rf"{_NUMBER}(?:(?:,| +){_NUMBER})*")

_COLOR_FIELDS = frozenset({"node_color", "backbone_color", "basepair_color"})
_DASHARRAY_FIELDS = frozenset({"backbone_dasharray", "basepair_dasharray"})


def _check_color(name: str, value: object) -> None:
    """Raise unless ``value`` is written the way SVG writes a colour."""
    if not isinstance(value, str):
        raise TypeError(
            f"{name} must be a string, such as 'crimson' or '#333333'; "
            f"got {type(value).__name__}."
        )
    if not (
        value == "none"
        or value in _COLOR_KEYWORDS
        or _HEX_COLOR.fullmatch(value)
        or _RGB_INTEGER.fullmatch(value)
        or _RGB_PERCENTAGE.fullmatch(value)
    ):
        raise ValueError(
            f"{name} must be an SVG color: a name such as 'crimson', "
            f"'#rgb' or '#rrggbb', 'rgb(r, g, b)', or 'none'; got {value!r}."
        )


def _check_dasharray(name: str, value: object) -> None:
    """Raise unless ``value`` is 'none' or a list of dash and gap lengths."""
    if not isinstance(value, str):
        raise TypeError(
            f"{name} must be a string, such as '4,2' or 'none'; "
            f"got {type(value).__name__}."
        )
    if not (value == "none" or _DASHARRAY.fullmatch(value)):
        raise ValueError(
            f"{name} must be 'none', or dash and gap lengths separated by "
            f"commas or by spaces, such as '4,2' or '4 2'; got {value!r}."
        )


@dataclass(kw_only=True)
class DrawingStyle:
    """How a structure looks, and the colors its probabilities are shown in.

    A colorbar is the key to those colors, so it is drawn from the style
    of the structure it belongs to, and takes the colormap and the font
    family from it. How the colorbar itself is laid out is not a style
    setting.

    Attributes
    ----------
    backbone_width : float
        Stroke width used for backbone edges.
    basepair_width : float
        Stroke width used for base-pair edges.
    backbone_dasharray : str
        Dash pattern used for backbone edges, specified as an SVG
        ``stroke-dasharray`` value. The default, ``"none"``, draws them
        solid.
    basepair_dasharray : str
        Dash pattern used for base-pair edges, specified as an SVG
        ``stroke-dasharray`` value.
    three_prime_arrow_length : float
        Length of the arrow drawn at each 3' terminus. The arrowhead is
        measured in stroke widths, so its size follows
        ``backbone_width`` rather than this.
    node_radius : float
        Radius of nucleotide nodes.
    node_color : str
        Color a node is drawn in when no probabilities are given. With
        ``probs``, a node takes its color from ``colormap`` instead.
    backbone_color : str
        Color of backbone edges. The arrow at each 3' terminus continues
        the backbone, so it is drawn in this color too.
    basepair_color : str
        Color of base-pair edges.
    font_family : str
        Font family of the letters: those inside the nodes, and those of a
        colorbar drawn with this style. One family name, not a CSS
        list: ``"Arial"``, not ``"Arial, sans-serif"``. The font is looked
        up on the machine doing the drawing, and its metrics decide where
        the letters sit, so a drawing can differ between machines.
        Arial is recommended for consistent rendering in PowerPoint.
    node_font_size : float
        Font size of the base letter drawn inside a node.
    colormap : mpl.colors.Colormap, default=mpl.colormaps["turbo"]
        Colormap a probability from 0 to 1 is shown in, on the nodes and
        on a colorbar drawn with this style.

    Raises
    ------
    ValueError
        If a color or a dash pattern is not written as described below.
        The check runs when the style is made and whenever one of those
        fields is assigned, so a mistake is reported where it is written
        rather than when a drawing is made from it.
    TypeError
        If a color or a dash pattern is not a string.

    Notes
    -----
    A color is written as SVG writes one: one of the color names SVG
    defines, such as ``"crimson"``; ``"#rgb"`` or ``"#rrggbb"``;
    ``"rgb(r, g, b)"`` with integers or percentages; or ``"none"``.
    Matplotlib's own spellings, such as ``"tab:blue"`` or ``"C0"``, are not
    SVG and are refused.

    A dash pattern is ``"none"``, or dash and gap lengths separated by
    commas or by spaces, such as ``"4,2"`` or ``"4 2"``, but not both at
    once. Lengths are plain non-negative numbers, in the units of the
    drawing.

    A structure's box runs 20 units past the centres of its outermost
    nucleotides, whatever the style. A ``node_radius``,
    ``node_font_size`` or ``three_prime_arrow_length`` large enough to
    draw past that is cut off at the edge of a scene.
    """
    backbone_width: float = 2.0
    basepair_width: float = 1.5
    backbone_dasharray: str = "none"
    basepair_dasharray: str = "1,1"

    three_prime_arrow_length: float = 7.0

    node_radius: float = 4.2


    node_color: str = "black"
    backbone_color: str = "black"
    basepair_color: str = "black"

    font_family: str = "Arial"
    node_font_size: float = 6.5

    colormap: mpl.colors.Colormap = field(
        default_factory=lambda: mpl.colormaps["turbo"]
    )

    def __setattr__(self, name: str, value: object) -> None:
        # The generated __init__ assigns every field through here as well,
        # so one check covers both a new style and a changed one.
        if name in _COLOR_FIELDS:
            _check_color(name, value)
        elif name in _DASHARRAY_FIELDS:
            _check_dasharray(name, value)
        super().__setattr__(name, value)
