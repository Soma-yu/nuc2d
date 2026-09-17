"""Nuc2D visualizes RNA and DNA secondary structures as SVG images.

The two entry points are :func:`draw_svg`, which produces a complete SVG
drawing, and :func:`draw_component`, which produces a single component
that the caller can place into a drawing of its own. :class:`Placement`
and :func:`compose` position such components relative to one another, and
:func:`render_colorbar` supplies a colorbar to place among them.
"""

from importlib.metadata import PackageNotFoundError, version

from .draw import draw_component, draw_svg
from .geometry import BBox, Vec2
from .layout import LayoutEngine, RadialLayoutEngine
from .parser import ParseError
from .style import DrawingStyle
from .svg import Placement, SVGComponent, compose, render_colorbar

try:
    __version__ = version("nuc2d")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0+unknown"

__all__ = [
    "BBox",
    "DrawingStyle",
    "LayoutEngine",
    "ParseError",
    "Placement",
    "RadialLayoutEngine",
    "SVGComponent",
    "Vec2",
    "__version__",
    "compose",
    "draw_component",
    "draw_svg",
    "render_colorbar",
]
