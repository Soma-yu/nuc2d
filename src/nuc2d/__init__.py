"""Nuc2D visualizes RNA and DNA secondary structures as SVG images.

The two entry points are :func:`draw_svg`, which produces a complete SVG
drawing, and :func:`draw_component`, which produces a single component
that the caller can place into a drawing of its own. :class:`Placement`
and :func:`compose` position such components relative to one another, and
:func:`render_colorbar` supplies a colorbar to place among them.

Everything public is imported from ``nuc2d`` itself. The modules inside the
package all begin with an underscore: they are where the code lives, not
part of what a version promises.
"""

from importlib.metadata import PackageNotFoundError, version

from ._draw import draw_component, draw_svg
from ._geometry import BBox
from ._layout import RadialLayoutEngine
from ._parser import ParseError
from ._style import DrawingStyle
from ._svg import Placement, SVGComponent, compose, render_colorbar

try:
    __version__ = version("nuc2d")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0+unknown"

__all__ = [
    "BBox",
    "DrawingStyle",
    "ParseError",
    "Placement",
    "RadialLayoutEngine",
    "SVGComponent",
    "__version__",
    "compose",
    "draw_component",
    "draw_svg",
    "render_colorbar",
]

# Each public name is defined in a private module, which would otherwise show
# through wherever Python reports where a name comes from: a traceback reads
# nuc2d._parser.ParseError, and repr() and help() name nuc2d._svg. Report
# them as nuc2d's own, the address a caller imports them from.
for _name in __all__:
    if _name != "__version__":
        globals()[_name].__module__ = __name__
del _name
