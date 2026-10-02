"""Nuc2D visualizes RNA and DNA secondary structures as SVG images.

:func:`draw_svg` draws a structure and returns a :class:`Scene`, ready to
save with :meth:`Scene.save_svg` or to show in Jupyter.

To put several parts together, draw each as a :class:`Component` with
:func:`draw_structure`, :func:`draw_colorbar` or :func:`draw_text`, say
where each goes with a :class:`Placement`, make one component of them
with :meth:`Component.from_placements`, and frame the result as a
:class:`Scene`.

Everything public is imported from ``nuc2d`` itself. The modules inside the
package all begin with an underscore: they are where the code lives, not
part of what a version promises.
"""

from importlib import metadata as _metadata

from ._component import Component, Placement
from ._draw import draw_colorbar, draw_structure, draw_svg, draw_text
from ._geometry import BBox
from ._layout import RadialLayoutEngine
from ._parse import ParseError
from ._scene import Scene
from ._style import StructureStyle

try:
    __version__ = _metadata.version("nuc2d")
except _metadata.PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0+unknown"

__all__ = [
    "BBox",
    "Component",
    "ParseError",
    "Placement",
    "RadialLayoutEngine",
    "Scene",
    "StructureStyle",
    "__version__",
    "draw_colorbar",
    "draw_structure",
    "draw_svg",
    "draw_text",
]

# Each public name is defined in a private module, which would otherwise show
# through wherever Python reports where a name comes from: a traceback reads
# nuc2d._parse.ParseError, and repr() and help() name nuc2d._svg. Report
# them as nuc2d's own, the address a caller imports them from.
for _name in __all__:
    if _name != "__version__":
        globals()[_name].__module__ = __name__
del _name
