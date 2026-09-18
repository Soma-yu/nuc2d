# Nuc2D

Nuc2D visualizes RNA and DNA secondary structures as publication-ready SVG
images. The output stays sharp at any size and remains editable in tools such
as Illustrator or Inkscape, so a figure can be adjusted without being redrawn.

<p align="center">
  <img src="https://raw.githubusercontent.com/Soma-yu/nuc2d/main/docs/images/example.png" width="80%">
</p>

## Installation

```bash
pip install nuc2d
```

## Quick start

```python
from nuc2d import draw_svg

drawing = draw_svg("(((..+...)))")

drawing.saveas("output.svg")
```

Structures are written in dot-parens-plus notation: `(` and `)` for the two
halves of a base pair, `.` for an unpaired nucleotide, and `+` for a break
between strands.

In Jupyter Notebook or JupyterLab the result can be displayed directly:

```python
from IPython.display import SVG, display

display(SVG(drawing.tostring()))
```

An input that is not a well-formed structure raises `ParseError`:

```python
from nuc2d import ParseError

try:
    draw_svg("(((")
except ParseError as error:
    print(error)
```

## Sequence annotation

Nucleotide sequences can be provided through the `sequences` argument, one per
strand, in the order the strands appear in the structure.

```python
drawing = draw_svg(
    "(((..+...)))",
    sequences=["AUGCA", "UGCCAU"],
)
```

A wrong number of sequences, or a sequence that is not as long as its strand,
raises `ValueError` rather than drawing something misleading.

## Base-pair probability visualization

Base-pair probabilities are visualized by passing a symmetric probability
matrix through the `probs` argument. A colorbar is placed beside the structure.

```python
# Base-pair probability matrix from a structure prediction tool.
# probs[i][j] is the probability of nucleotides i and j forming a base pair.
# The diagonal probs[i][i] is the probability that nucleotide i is unpaired.
probs = ...

drawing = draw_svg(
    "(((..+...)))",
    probs=probs,
)
```

The colorbar is labelled `Base-pair probability` unless another label is given:

```python
drawing = draw_svg(
    "(((..+...)))",
    probs=probs,
    colorbar_label="Pairing probability",
)
```

## Output size

```python
drawing = draw_svg("(((..+...)))", width_px=600)
```

Giving `width_px` or `height_px` alone lets the other follow from the aspect
ratio of the drawing. Giving neither defaults the height to 500 px.

## Style and layout

`DrawingStyle` controls appearance — colors, stroke widths, node size, fonts,
and the colormap used for probabilities. `RadialLayoutEngine` controls
geometry — how far apart nucleotides are placed.

```python
import matplotlib as mpl

from nuc2d import DrawingStyle, RadialLayoutEngine, draw_svg

drawing = draw_svg(
    "(((..+...)))",
    style=DrawingStyle(
        node_fill="steelblue",
        edge_color="dimgray",
        node_radius=5.0,
        cmap=mpl.colormaps["viridis"],
    ),
    layout_engine=RadialLayoutEngine(
        backbone_spacing=20.0,
        loop_spacing=25.0,
    ),
)
```

The defaults are on the left, the settings above on the right.

<p align="center">
  <img src="https://raw.githubusercontent.com/Soma-yu/nuc2d/main/docs/images/styling.png" width="80%">
</p>

## Combining several structures

`draw_component` renders one structure into an SVG component without deciding
where it goes, so several structures can share a single drawing. Each component
carries the bounding box it occupies, and `compose` collects placed components
into one group whose bounding box encloses them all.

```python
import svgwrite

from nuc2d import Placement, compose, draw_component

drawing = svgwrite.Drawing()

components = [
    draw_component(drawing, "(((...)))"),
    draw_component(drawing, "((..((...))..))"),
    draw_component(drawing, "((((....))))"),
]

# Lay the structures out in a row, with a gap between them.
placements = []
x = 0.0
for component in components:
    placements.append(
        Placement(component=component, x=x - component.bbox.xmin, y=0.0, scale=1.0)
    )
    x += component.bbox.width + 10.0

panel = compose(drawing.g(), placements)

drawing.add(panel.group)
drawing.viewbox(*panel.bbox.to_viewbox())
drawing.saveas("panel.svg")
```

<p align="center">
  <img src="https://raw.githubusercontent.com/Soma-yu/nuc2d/main/docs/images/composing.png" width="80%">
</p>

## Changes in 0.7.0

Optional arguments are now keyword-only. The arguments a call is about stay
positional — the structure for `draw_svg`, the drawing and the structure for
`draw_component` — and everything else is passed by name.

```python
draw_svg("(((..+...)))", ["AUGCA", "UGCCAU"])            # 0.6.0
draw_svg("(((..+...)))", sequences=["AUGCA", "UGCCAU"])  # 0.7.0
```

Every type the package defines is built by name too, with two exceptions:
`Vec2` and `BBox`, whose numbers are written out in order as in any other
geometry library. Code that already passes these by name, as the examples
above do, is unaffected.

```python
ArcEdge(start, end, EdgeType.BACKBONE, r, r, 0, 0, 1)  # 0.6.0
ArcEdge(                                               # 0.7.0
    start=start, end=end, edge_type=EdgeType.BACKBONE,
    rx=r, ry=r, x_axis_rotation=0, large_arc=False, sweep=True,
)
```

This is the change that lets the ones after it be additions: a new argument can
go where it belongs, instead of being appended to leave the existing order
intact.

`Edge.type` is now `Edge.edge_type`. Keyword-only construction makes a field
name the only way to reach it, which is a reason not to leave one sharing a
name with a builtin.

Three more names change, for the same reason that a name is now the whole
interface.

| 0.6.0 | 0.7.0 | |
|---|---|---|
| `DrawingStyle.colorbar_width_ratio` | `DrawingStyle.colorbar_aspect_ratio` | The same number, `1/30`, under a name that says what it is the ratio of: the bar's width over its height, as CSS defines an aspect ratio. |
| `compose(container, ...)` | `compose(group, ...)` | The argument is the group the composed component is returned with, and the package calls one of those a group everywhere else. |
| `RadialLayoutEngine(pair_width=...)` | `RadialLayoutEngine(pair_spacing=...)` | A distance between two nucleotides, like `backbone_spacing` and `loop_spacing`. `DrawingStyle.basepair_width`, a stroke width, keeps its name. |

`SVGComponent` and `Placement` no longer carry `width` and `height`. Both
still carry `bbox`, which reports where the component sits and, through its
own `width` and `height`, how large it is.

```python
component.width        # 0.6.0
component.bbox.width   # 0.7.0

placement.height       # 0.6.0
placement.bbox.height  # 0.7.0
```

The numbers are the same; only the spelling changes. Code that already reads
`component.bbox.width`, as the examples above do, is unaffected.

## Changes in 0.6.0

`draw_svg` and `draw_component` accept `add_colorbar=False`, which colors the
nucleotides from `probs` but leaves the colorbar out. It is for placing a
colorbar of your own with the newly exported `render_colorbar`: the one these
functions place is as tall as the structure, which leaves it small beside a
structure much wider than it is tall.

Nothing else changes, and neither does the output unless `add_colorbar` is used.

## Changes in 0.5.0

Version 0.5.0 rejects two kinds of string that earlier versions drew.

- Strands that no base pair connects, such as `...+...` or `((...))+((...))`,
  raise `ParseError`. A secondary structure describes one complex, and
  strands nothing holds together are separate molecules that happen to share
  a string.
- Hairpin loops of fewer than three nucleotides, such as `(..)`, raise
  `ParseError`. A backbone cannot turn back on itself in fewer, which is the
  same minimum structure prediction tools impose.

Structures that came from a prediction tool are unaffected: neither shape can
occur in one.

## Changes in 0.4.0

Version 0.4.0 changes the public API. Existing code written against 0.3.0 needs
the following adjustments.

- `draw_group` is now `draw_component`. It returns a single `SVGComponent`
  instead of a `(Group, BoundingBox)` tuple; use `component.group` and
  `component.bbox`.
- `BoundingBox(xmin, ymin, width, height)` is now `BBox(xmin, ymin, xmax,
  ymax)`, with `width` and `height` as derived properties. A component's
  bounding box now reports where the component actually sits, instead of
  always starting at the origin.
- `draw_svg` and `draw_component` accept `layout_engine` and `colorbar_label`.
  On `draw_svg` these come before `width_px` and `height_px`, so any call that
  passes those two positionally needs updating.
- Malformed structures raise `ParseError`, and sequences or probability
  matrices that do not match the structure raise `ValueError`. Both previously
  surfaced as `IndexError`, or as a silently wrong drawing.
- `DrawingStyle.colorbar_spacing` is gone; it never affected the output.

## License

This project is licensed under the MIT License.
