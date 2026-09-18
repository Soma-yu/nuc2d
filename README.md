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

## Versioning

Nuc2D follows [Semantic Versioning](https://semver.org/). What a version
promises is the public API: the names the package exports, the arguments they
take, and the exceptions they raise. Those change only in a major release.

The drawing is not part of that promise. A minor release may place a
nucleotide differently, enclose a structure more tightly, or write the same
shape as different SVG, so a figure regenerated under a newer version can
come out different. Text is measured with the font installed on the machine,
so a drawing can differ between two machines running the same version as
well. An SVG already saved to disk is of course unaffected.

## Changes

Release notes for every version are on the
[releases page](https://github.com/Soma-yu/nuc2d/releases).

## License

This project is licensed under the MIT License.
