# Nuc2D: Visualize RNA and DNA secondary structures

The output is SVG, so a figure stays sharp at any size and remains editable in
tools such as Illustrator or Inkscape: it can be adjusted without being
redrawn.

<p align="center">
  <img src="https://raw.githubusercontent.com/Soma-yu/nuc2d/main/docs/images/example.png" width="100%">
</p>

One structure, drawn twice: on its own, and with its sequences and the
probability of each nucleotide being in the state the structure puts it in.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Soma-yu/nuc2d/blob/main/examples/nuc2d_intro.ipynb)

An introductory notebook, written in Japanese. It runs in Google Colab, so
there is nothing to set up on your own machine.

## Installation

```bash
pip install nuc2d
```

## Quick start

```python
from nuc2d import draw_svg

# A tRNA cloverleaf, split into two strands.
CLOVERLEAF = "(((((((..((((........)))).(((((.......+))))).....(((((.......))))))))))))...."

drawing = draw_svg(dot_bracket=CLOVERLEAF)

drawing.saveas("output.svg")
```

<p align="center">
  <img src="https://raw.githubusercontent.com/Soma-yu/nuc2d/main/docs/images/structure.png" width="55%">
</p>

Structures are written with `(` and `)` for the two halves of a base pair,
`.` for an unpaired nucleotide, and `+` for a break between strands. An arrow
marks each 3' terminus.

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
SEQUENCES = [
    "GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGA",
    "UCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCACCA",
]

drawing = draw_svg(dot_bracket=CLOVERLEAF, sequences=SEQUENCES)
```

<p align="center">
  <img src="https://raw.githubusercontent.com/Soma-yu/nuc2d/main/docs/images/sequences.png" width="55%">
</p>

A wrong number of sequences, or a sequence that is not as long as its strand,
raises `ValueError` rather than drawing something misleading.

## Equilibrium probability visualization

Base-pair probabilities are visualized by passing a symmetric probability
matrix through the `probs` argument. A colorbar is placed beside the structure.

`probs[i][j]` is how likely nucleotides `i` and `j` are to be paired with each
other, and `probs[i][i]` how likely nucleotide `i` is to be left unpaired. A
real matrix comes from a structure prediction tool; the one below is made up,
which is enough to see what the drawing does.

```python
import numpy as np

flat = CLOVERLEAF.replace("+", "")
probs = np.zeros((len(flat), len(flat)))

stack = []
for i, char in enumerate(flat):
    if char == "(":
        stack.append(i)
    elif char == ")":
        left = stack.pop()
        # Made up: the acceptor and anticodon stems are the certain ones.
        probs[left][i] = probs[i][left] = 0.9 if left < 8 or 25 < left < 32 else 0.45

# Whatever is left over is the probability of staying unpaired.
probs[np.diag_indices_from(probs)] = 1.0 - probs.sum(axis=1)

drawing = draw_svg(dot_bracket=CLOVERLEAF, sequences=SEQUENCES, probs=probs)
```

<p align="center">
  <img src="https://raw.githubusercontent.com/Soma-yu/nuc2d/main/docs/images/probabilities.png" width="65%">
</p>

Each nucleotide is colored by the probability of the state the structure puts
it in: of pairing with its partner if it is paired, and of being unpaired if it
is not.

The colorbar is labelled `Equilibrium probability` unless another label is
given:

```python
drawing = draw_svg(
    dot_bracket=CLOVERLEAF,
    probs=probs,
    colorbar_label="Pairing probability",
)
```

## Output size

```python
# One of the two: the other follows from the aspect ratio of the drawing.
drawing = draw_svg(dot_bracket=CLOVERLEAF, width_px=600)

# Both: used as written.
drawing = draw_svg(dot_bracket=CLOVERLEAF, width_px=600, height_px=600)
```

Giving neither defaults the height to 500 px. Giving both keeps the drawing's
own proportions and centres it in the box, with space above and below or at
the sides, rather than stretching it to fit.

## Style and layout

`DrawingStyle` controls appearance — colors, stroke widths, node size, fonts,
and the colormap used for probabilities. `RadialLayoutEngine` controls
geometry — how far apart nucleotides are placed.

```python
import matplotlib as mpl

from nuc2d import DrawingStyle, RadialLayoutEngine

drawing = draw_svg(
    dot_bracket=CLOVERLEAF,
    sequences=SEQUENCES,
    probs=probs,
    style=DrawingStyle(
        backbone_color="#333333",
        basepair_color="crimson",
        node_radius=5.0,
        cmap=mpl.colormaps["viridis"],
    ),
    layout_engine=RadialLayoutEngine(
        backbone_spacing=18.0,
        loop_spacing=24.0,
    ),
)
```

<p align="center">
  <img src="https://raw.githubusercontent.com/Soma-yu/nuc2d/main/docs/images/styling.png" width="65%">
</p>

## Combining several structures

`draw_component` renders one structure into an SVG component without deciding
where it goes, so several structures can share a single drawing. Each component
carries the bounding box it occupies, and `compose` collects placed components
into one group whose bounding box encloses them all.

This is how the picture at the top of this page is drawn:

```python
import svgwrite

from nuc2d import Placement, compose, draw_component

drawing = svgwrite.Drawing()

components = [
    draw_component(drawing, dot_bracket=CLOVERLEAF),
    draw_component(
        drawing, dot_bracket=CLOVERLEAF, sequences=SEQUENCES, probs=probs
    ),
]

# Lay the components out in a row, aligned on their tops, with a gap between.
placements, cursor_x = [], 0.0
for component in components:
    box = component.bbox
    placements.append(
        Placement(component=component, x=cursor_x - box.xmin, y=-box.ymin)
    )
    cursor_x += box.width + 20.0

panel = compose(drawing.g(), placements)

drawing.add(panel.group)
drawing.viewbox(*panel.bbox.to_viewbox())
drawing["width"] = f"{panel.bbox.width}px"
drawing["height"] = f"{panel.bbox.height}px"
drawing.saveas("panel.svg")
```

<p align="center">
  <img src="https://raw.githubusercontent.com/Soma-yu/nuc2d/main/docs/images/example.png" width="100%">
</p>

`Placement.x` and `Placement.y` say how far to move a component after scaling
it, so aligning an edge means subtracting the scaled edge of its bounding box.

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
