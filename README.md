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

scene = draw_svg(dot_bracket=CLOVERLEAF)

scene.save_svg("output.svg")
```

<p align="center">
  <img src="https://raw.githubusercontent.com/Soma-yu/nuc2d/main/docs/images/structure.png" width="55%">
</p>

Structures are written with `(` and `)` for the two halves of a base pair,
`.` for an unpaired nucleotide, and `+` for a break between strands. An arrow
marks each 3' terminus.

In Jupyter, a scene that ends a cell is displayed as it is:

```python
scene
```

Anywhere else in a cell, `display(scene)` from `IPython.display` shows it.

An input that is not a well-formed structure raises `ParseError`:

```python
from nuc2d import ParseError

try:
    draw_svg("(((")
except ParseError as error:
    print(error)
```

`ParseError` is a `ValueError`, so `except ValueError` catches it too, along
with the mismatched sequences and probabilities described below.

## Sequence annotation

Nucleotide sequences can be provided through the `sequences` argument, one per
strand, in the order the strands appear in the structure.

```python
SEQUENCES = [
    "GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGA",
    "UCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCACCA",
]

scene = draw_svg(dot_bracket=CLOVERLEAF, sequences=SEQUENCES)
```

<p align="center">
  <img src="https://raw.githubusercontent.com/Soma-yu/nuc2d/main/docs/images/sequences.png" width="55%">
</p>

A wrong number of sequences, or a sequence that is not as long as its strand,
raises `ValueError` rather than drawing something misleading.

## Equilibrium probability visualization

Base-pair probabilities are visualized by passing a symmetric probability
matrix through the `probs` argument. A colorbar is placed beside the structure,
which is fitted into a square as tall as the colorbar, so that the colorbar
keeps its size whatever the shape of the structure.

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

scene = draw_svg(dot_bracket=CLOVERLEAF, sequences=SEQUENCES, probs=probs)
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
scene = draw_svg(
    dot_bracket=CLOVERLEAF,
    probs=probs,
    colorbar_label="Pairing probability",
)
```

`colorbar_label=None` leaves the colorbar without a label.

## Output size

```python
# One of the two: the other follows from the aspect ratio of the structure.
scene = draw_svg(dot_bracket=CLOVERLEAF, width_px=600)

# Both: used as written.
scene = draw_svg(dot_bracket=CLOVERLEAF, width_px=600, height_px=600)
```

Giving neither defaults the height to 500 px. Giving both keeps the structure's
own proportions and centres it in the box, with space above and below or at
the sides, rather than stretching it to fit.

## Style and layout

`DrawingStyle` controls appearance — colors, stroke widths, node size, fonts,
and the colormap used for probabilities. `RadialLayoutEngine` controls
geometry — how far apart nucleotides are placed.

```python
import matplotlib as mpl

from nuc2d import DrawingStyle, RadialLayoutEngine

scene = draw_svg(
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

Colors are written as SVG writes them: a name such as `crimson`, `#rgb` or
`#rrggbb`, `rgb(r, g, b)`, or `none`. Dash patterns are `none` or lengths such
as `4,2`. Anything else raises `ValueError` as soon as it is set, rather than
when something is drawn with the style.

## Combining several structures

`draw_svg` draws one structure and frames it as a `Scene`. To put several
parts in one picture, draw each as a component instead, place the components,
and frame the result:

- `draw_structure` takes the same arguments as `draw_svg`, less the size, and
  returns the structure, with its colorbar if it has one, as a `Component`.
  `draw_svg(...)` is `Scene(draw_structure(...))`.
- `draw_colorbar` draws a colorbar on its own, for a structure drawn with
  `add_colorbar=False`.
- `Placement` says where a component goes and at what size, and `compose`
  gathers placed components into a new one.
- `Scene` frames a component, and is what is saved or shown.

This is how the picture at the top of this page is drawn:

```python
from nuc2d import Placement, Scene, compose, draw_structure

components = [
    draw_structure(CLOVERLEAF),
    draw_structure(CLOVERLEAF, sequences=SEQUENCES, probs=probs),
]

# Lay the components out in a row, all as tall as the first, with a gap between.
height = components[0].bbox.height
placements, cursor_x = [], 0.0
for component in components:
    placement = Placement(
        component=component, x=cursor_x, scale=height / component.bbox.height
    )
    placements.append(placement)
    cursor_x = placement.bbox.xmax + 20.0

Scene(compose(placements)).save_svg("panel.svg")
```

<p align="center">
  <img src="https://raw.githubusercontent.com/Soma-yu/nuc2d/main/docs/images/example.png" width="100%">
</p>

`Placement` puts one point of a component at `x` and `y`, and scales the
component about that point. The point is the upper left corner of the
component's box unless `anchor` names another: one of `"upper left"`,
`"upper center"`, `"upper right"`, `"center left"`, `"center"`,
`"center right"`, `"lower left"`, `"lower center"` and `"lower right"`, or a
pair of fractions of the box's width and height, such as `(0.5, 0.0)` for the
middle of its top edge.

## Versioning

Nuc2D follows [Semantic Versioning](https://semver.org/). What a version
promises is the public API: the names the package exports, the arguments they
take, and the exceptions they raise. Those change only in a major release.

Everything public is imported from `nuc2d` itself. The modules inside the
package all begin with an underscore, such as `nuc2d._svg`: they are where the
code lives, and they can change in any release.

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
