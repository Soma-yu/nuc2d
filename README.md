# nuc2d

`nuc2d` is a Python library for parsing, annotating, laying out, and
rendering nucleic acid secondary structures.

![nuc2d example](https://raw.githubusercontent.com/Soma-yu/nuc2d/main/docs/images/example.png)

## Installation

```bash
pip install nuc2d
```

## Quick start

```python
from nuc2d import draw_svg

dpp_string = "(((..+...)))"

svg = draw_svg(
    dpp_string=dpp_string,
)

svg.saveas("output.svg")
```

The `dpp_string` argument should be specified in dot-parens-plus notation.

If you are using Jupyter Notebook or JupyterLab, you can also display
the generated SVG directly:

```python
from IPython.display import SVG, display

display(SVG(svg.tostring()))
```

## Sequence annotation

Nucleotide sequences can be provided through the `sequences` argument.
```python
sequences = ["AUGCA", "UGCCAU"]

svg = draw_svg(
    dpp_string=dpp_string,
    sequences=sequences,
)
```

## Base-pair probability visualization

Base-pair probabilities can be visualized by providing a symmetric
base-pair probability matrix through the `probs` argument.
```python
# Base-pair probability matrix from a structure prediction tool.
# probs[i][j] is the probability of nucleotides i and j forming a base pair.
# The diagonal probs[i][i] represents the probability that nucleotide i is unpaired.
probs = ...

svg = draw_svg(
    dpp_string=dpp_string,
    probs=probs,
)
```

## Output

`nuc2d` renders structures as SVG, making the resulting figures
suitable for further editing and use in presentations and
publications.

## License

This project is licensed under the MIT License.
