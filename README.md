# nuc2d

`nuc2d` is a Python library for parsing, annotating, laying out, and rendering nucleic acid secondary structures.

## Installation

```bash
pip install nuc2d
```

## Example

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

Optional sequence and base-pair probability annotations can be provided
through the `sequences` and `probs` arguments:

```python
svg = draw_svg(
    dpp_string=dpp_string,
    sequences=sequences,
    probs=probs,
)
```

## License

This project is licensed under the MIT License.
