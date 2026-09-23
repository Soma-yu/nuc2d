"""Regenerate the images the README shows.

Run after any change that affects rendering, so the pictures on the front
page keep matching the code they illustrate::

    uv run python docs/make_images.py

Each image is written to ``docs/images/`` as both SVG and PNG. The structure,
the sequences and the probability matrix here are the ones the README's own
examples build, so the pictures are what its code produces.
"""

from pathlib import Path

import cairosvg
import matplotlib as mpl
import numpy as np
import svgwrite

from nuc2d import (
    DrawingStyle,
    Placement,
    RadialLayoutEngine,
    SVGComponent,
    compose,
    draw_component,
    draw_svg,
)

DOCS = Path(__file__).parent
IMAGES = DOCS / "images"

# The PNG is what the README displays, so it is sized for a high-density
# screen. The SVG keeps whatever size the README's own code gives it.
PNG_WIDTH = 1600

# A tRNA cloverleaf split into two strands, so that one example shows the
# strand break as well. The sequence is yeast tRNA-Phe.
CLOVERLEAF = "(((((((..((((........)))).(((((.......+))))).....(((((.......))))))))))))...."
SEQUENCES = [
    "GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGA",
    "UCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCACCA",
]


def demo_probs(dot_bracket: str) -> np.ndarray:
    """Build a probability matrix from a structure.

    A real matrix comes from a structure prediction tool. This one is made
    up, and is the same construction the README shows, so that the pictures
    written here are the ones its code produces.
    """
    flat = dot_bracket.replace("+", "")
    probs = np.zeros((len(flat), len(flat)))

    stack: list[int] = []
    for i, char in enumerate(flat):
        if char == "(":
            stack.append(i)
        elif char == ")":
            left = stack.pop()
            probs[left][i] = probs[i][left] = 0.9 if left < 8 or 25 < left < 32 else 0.45

    probs[np.diag_indices_from(probs)] = 1.0 - probs.sum(axis=1)
    return probs


PROBS = demo_probs(CLOVERLEAF)


def row(
    drawing: svgwrite.Drawing,
    components: list[SVGComponent],
    gap: float,
) -> svgwrite.Drawing:
    """Lay components out left to right, aligned on their tops."""
    placements = []
    cursor_x = 0.0
    for component in components:
        box = component.bbox
        placements.append(
            Placement(component=component, x=cursor_x - box.xmin, y=-box.ymin)
        )
        cursor_x += box.width + gap

    panel = compose(drawing.g(), placements)

    drawing.add(panel.group)
    drawing.viewbox(*panel.bbox.to_viewbox())
    drawing["width"] = f"{panel.bbox.width}px"
    drawing["height"] = f"{panel.bbox.height}px"
    return drawing


def write(drawing: svgwrite.Drawing, name: str) -> None:
    """Write one drawing as both SVG and PNG."""
    svg_path = IMAGES / f"{name}.svg"
    drawing.saveas(str(svg_path))
    cairosvg.svg2png(
        url=str(svg_path),
        write_to=str(IMAGES / f"{name}.png"),
        output_width=PNG_WIDTH,
        background_color="white",
    )
    print(f"wrote {name}.svg and {name}.png")


def example() -> svgwrite.Drawing:
    """The drawing at the top of the README: the same structure three times."""
    drawing = svgwrite.Drawing()

    components = [
        draw_component(drawing, dot_bracket=CLOVERLEAF),
        draw_component(drawing, dot_bracket=CLOVERLEAF, sequences=SEQUENCES),
        draw_component(
            drawing, dot_bracket=CLOVERLEAF, sequences=SEQUENCES, probs=PROBS
        ),
    ]

    return row(drawing, components, gap=40.0)


def structure() -> svgwrite.Drawing:
    """The quick start: a structure on its own."""
    return draw_svg(dot_bracket=CLOVERLEAF)


def sequences() -> svgwrite.Drawing:
    """Sequence annotation: the same structure with its bases."""
    return draw_svg(dot_bracket=CLOVERLEAF, sequences=SEQUENCES)


def probabilities() -> svgwrite.Drawing:
    """Equilibrium probabilities: the same structure, coloured."""
    return draw_svg(dot_bracket=CLOVERLEAF, sequences=SEQUENCES, probs=PROBS)


def styling() -> svgwrite.Drawing:
    """The settings the README's style example uses."""
    return draw_svg(
        dot_bracket=CLOVERLEAF,
        sequences=SEQUENCES,
        probs=PROBS,
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


def composing() -> svgwrite.Drawing:
    """The three-structure panel from the README."""
    drawing = svgwrite.Drawing()

    components = [
        draw_component(drawing, dot_bracket="(((...)))"),
        draw_component(drawing, dot_bracket="((..((...))..))"),
        draw_component(drawing, dot_bracket="((((....))))"),
    ]

    return row(drawing, components, gap=10.0)


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    write(example(), "example")
    write(structure(), "structure")
    write(sequences(), "sequences")
    write(probabilities(), "probabilities")
    write(styling(), "styling")
    write(composing(), "composing")


if __name__ == "__main__":
    main()
