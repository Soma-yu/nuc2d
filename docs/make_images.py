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

from nuc2d import (
    Component,
    Placement,
    RadialLayoutEngine,
    Scene,
    StructureStyle,
    draw_svg,
    draw_svg_as_component,
)

DOCS = Path(__file__).parent
IMAGES = DOCS / "images"

# The PNG is what the README displays, so it is sized for a high-density
# screen. The SVG keeps whatever size the README's own code gives it.
PNG_WIDTH = 2400

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

    return probs


PROBS = demo_probs(CLOVERLEAF)


def row(components: list[Component], gap: float) -> Scene:
    """Lay components out left to right, all as tall as the first."""
    height = components[0].bbox.height
    placements = []
    cursor_x = 0.0
    for component in components:
        placement = Placement(
            component, x=cursor_x, scale=height / component.bbox.height
        )
        placements.append(placement)
        cursor_x = placement.bbox.xmax + gap

    return Scene(Component.from_placements(placements))


def write(scene: Scene, name: str) -> None:
    """Write one scene as both SVG and PNG."""
    svg_path = IMAGES / f"{name}.svg"
    scene.save_svg(svg_path)
    cairosvg.svg2png(
        url=str(svg_path),
        write_to=str(IMAGES / f"{name}.png"),
        output_width=PNG_WIDTH,
        background_color="white",
    )
    print(f"wrote {name}.svg and {name}.png")


def example() -> Scene:
    """The picture at the top of the README: the same structure twice.

    Two rather than three, because a third would shrink each of them by
    almost half and the bases would stop being legible.
    """
    return row(
        [
            draw_svg_as_component(CLOVERLEAF),
            draw_svg_as_component(
                CLOVERLEAF, sequences=SEQUENCES, basepair_probabilities=PROBS
            ),
        ],
        gap=20.0,
    )


def structure() -> Scene:
    """The quick start: a structure on its own."""
    return draw_svg(CLOVERLEAF)


def sequences() -> Scene:
    """Sequence annotation: the same structure with its bases."""
    return draw_svg(CLOVERLEAF, sequences=SEQUENCES)


def probabilities() -> Scene:
    """Base-pair probabilities: the same structure, colored."""
    return draw_svg(CLOVERLEAF, sequences=SEQUENCES, basepair_probabilities=PROBS)


def styling() -> Scene:
    """The settings the README's style example uses."""
    return draw_svg(
        CLOVERLEAF,
        sequences=SEQUENCES,
        basepair_probabilities=PROBS,
        colormap=mpl.colormaps["viridis"],
        layout_engine=RadialLayoutEngine(
            stem_spacing=18.0,
            loop_spacing=24.0,
        ),
        style=StructureStyle(
            backbone_color="#333333",
            basepair_color="crimson",
            nucleotide_radius=5.0,
        ),
    )


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    write(example(), "example")
    write(structure(), "structure")
    write(sequences(), "sequences")
    write(probabilities(), "probabilities")
    write(styling(), "styling")


if __name__ == "__main__":
    main()
