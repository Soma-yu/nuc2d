"""Regenerate the images the README shows.

Run after any change that affects rendering, so the pictures on the front
page keep matching the code they illustrate::

    uv run python docs/make_images.py

Each image is written to ``docs/images/`` as both SVG and PNG. The
structures and settings here are the ones the README's own examples use,
so the two stay in step.
"""

from pathlib import Path

import cairosvg
import numpy as np
import matplotlib as mpl
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
# screen. The SVG carries only a default display size for anyone who opens
# it directly; 500 px tall is what draw_svg itself defaults to.
PNG_WIDTH = 1600
SVG_HEIGHT = 500.0


def row(
    drawing: svgwrite.Drawing,
    components: list[SVGComponent],
    gap: float,
) -> svgwrite.Drawing:
    """Lay components out left to right and frame the drawing on them."""
    placements = []
    x = 0.0
    for component in components:
        placements.append(
            Placement(component=component, x=x - component.bbox.xmin, y=0.0, scale=1.0)
        )
        x += component.bbox.width + gap

    panel = compose(drawing.g(), placements)

    drawing.add(panel.group)
    drawing.viewbox(*panel.bbox.to_viewbox())
    drawing["height"] = f"{SVG_HEIGHT}px"
    drawing["width"] = f"{SVG_HEIGHT * panel.bbox.width / panel.bbox.height}px"
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
    """The drawing at the top of the README: sequences and probabilities.

    The probability matrix comes from a structure prediction tool rather
    than from this package, so it is stored beside this script instead of
    being recomputed. Regenerate it with np.save if the structure shown
    here ever changes.
    """
    dot_bracket = (
        ".....((((((((((..((("
        "+(((((.....))))))))..(((((.(((((.....))))))))))..)))))"
        "+.....)))))"
    )
    sequences = [
        "TTTTTATATGAGCGTTTCCG",
        "CGTGCTTTTTGCACGCGGTTACCACTGTGCCTTTTTGGCACGTGGTTTACGCT",
        "TGCACCATAT",
    ]
    probs = np.load(DOCS / "example_probs.npy")

    return draw_svg(dot_bracket, sequences=sequences, probs=probs)


def styling() -> svgwrite.Drawing:
    """The default drawing beside the styled one, as the README shows it."""
    dot_bracket = "(((..+...)))"
    drawing = svgwrite.Drawing()

    components = [
        draw_component(drawing, dot_bracket),
        draw_component(
            drawing,
            dot_bracket,
            style=DrawingStyle(
                node_color="steelblue",
                edge_color="dimgray",
                node_radius=5.0,
                cmap=mpl.colormaps["viridis"],
            ),
            layout_engine=RadialLayoutEngine(
                backbone_spacing=20.0,
                loop_spacing=25.0,
            ),
        ),
    ]

    return row(drawing, components, gap=30.0)


def composing() -> svgwrite.Drawing:
    """The three-structure panel from the README."""
    drawing = svgwrite.Drawing()

    components = [
        draw_component(drawing, "(((...)))"),
        draw_component(drawing, "((..((...))..))"),
        draw_component(drawing, "((((....))))"),
    ]

    return row(drawing, components, gap=10.0)


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    write(example(), "example")
    write(styling(), "styling")
    write(composing(), "composing")


if __name__ == "__main__":
    main()
