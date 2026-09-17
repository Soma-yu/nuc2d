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
import matplotlib as mpl
import svgwrite

from nuc2d import (
    DrawingStyle,
    Placement,
    RadialLayoutEngine,
    SVGComponent,
    compose,
    draw_component,
)

IMAGES = Path(__file__).parent / "images"
PIXEL_WIDTH = 900


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
    drawing["width"] = f"{PIXEL_WIDTH}px"
    drawing["height"] = f"{PIXEL_WIDTH * panel.bbox.height / panel.bbox.width}px"
    return drawing


def write(drawing: svgwrite.Drawing, name: str) -> None:
    """Write one drawing as both SVG and PNG."""
    svg_path = IMAGES / f"{name}.svg"
    drawing.saveas(str(svg_path))
    cairosvg.svg2png(
        url=str(svg_path),
        write_to=str(IMAGES / f"{name}.png"),
        output_width=PIXEL_WIDTH,
        background_color="white",
    )
    print(f"wrote {name}.svg and {name}.png")


def styling() -> svgwrite.Drawing:
    """The default drawing beside the styled one, as the README shows it."""
    dpp_string = "(((..+...)))"
    drawing = svgwrite.Drawing()

    components = [
        draw_component(drawing, dpp_string),
        draw_component(
            drawing,
            dpp_string,
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
    write(styling(), "styling")
    write(composing(), "composing")


if __name__ == "__main__":
    main()
