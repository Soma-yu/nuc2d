"""Drawing secondary structures, from a dot-bracket string to a scene.

:func:`draw_structure`, :func:`draw_colorbar` and :func:`draw_text` each
draw one component, for a caller to place with others. :func:`draw_svg` draws a structure and
frames it as a scene in one call: ``draw_svg(...)`` is
``Scene(draw_structure(...))``, with the same arguments.
"""

import math

import numpy as np

from ._annotation import attach_equilibrium_probabilities, attach_sequences
from ._component import Component, Placement, fit
from ._geometry import BBox
from ._layout import RadialLayoutEngine
from ._parse import parse
from ._scene import Scene
from ._style import DrawingStyle
from ._svg import render_colorbar, render_structure, render_text


# The structure beside a colorbar is fitted into a square as tall as the
# colorbar, so that the colorbar keeps its own size whatever the shape of
# the structure, and a very wide structure does not shrink it to a sliver.
_STRUCTURE_SLOT_ASPECT_RATIO = 1.0


def draw_colorbar(
    *,
    label: str | None = "Equilibrium probability",
    style: DrawingStyle | None = None,
) -> Component:
    """Draw a colorbar for the probabilities a structure is colored by.

    Parameters
    ----------
    label : str or None, default="Equilibrium probability"
        Text written alongside the colorbar, or None to leave it without
        one. The colorbar occupies the same box either way, so colorbars
        with and without a label line up.
    style : DrawingStyle, optional
        Drawing style. Its colormap and colorbar settings apply.

    Returns
    -------
    Component
        The colorbar, to be placed with :class:`Placement`.
    """
    return render_colorbar(label=label, style=style)


def draw_text(
    text: str,
    *,
    font_family: str = "Arial",
    font_size: float = 12.0,
) -> Component:
    """Draw one line of text, such as a title, as a component.

    Parameters
    ----------
    text : str
        The text to draw, on one line.
    font_family : str, default="Arial"
        One font family name, not a CSS list. The font is looked up on the
        machine doing the drawing, and its metrics decide how large the
        component's box is.
    font_size : float, default=12.0
        Font size, in the same units as everything the text is placed
        with.

    Returns
    -------
    Component
        The text, in a box that runs from the font's ascender to its
        descender and from the start of the first character to the end of
        the last, so that it can be placed by any point of that box.

    Raises
    ------
    ValueError
        If ``text`` is empty or holds a line break, or ``font_size`` is
        not a positive finite number.
    TypeError
        If ``text`` or ``font_family`` is not a string.

    Notes
    -----
    The box is measured from the font's own widths, without kerning, and
    from the font found on this machine. Shown with a different font, the
    text can run a little longer or shorter than its box.
    """
    for name, value in (("text", text), ("font_family", font_family)):
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string; got {type(value).__name__}.")
    if not text:
        raise ValueError("text is empty, so there is nothing to draw.")
    if "\n" in text or "\r" in text:
        raise ValueError(
            "draw_text draws one line; place one text per line instead. "
            f"Got {text!r}."
        )
    if not (math.isfinite(font_size) and font_size > 0):
        raise ValueError(
            f"font_size must be a positive finite number; got {font_size!r}."
        )
    return render_text(text, font_family=font_family, font_size=font_size)


def draw_structure(
    dot_bracket: str,
    *,
    sequences: list[str] | None = None,
    probs: np.ndarray | None = None,
    style: DrawingStyle | None = None,
    layout_engine: RadialLayoutEngine | None = None,
    colorbar_label: str | None = "Equilibrium probability",
    add_colorbar: bool = True,
) -> Component:
    """Draw a secondary structure as a component.

    Parameters
    ----------
    dot_bracket : str
        A secondary structure written with ``(`` and ``)`` for the two
        halves of a base pair, ``.`` for an unpaired nucleotide, and ``+``
        for a break between strands.
    sequences : list[str], optional
        Nucleotide sequences, one per strand, in the order the strands
        appear in the structure.
    probs : ndarray, optional
        Base-pair probability matrix: ``probs[i][j]`` is how likely
        nucleotides ``i`` and ``j`` are to be paired with each other, and
        ``probs[i][i]`` how likely nucleotide ``i`` is to be left unpaired.
        When given, each nucleotide is colored by it, and a colorbar is
        set beside the structure.
    style : DrawingStyle, optional
        Drawing style.
    layout_engine : RadialLayoutEngine, optional
        Engine computing nucleotide positions. Defaults to a
        :class:`~nuc2d.RadialLayoutEngine` with its own defaults.
    colorbar_label : str or None, default="Equilibrium probability"
        Text written alongside the colorbar, or None to leave it without
        one. Has no effect unless ``probs`` is given, since the colorbar is
        drawn only then.
    add_colorbar : bool, default=True
        Whether to set a colorbar beside the structure when ``probs`` is
        given. Passing False colors the nucleotides but leaves the
        colorbar out, for a caller placing one of its own from
        :func:`~nuc2d.draw_colorbar`.

    Returns
    -------
    Component
        The structure, with its colorbar if one is drawn, to be placed
        with :class:`Placement` or shown with :class:`Scene`.

    Raises
    ------
    ParseError
        If ``dot_bracket`` is not a well-formed secondary structure.
    ValueError
        If ``sequences`` or ``probs`` does not match the structure.

    Notes
    -----
    With a colorbar, the structure is fitted into a square as tall as the
    colorbar and centred in it. The colorbar keeps its own size, so that
    it stays legible beside a structure of any shape, and every structure
    drawn this way takes the same room.
    """
    root_loop = parse(dot_bracket)
    if sequences is not None:
        attach_sequences(root_loop, sequences)
    if probs is not None:
        attach_equilibrium_probabilities(root_loop, probs)

    engine = layout_engine if layout_engine is not None else RadialLayoutEngine()
    structure = render_structure(engine.layout(root_loop), style=style)

    if probs is None or not add_colorbar:
        return structure

    colorbar = draw_colorbar(label=colorbar_label, style=style)
    colorbar_bbox = colorbar.bbox
    slot = BBox(
        colorbar_bbox.xmin - colorbar_bbox.height * _STRUCTURE_SLOT_ASPECT_RATIO,
        colorbar_bbox.ymin,
        colorbar_bbox.xmin,
        colorbar_bbox.ymax,
    )
    return Component.from_placements(
        [fit(structure, slot, anchor="center"), Placement(component=colorbar)]
    )


def draw_svg(
    dot_bracket: str,
    *,
    sequences: list[str] | None = None,
    probs: np.ndarray | None = None,
    style: DrawingStyle | None = None,
    layout_engine: RadialLayoutEngine | None = None,
    colorbar_label: str | None = "Equilibrium probability",
    add_colorbar: bool = True,
    width_px: float | None = None,
    height_px: float | None = None,
) -> Scene:
    """Draw a secondary structure as a scene, ready to save or show.

    ``draw_svg(...)`` is ``Scene(draw_structure(...))``: the arguments up
    to ``add_colorbar`` are those of :func:`draw_structure`, and the last
    two those of :class:`Scene`.

    Parameters
    ----------
    dot_bracket : str
        A secondary structure written with ``(`` and ``)`` for the two
        halves of a base pair, ``.`` for an unpaired nucleotide, and ``+``
        for a break between strands.
    sequences : list[str], optional
        Nucleotide sequences, one per strand, in the order the strands
        appear in the structure.
    probs : ndarray, optional
        Base-pair probability matrix: ``probs[i][j]`` is how likely
        nucleotides ``i`` and ``j`` are to be paired with each other, and
        ``probs[i][i]`` how likely nucleotide ``i`` is to be left unpaired.
    style : DrawingStyle, optional
        Drawing style.
    layout_engine : RadialLayoutEngine, optional
        Engine computing nucleotide positions. Defaults to a
        :class:`~nuc2d.RadialLayoutEngine` with its own defaults.
    colorbar_label : str or None, default="Equilibrium probability"
        Text written alongside the colorbar, or None to leave it without
        one. Has no effect unless ``probs`` is given.
    add_colorbar : bool, default=True
        Whether to set a colorbar beside the structure when ``probs`` is
        given.
    width_px : float, optional
        Width of the scene in pixels. Given alone, the height follows
        from the proportions of the structure.
    height_px : float, optional
        Height of the scene in pixels. Given alone, the width follows.
        Giving neither sets the height to 500.

    Returns
    -------
    Scene
        The structure, framed and sized. Save it with
        :meth:`Scene.save_svg`; in Jupyter, a scene that ends a cell is
        displayed as it is.

    Raises
    ------
    ParseError
        If ``dot_bracket`` is not a well-formed secondary structure.
    ValueError
        If ``sequences`` or ``probs`` does not match the structure, or a
        size is not a positive number.
    """
    component = draw_structure(
        dot_bracket,
        sequences=sequences,
        probs=probs,
        style=style,
        layout_engine=layout_engine,
        colorbar_label=colorbar_label,
        add_colorbar=add_colorbar,
    )
    return Scene(component, width_px=width_px, height_px=height_px)
