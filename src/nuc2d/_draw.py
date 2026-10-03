"""Drawing secondary structures, from a dot-bracket string to a scene.

:func:`draw_svg` draws a structure and frames it as a scene, ready to save
or show. :func:`draw_svg_as_component` draws the same as a component, for
a caller to place with others: ``draw_svg(...)`` is
``Scene(draw_svg_as_component(...))``, with the same arguments.
"""

from __future__ import annotations

import numpy.typing as npt

from ._annotation import attach_basepair_probabilities, attach_sequences
from ._validation import check_single_line_text, exact_str
from ._component import Component, Placement, fit
from ._geometry import make_bbox
from ._layout import RadialLayoutEngine, layout
from ._parse import parse
from ._scene import Scene
from ._style import StructureStyle
from ._svg import render_colorbar, render_structure


# The structure beside a colorbar is fitted into a square as tall as the
# colorbar, so that the colorbar keeps its own size whatever the shape of
# the structure, and a very wide structure does not shrink it to a sliver.
_STRUCTURE_SLOT_ASPECT_RATIO = 1.0


def _check_label(name: str, label: object) -> str | None:
    """Raise unless ``label`` is None or single-line text for a colorbar.

    The label is returned, a string as a str.
    """
    if label is None:
        return None
    if not isinstance(label, str):
        raise TypeError(
            f"{name} must be a string or None; got {type(label).__name__}."
        )
    if not label:
        raise ValueError(
            f"{name} is empty; pass None to leave the colorbar without "
            "a label."
        )
    return check_single_line_text(name, label)


def _check_style(name: str, style: object) -> None:
    if style is not None and not isinstance(style, StructureStyle):
        raise TypeError(
            f"{name} must be a StructureStyle, such as StructureStyle(); "
            f"got {type(style).__name__}."
        )


def _check_layout_engine(name: str, layout_engine: object) -> None:
    if layout_engine is not None and not isinstance(
        layout_engine, RadialLayoutEngine
    ):
        raise TypeError(
            f"{name} must be a RadialLayoutEngine; "
            f"got {type(layout_engine).__name__}."
        )


def draw_svg_as_component(
    structure: str,
    *,
    sequences: list[str] | None = None,
    basepair_probabilities: npt.ArrayLike | None = None,
    colorbar_label: str | None = "Equilibrium probability",
    layout_engine: RadialLayoutEngine | None = None,
    style: StructureStyle | None = None,
) -> Component:
    """Draw a secondary structure as a component, to place with others.

    It draws what :func:`draw_svg` draws, from the same arguments less the
    size: ``draw_svg(...)`` is ``Scene(draw_svg_as_component(...))``.

    Parameters
    ----------
    structure : str
        A secondary structure in dot-bracket notation: ``(`` and ``)`` for
        the two halves of a base pair, ``.`` for an unpaired nucleotide,
        and ``+`` for a break between strands.
    sequences : list[str], optional
        Nucleotide sequences, one per strand, in the order the strands
        appear in the structure.
    basepair_probabilities : array_like, optional
        Base-pair probability matrix. Element ``(i, j)`` is how likely
        nucleotides ``i`` and ``j`` are to be paired with each other, and
        element ``(i, i)`` how likely nucleotide ``i`` is to be left
        unpaired. Only the elements on and above the diagonal are read,
        so the matrix may be symmetric or have only its upper triangle
        filled in. When given, each nucleotide is colored by it, and a
        colorbar is set beside the structure. A value below 0 or above 1
        is shown in the color of 0 or of 1.
    colorbar_label : str or None, default="Equilibrium probability"
        Text written alongside the colorbar, on one line, or None to leave
        it without one. Has no effect unless ``basepair_probabilities`` is
        given, since the colorbar is drawn only then.
    layout_engine : RadialLayoutEngine, optional
        Engine computing nucleotide positions. Defaults to a
        :class:`~nuc2d.RadialLayoutEngine` with its own defaults.
    style : StructureStyle, optional
        How the structure looks.

    Returns
    -------
    Component
        The structure, with its colorbar if one is drawn, to be placed
        with :class:`Placement` or shown with :class:`Scene`.

    Raises
    ------
    ParseError
        If ``structure`` is not a well-formed secondary structure.
    TypeError
        If an argument is not of the type described above: in particular,
        if ``sequences`` is not a list of strings, or
        ``basepair_probabilities`` does not hold numbers.
    ValueError
        If ``sequences`` or ``basepair_probabilities`` does not match the
        structure, a sequence holds a line break, a tab or another control
        character, a probability a nucleotide is colored by is NaN, or
        ``colorbar_label`` is empty or holds a line break, a tab or
        another control character.

    Notes
    -----
    The structure's box leaves a margin around the centers of its
    outermost nucleotides. It is not measured from what is drawn, so
    nodes, letters or 3' arrows drawn large enough reach past it, and a
    scene cuts them off at its edge. How the box is drawn around a
    structure belongs to the drawing rather than to the API, and a minor
    release may change it.

    With a colorbar, the structure is fitted into a square as tall as the
    colorbar and centered in it. The colorbar keeps its own size, so that
    it stays legible beside a structure of any shape, and every structure
    drawn this way takes the same room.
    """
    if not isinstance(structure, str):
        raise TypeError(
            "structure must be a string, such as '((...))'; "
            f"got {type(structure).__name__}."
        )
    structure = exact_str(structure)
    _check_layout_engine("layout_engine", layout_engine)
    _check_style("style", style)
    colorbar_label = _check_label("colorbar_label", colorbar_label)

    root_loop = parse(structure)
    if sequences is not None:
        attach_sequences(root_loop, sequences)
    if basepair_probabilities is not None:
        attach_basepair_probabilities(root_loop, basepair_probabilities)

    engine = layout_engine if layout_engine is not None else RadialLayoutEngine()
    drawn = render_structure(layout(root_loop, engine), style=style)

    if basepair_probabilities is None:
        return drawn

    colorbar = render_colorbar(label=colorbar_label, style=style)
    colorbar_bbox = colorbar.bbox
    slot = make_bbox(
        colorbar_bbox.xmin - colorbar_bbox.height * _STRUCTURE_SLOT_ASPECT_RATIO,
        colorbar_bbox.ymin,
        colorbar_bbox.xmin,
        colorbar_bbox.ymax,
    )
    return Component.from_placements(
        [fit(drawn, slot, anchor="center"), Placement(colorbar)]
    )


def draw_svg(
    structure: str,
    *,
    sequences: list[str] | None = None,
    basepair_probabilities: npt.ArrayLike | None = None,
    colorbar_label: str | None = "Equilibrium probability",
    layout_engine: RadialLayoutEngine | None = None,
    style: StructureStyle | None = None,
    width_px: float | None = None,
    height_px: float | None = None,
) -> Scene:
    """Draw a secondary structure as a scene, ready to save or show.

    ``draw_svg(...)`` is ``Scene(draw_svg_as_component(...))``: the
    arguments up to ``style`` are those of :func:`draw_svg_as_component`,
    and the last two those of :class:`Scene`.

    Parameters
    ----------
    structure : str
        A secondary structure in dot-bracket notation: ``(`` and ``)`` for
        the two halves of a base pair, ``.`` for an unpaired nucleotide,
        and ``+`` for a break between strands.
    sequences : list[str], optional
        Nucleotide sequences, one per strand, in the order the strands
        appear in the structure.
    basepair_probabilities : array_like, optional
        Base-pair probability matrix. Element ``(i, j)`` is how likely
        nucleotides ``i`` and ``j`` are to be paired with each other, and
        element ``(i, i)`` how likely nucleotide ``i`` is to be left
        unpaired. Only the elements on and above the diagonal are read,
        so the matrix may be symmetric or have only its upper triangle
        filled in. A value below 0 or above 1 is shown in the color of 0
        or of 1.
    colorbar_label : str or None, default="Equilibrium probability"
        Text written alongside the colorbar, on one line, or None to leave
        it without one. Has no effect unless ``basepair_probabilities`` is
        given.
    layout_engine : RadialLayoutEngine, optional
        Engine computing nucleotide positions. Defaults to a
        :class:`~nuc2d.RadialLayoutEngine` with its own defaults.
    style : StructureStyle, optional
        How the structure looks.
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
        If ``structure`` is not a well-formed secondary structure.
    TypeError
        If an argument is not of the type described above, as for
        :func:`draw_svg_as_component` and :class:`Scene`.
    ValueError
        If an argument is refused by :func:`draw_svg_as_component`, or a
        size is not a positive finite number.
    """
    component = draw_svg_as_component(
        structure,
        sequences=sequences,
        basepair_probabilities=basepair_probabilities,
        colorbar_label=colorbar_label,
        layout_engine=layout_engine,
        style=style,
    )
    return Scene(component, width_px=width_px, height_px=height_px)
