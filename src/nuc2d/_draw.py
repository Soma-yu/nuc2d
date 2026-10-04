"""Drawing secondary structures, from a dot-bracket string to a scene.

:func:`draw_svg` draws a structure and frames it as a scene, ready to save
or show. :func:`draw_svg_as_component` draws the same as a component, for
a caller to place with others: ``draw_svg(...)`` is
``Scene(draw_svg_as_component(...))``, with the same arguments.
"""

from __future__ import annotations

import matplotlib.colors
import numpy.typing

from ._annotation import attach_basepair_probabilities, attach_sequences
from ._validation import check_single_line_text, exact_str
from ._component import Component, Placement, fit
from ._geometry import bbox_around, make_bbox
from ._layout import RadialLayoutEngine, layout
from ._parse import parse
from ._scene import Scene
from ._style import StructureStyle
from ._svg import (
    COLORBAR_HEIGHT,
    render_colorbar,
    render_structure,
    render_text,
)


# The structure is fitted into a square as tall as a colorbar, whatever
# is drawn with it, so that every structure drawn takes the same room.
# The colorbar and the title each keep one size: a very wide structure
# does not shrink them to a sliver, and structures placed at one size
# show titles of one size.
_STRUCTURE_SLOT_ASPECT_RATIO = 1.0

_TITLE_FONT_SIZE = 20.0
_TITLE_GAP = 10.0  # between the title and what it is set over


def _check_label(name: str, value: object, *, without: str) -> str | None:
    """Raise unless ``value`` is None or one line of text, not all blank.

    ``without`` says what None leaves without the text, for the message
    a blank string is answered with. The text is returned, as a str.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(
            f"{name} must be a string or None; got {type(value).__name__}."
        )
    if not value.strip():
        raise ValueError(f"{name} is blank; pass None to leave {without}.")
    return check_single_line_text(name, value)


def _check_colormap(name: str, colormap: object) -> None:
    if colormap is not None and not isinstance(
        colormap, matplotlib.colors.Colormap
    ):
        raise TypeError(
            f"{name} must be a matplotlib Colormap, such as "
            f"mpl.colormaps['turbo']; got {type(colormap).__name__}."
        )


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
            f"{name} must be a RadialLayoutEngine, such as "
            f"RadialLayoutEngine(); got {type(layout_engine).__name__}."
        )


def draw_svg_as_component(
    structure: str,
    *,
    sequences: list[str] | None = None,
    basepair_probabilities: numpy.typing.ArrayLike | None = None,
    colormap: matplotlib.colors.Colormap | None = None,
    layout_engine: RadialLayoutEngine | None = None,
    style: StructureStyle | None = None,
    title: str | None = None,
    colorbar_label: str | None = "Equilibrium probability",
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
    basepair_probabilities : numpy.typing.ArrayLike, optional
        Base-pair probability matrix. Element ``(i, j)`` is how likely
        nucleotides ``i`` and ``j`` are to be paired with each other, and
        element ``(i, i)`` how likely nucleotide ``i`` is to be left
        unpaired. Only the elements on and above the diagonal are read,
        so the matrix may be symmetric or have only its upper triangle
        filled in. When given, each nucleotide is colored by it, and a
        colorbar is set beside the structure. A value below 0 or above 1
        is shown in the color of 0 or of 1.
    colormap : matplotlib.colors.Colormap, optional
        Colormap a probability from 0 to 1 is shown in, on the nucleotides
        and on the colorbar. It is a matplotlib colormap itself, such as
        ``mpl.colormaps["turbo"]``, the default, rather than the name of
        one. Has no effect unless ``basepair_probabilities`` is given.
    layout_engine : RadialLayoutEngine, optional
        Engine computing nucleotide positions. Defaults to a
        :class:`~nuc2d.RadialLayoutEngine` with its own defaults.
    style : StructureStyle, optional
        How the structure looks.
    title : str, optional
        Text written above the structure and its colorbar, on one line.
    colorbar_label : str or None, default="Equilibrium probability"
        Text written alongside the colorbar, on one line, or None to leave
        it without one. Has no effect unless ``basepair_probabilities`` is
        given, since the colorbar is drawn only then.

    Returns
    -------
    Component
        The structure, with its title and its colorbar if they are drawn,
        to be placed with :class:`Placement` or shown with :class:`Scene`.

    Raises
    ------
    ParseError
        If ``structure`` is not a well-formed secondary structure.
    TypeError
        If an argument is not of the type described above: in particular,
        if ``sequences`` is not a list of strings,
        ``basepair_probabilities`` does not hold numbers, or ``colormap``
        is the name of a colormap rather than the colormap itself.
    ValueError
        If ``sequences`` or ``basepair_probabilities`` does not match the
        structure, a sequence holds a line break, a tab or another control
        character, a probability a nucleotide is colored by is NaN, or
        ``title`` or ``colorbar_label`` is empty or only spaces, or holds
        a line break, a tab or another control character.
    """
    if not isinstance(structure, str):
        raise TypeError(
            "structure must be a string, such as '((...))'; "
            f"got {type(structure).__name__}."
        )
    structure = exact_str(structure)
    _check_colormap("colormap", colormap)
    _check_layout_engine("layout_engine", layout_engine)
    _check_style("style", style)
    title = _check_label(
        "title", title, without="the structure without a title"
    )
    colorbar_label = _check_label(
        "colorbar_label",
        colorbar_label,
        without="the colorbar without a label",
    )

    root_loop = parse(structure)
    if sequences is not None:
        attach_sequences(root_loop, sequences)
    if basepair_probabilities is not None:
        attach_basepair_probabilities(root_loop, basepair_probabilities)

    drawn = render_structure(
        layout(root_loop, layout_engine), colormap=colormap, style=style
    )

    slot = make_bbox(
        0.0,
        0.0,
        COLORBAR_HEIGHT * _STRUCTURE_SLOT_ASPECT_RATIO,
        COLORBAR_HEIGHT,
    )
    placements = [fit(drawn, slot, anchor="center")]
    if basepair_probabilities is not None:
        colorbar = render_colorbar(colormap=colormap, label=colorbar_label)
        placements.append(Placement(colorbar, x=slot.xmax, y=slot.ymin))
    if title is not None:
        under = bbox_around(p.bbox for p in placements)
        placements.append(
            Placement(
                render_text(title, font_size=_TITLE_FONT_SIZE),
                anchor="lower center",
                x=(under.xmin + under.xmax) / 2,
                y=under.ymin - _TITLE_GAP,
            )
        )
    return Component.from_placements(placements)


def draw_svg(
    structure: str,
    *,
    sequences: list[str] | None = None,
    basepair_probabilities: numpy.typing.ArrayLike | None = None,
    colormap: matplotlib.colors.Colormap | None = None,
    layout_engine: RadialLayoutEngine | None = None,
    style: StructureStyle | None = None,
    title: str | None = None,
    colorbar_label: str | None = "Equilibrium probability",
    width_px: float | None = None,
    height_px: float | None = None,
) -> Scene:
    """Draw a secondary structure as a scene, ready to save or show.

    ``draw_svg(...)`` is ``Scene(draw_svg_as_component(...))``: the
    arguments up to ``colorbar_label`` are those of
    :func:`draw_svg_as_component`, and the last two those of
    :class:`Scene`.

    Parameters
    ----------
    structure : str
        A secondary structure in dot-bracket notation: ``(`` and ``)`` for
        the two halves of a base pair, ``.`` for an unpaired nucleotide,
        and ``+`` for a break between strands.
    sequences : list[str], optional
        Nucleotide sequences, one per strand, in the order the strands
        appear in the structure.
    basepair_probabilities : numpy.typing.ArrayLike, optional
        Base-pair probability matrix. Element ``(i, j)`` is how likely
        nucleotides ``i`` and ``j`` are to be paired with each other, and
        element ``(i, i)`` how likely nucleotide ``i`` is to be left
        unpaired. Only the elements on and above the diagonal are read,
        so the matrix may be symmetric or have only its upper triangle
        filled in. A value below 0 or above 1 is shown in the color of 0
        or of 1.
    colormap : matplotlib.colors.Colormap, optional
        Colormap a probability from 0 to 1 is shown in, on the nucleotides
        and on the colorbar. It is a matplotlib colormap itself, such as
        ``mpl.colormaps["turbo"]``, the default, rather than the name of
        one. Has no effect unless ``basepair_probabilities`` is given.
    layout_engine : RadialLayoutEngine, optional
        Engine computing nucleotide positions. Defaults to a
        :class:`~nuc2d.RadialLayoutEngine` with its own defaults.
    style : StructureStyle, optional
        How the structure looks.
    title : str, optional
        Text written above the structure and its colorbar, on one line.
    colorbar_label : str or None, default="Equilibrium probability"
        Text written alongside the colorbar, on one line, or None to leave
        it without one. Has no effect unless ``basepair_probabilities`` is
        given.
    width_px : float, optional
        Width of the scene in pixels. Given alone, the height follows
        from the proportions of the structure.
    height_px : float, optional
        Height of the scene in pixels. Given alone, the width follows.
        Giving neither draws the scene at the size the structure is drawn
        at, one unit to a pixel.

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
        colormap=colormap,
        layout_engine=layout_engine,
        style=style,
        title=title,
        colorbar_label=colorbar_label,
    )
    return Scene(component, width_px=width_px, height_px=height_px)
