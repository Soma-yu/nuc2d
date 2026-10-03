import pathlib
import xml.etree.ElementTree as ET

import matplotlib as mpl
import numpy as np
import pytest
from fontTools.ttLib import TTCollection, TTFont
from matplotlib import font_manager

from nuc2d import Scene, draw_svg
from nuc2d._font import find_font, text_width, vertical_extent
from nuc2d._svg import render_text


def texts(svg_string):
    return [
        element for element in ET.fromstring(svg_string).iter()
        if element.tag.endswith("text")
    ]


def line(text, font_size=12.0):
    """One line of text, as a component, in the font everything is set in."""
    return render_text(text, font_size=font_size)


def test_the_text_is_written_as_given():
    [element] = texts(Scene(line("tRNA cloverleaf")).to_svg())

    assert element.text == "tRNA cloverleaf"
    assert element.attrib["font-family"] == "Arial"
    assert float(element.attrib["font-size"]) == 12.0


def test_the_text_is_anchored_at_the_middle_of_its_box():
    """Shown in a font of other widths, it stays centered on the box."""
    component = line("tRNA cloverleaf")

    [element] = texts(Scene(component).to_svg())

    assert element.attrib["text-anchor"] == "middle"
    assert float(element.attrib["x"]) == pytest.approx(component.bbox.width / 2)


def test_markup_in_the_text_is_written_as_text():
    [element] = texts(Scene(line("a < b & c")).to_svg())

    assert element.text == "a < b & c"


def test_the_box_runs_from_the_ascender_to_the_descender():
    above, below = vertical_extent(find_font("Arial"), 20.0)

    bbox = line("Wg", font_size=20.0).bbox

    assert (bbox.xmin, bbox.ymin) == (0.0, 0.0)
    assert bbox.height == pytest.approx(above + below)


def test_the_box_grows_with_the_text_and_the_size():
    short = line("tRNA").bbox
    longer = line("tRNA cloverleaf").bbox
    larger = line("tRNA", font_size=24.0).bbox

    assert longer.width > short.width
    assert larger.width == pytest.approx(short.width * 2)
    assert larger.height == pytest.approx(short.height * 2)


def written(name, text):
    """A scene with text as its title or as its colorbar label."""
    return draw_svg(
        "(((...)))", basepair_probabilities=np.eye(9) * 0.5, **{name: text}
    )


@pytest.mark.parametrize("name", ["title", "colorbar_label"])
@pytest.mark.parametrize(
    "text",
    ["", "two\nlines", "two\rlines", "two\r\nlines", "trailing\n",
     "a\tb", "form\x0cfeed", "nul\x00", "next\x85line", "line\u2028separator",
     "paragraph\u2029separator", "lone \ud800 surrogate",
     "noncharacter \ufffe", "noncharacter \uffff"],
)
def test_text_that_is_not_a_single_line_is_refused(name, text):
    """Tabs too: SVG shows one as a space, but it is measured as a tab."""
    with pytest.raises(ValueError, match=name):
        written(name, text)


@pytest.mark.parametrize("name", ["title", "colorbar_label"])
@pytest.mark.parametrize(
    "text",
    ["平衡確率", "\u0394G (kcal/mol)", "5\u2032 end", "100 %", "a\\nb",
     "no\u00a0break", "全角\u3000スペース", "<b>&amp;</b>", " padded "],
)
def test_any_other_character_is_drawn_as_written(name, text):
    """A backslash is a character like any other: nothing interprets it."""
    svg = written(name, text).to_svg()

    assert text in [element.text for element in texts(svg)]


# ---------------------------------------------------------------- finding fonts


def test_a_family_is_looked_up_by_its_name_and_not_as_a_pattern():
    """In a fontconfig pattern, "DejaVu Sans:bold" is DejaVu Sans in bold.

    That is what Matplotlib reads a string alone as, so the text was
    measured in bold while the SVG asked for a family of that name.
    """
    path, _ = find_font("DejaVu Sans:bold")

    assert pathlib.Path(path).name != "DejaVuSans-Bold.ttf"


@pytest.fixture(scope="module")
def collection(tmp_path_factory):
    """Install a font collection whose second font is a family of its own.

    Each is one of the DejaVu fonts Matplotlib ships, under a family name
    no other font has, so that the second can only be found in here.
    """
    ttf_dir = pathlib.Path(mpl.get_data_path()) / "fonts" / "ttf"
    fonts = []
    for file, family in [
        ("DejaVuSans.ttf", "Nuc2D Collected Sans"),
        ("DejaVuSerif.ttf", "Nuc2D Collected Serif"),
    ]:
        font = TTFont(ttf_dir / file)
        for record in font["name"].names:
            if record.nameID in (1, 4, 16):
                record.string = family
            elif record.nameID == 6:
                record.string = family.replace(" ", "")
        fonts.append(font)
    path = tmp_path_factory.mktemp("fonts") / "collected.ttc"
    ttc = TTCollection()
    ttc.fonts = fonts
    ttc.save(path)
    font_manager.fontManager.addfont(str(path))
    return str(path)


@pytest.mark.skipif(
    mpl.__version_info__ < (3, 11),
    reason="Matplotlib reads only the first font of a collection before 3.11",
)
def test_the_font_of_a_collection_that_matched_is_the_one_measured(collection):
    """It used to be the first font of the file, whichever had matched."""
    assert find_font("Nuc2D Collected Serif") == (collection, 1)

    def width(index):
        font = TTFont(collection, fontNumber=index)
        cmap, units = font.getBestCmap(), font["head"].unitsPerEm
        return sum(font["hmtx"][cmap[ord(c)]][0] for c in "Hello") / units * 10

    measured = text_width(find_font("Nuc2D Collected Serif"), "Hello", 10)

    assert measured == pytest.approx(width(1)) and width(1) != width(0)
