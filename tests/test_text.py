import pathlib
import xml.etree.ElementTree as ET

import matplotlib as mpl
import pytest
from fontTools.ttLib import TTCollection, TTFont
from matplotlib import font_manager

from nuc2d import Component, Placement, Scene, draw_structure, draw_text
from nuc2d._font import find_font, text_width, vertical_extent


def texts(svg_string):
    return [
        element for element in ET.fromstring(svg_string).iter()
        if element.tag.endswith("text")
    ]


def test_the_text_is_written_as_given():
    [element] = texts(Scene(draw_text("tRNA cloverleaf")).to_svg())

    assert element.text == "tRNA cloverleaf"
    assert element.attrib["font-family"] == "Arial"
    assert float(element.attrib["font-size"]) == 12.0


def test_markup_in_the_text_is_written_as_text():
    [element] = texts(Scene(draw_text("a < b & c")).to_svg())

    assert element.text == "a < b & c"


def test_the_box_runs_from_the_ascender_to_the_descender():
    above, below = vertical_extent(find_font("Arial"), 20.0)

    bbox = draw_text("Wg", font_size=20.0).bbox

    assert (bbox.xmin, bbox.ymin) == (0.0, 0.0)
    assert bbox.height == pytest.approx(above + below)


def test_the_box_grows_with_the_text_and_the_size():
    short = draw_text("tRNA").bbox
    longer = draw_text("tRNA cloverleaf").bbox
    larger = draw_text("tRNA", font_size=24.0).bbox

    assert longer.width > short.width
    assert larger.width == pytest.approx(short.width * 2)
    assert larger.height == pytest.approx(short.height * 2)


def test_a_title_can_be_centered_over_a_structure():
    title = Placement(component=draw_text("tRNA", font_size=15.0), anchor="upper center")
    structure = Placement(
        component=draw_structure("(((...)))"),
        y=title.bbox.ymax + 10.0,
        anchor="upper center",
    )

    panel = Component.from_placements([title, structure])

    def center_x(bbox):
        return (bbox.xmin + bbox.xmax) / 2

    assert center_x(title.bbox) == pytest.approx(center_x(structure.bbox))
    assert panel.bbox.height == pytest.approx(
        title.bbox.height + 10.0 + structure.bbox.height
    )


@pytest.mark.parametrize(
    "text",
    ["", "two\nlines", "two\rlines", "two\r\nlines", "trailing\n",
     "a\tb", "form\x0cfeed", "nul\x00", "next\x85line", "line\u2028separator",
     "paragraph\u2029separator", "lone \ud800 surrogate",
     "noncharacter \ufffe", "noncharacter \uffff"],
)
def test_text_that_is_not_a_single_line_is_refused(text):
    """Tabs too: SVG shows one as a space, but it is measured as a tab."""
    with pytest.raises(ValueError, match="text"):
        draw_text(text)


@pytest.mark.parametrize(
    "text",
    ["平衡確率", "\u0394G (kcal/mol)", "5\u2032 end", "100 %", "a\\nb",
     "no\u00a0break", "全角\u3000スペース", "<b>&amp;</b>", " padded "],
)
def test_any_other_character_is_drawn_as_written(text):
    """A backslash is a character like any other: nothing interprets it."""
    svg = Scene(draw_text(text)).to_svg()

    assert [element.text for element in texts(svg)] == [text]


@pytest.mark.parametrize("font_size", [0.0, -1.0, float("inf"), float("nan")])
def test_the_font_size_must_be_positive_and_finite(font_size):
    with pytest.raises(ValueError, match="font_size"):
        draw_text("tRNA", font_size=font_size)


@pytest.mark.parametrize("kwargs", [{"text": 3}, {"text": "tRNA", "font_family": None}])
def test_text_and_family_must_be_strings(kwargs):
    text = kwargs.pop("text")

    with pytest.raises(TypeError):
        draw_text(text, **kwargs)


@pytest.mark.parametrize(
    "font_family", ["", "  ", "Arial, sans-serif", "Arial\n", "Ari\tal"]
)
def test_the_font_family_is_one_family_name(font_family):
    with pytest.raises(ValueError, match="font_family"):
        draw_text("tRNA", font_family=font_family)


@pytest.mark.parametrize("font_size", ["12", True])
def test_the_font_size_must_be_a_number(font_size):
    with pytest.raises(TypeError, match="font_size"):
        draw_text("tRNA", font_size=font_size)


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
