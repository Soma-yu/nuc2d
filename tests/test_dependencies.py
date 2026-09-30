"""Which module may depend on which.

These pin the two boundaries the package is built around: SVG is written
in one module, and what is placed does not know how it is written.
"""

import ast
import importlib
import inspect
import pkgutil

import nuc2d


def imports_of(module):
    """Return the top-level names of every package a module imports."""
    tree = ast.parse(inspect.getsource(module))
    return {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0
    }


def package_imports_of(module):
    """Return the modules of this package that a module imports."""
    tree = ast.parse(inspect.getsource(module))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 1:
            if node.module:
                names.add(node.module.split(".")[0])
            else:
                names.update(alias.name for alias in node.names)
    return names


def test_only_the_svg_module_knows_how_svg_is_written():
    """Writing SVG another way changes nuc2d._svg and no other module."""
    importers = {
        info.name
        for info in pkgutil.iter_modules(nuc2d.__path__)
        if "svgwrite" in imports_of(importlib.import_module(f"nuc2d.{info.name}"))
    }

    assert importers == {"_svg"}


def test_what_is_placed_does_not_know_how_it_is_written():
    """Components and placements depend on geometry and nothing else.

    The module that writes SVG makes components, so it depends on this
    one. Were this one to depend back on it, for putting components
    together, the two would import each other. The validation of
    arguments is not a dependency in that sense, since it imports nothing.
    """
    module = importlib.import_module("nuc2d._component")

    assert package_imports_of(module) == {"_geometry", "_validation"}


def test_the_validation_of_arguments_can_be_used_by_any_module():
    """It imports nothing from the package.

    So no import cycle can run through it, whichever module uses it.
    """
    module = importlib.import_module("nuc2d._validation")

    assert package_imports_of(module) == set()
