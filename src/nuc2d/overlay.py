"""Overlay elements for secondary structure visualizations.

This module defines auxiliary graphical elements that can be added to
rendered secondary structure figures. Overlays are independent from the
secondary structure itself and provide supplementary information such as
titles, colorbars, legends, or other annotations.

Overlay objects are consumed by rendering backends and drawn in addition
to the structure layout.
"""

from abc import ABC
from dataclasses import dataclass


class Overlay(ABC):
    """Base class for rendering overlays."""
    pass


@dataclass
class Colorbar(Overlay):
    """Colorbar overlay for visualizing scalar values.

    Parameters
    ----------
    label : str
        Label displayed alongside the colorbar.
    """
    label: str


@dataclass
class Title(Overlay):
    """Title overlay displayed above the rendered figure.

    Parameters
    ----------
    text : str
        Title text.
    """
    text: str
