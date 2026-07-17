"""Compatibility shim for older pip/setuptools.

All project configuration lives in ``pyproject.toml``; setuptools reads it
from there. This file only exists so that ``pip install -e .`` works with
pip versions that predate PEP 660 editable installs.
"""

from setuptools import setup

setup()
