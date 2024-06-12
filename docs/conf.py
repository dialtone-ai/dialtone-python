"""Sphinx configuration."""
project = "Dialtone"
author = "Petr Gazarov"
copyright = "2024, Petr Gazarov"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
