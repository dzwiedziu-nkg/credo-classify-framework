# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath('../src'))


# -- Project information -----------------------------------------------------

project = 'credo_cf'
copyright = '2020, Michał Niedźwiecki, CREDO Team'
author = 'nkg'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    #"sphinx_autodoc_typehints",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "releases",
    "m2r"
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
# sphinx_rtd_theme
# html_theme = 'neo_rtd_theme'
# import sphinx_theme
# html_theme_path = [sphinx_theme.get_html_theme_path()]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

intersphinx_mapping = {'python': ('https://docs.python.org/3', None)}

source_suffix = ['.rst', '.md']

add_module_names = False
autodoc_typehints = 'description'
autodoc_member_order = 'bysource'


# https://github.com/miyakogi/m2r/issues/51 - workaround, TODO: please remove when m2r publish new version with fixed it
import sphinx

def monkeypatch(cls):
    """ decorator to monkey-patch methods """
    def decorator(f):
        method = f.__name__
        old_method = getattr(cls, method)
        setattr(cls, method, lambda self, *args, **kwargs: f(old_method, self, *args, **kwargs))
    return decorator

# workaround until https://github.com/miyakogi/m2r/pull/55 is merged
@monkeypatch(sphinx.registry.SphinxComponentRegistry)
def add_source_parser(_old_add_source_parser, self, *args, **kwargs):
    # signature is (parser: Type[Parser], **kwargs), but m2r expects
    # the removed (str, parser: Type[Parser], **kwargs).
    if isinstance(args[0], str):
        args = args[1:]
    return _old_add_source_parser(self, *args, **kwargs)
# end of workaround
