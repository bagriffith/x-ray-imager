import sys
from pathlib import Path

sys.path.insert(0, str(Path('..', 'src').resolve()))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'X-ray Imager Processing'
copyright = '2026, Brady Griffith'
author = 'Brady Griffith'
release = '0.3.0.dev0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.napoleon',
              'sphinx.ext.intersphinx',
              'myst_parser',
              'sphinx_click',
              'matplotlib.sphinxext.roles'
              ]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
source_suffix = {
    '.md': 'markdown',
}

intersphinx_mapping = {'python': ('https://docs.python.org/3', None),
                       'numpy': ('https://numpy.org/doc/stable/', None),
                       'numpy': ('https://numpy.org/doc/stable/', None),
                       'scipy': ('https://docs.scipy.org/doc/scipy/', None),
                       'matplotlib': ('https://matplotlib.org/stable', None)}

# -- Napoleon settings -------------------------------------------------------
napoleon_google_docstring = True


autodoc_typehints = "description"
autodoc_class_signature = "separated"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ['_static']
