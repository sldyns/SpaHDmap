# Configuration file for the Sphinx documentation builder.

import sys
from datetime import datetime
from pathlib import Path

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
import SpaHDmap as hdmap

sys.path.insert(0, str(Path(__file__).parent / "_ext"))

# -- Project information -----------------------------------------------------

project = hdmap.__name__
author = hdmap.__author__
version = hdmap.__version__
repository_url = "https://github.com/sldyns/SpaHDmap"
copyright = f"{datetime.now():%Y}, Xilab"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "nbsphinx",
    "myst_parser",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "seaborn": ("https://seaborn.pydata.org/", None),
    "anndata": ("https://anndata.readthedocs.io/en/latest/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/latest/", None),
    "squidpy": ("https://squidpy.readthedocs.io/en/latest/", None),
}

master_doc = "index"
pygments_style = "tango"
pygments_dark_style = "monokai"

nitpicky = True


templates_path = ["_templates"]
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# myst
nb_execution_mode = "off"
myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "amsmath",
]
myst_heading_anchors = 2

# autodoc + napoleon
autosummary_generate = True
autodoc_member_order = "alphabetical"
autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# spelling
spelling_lang = "en_US"
spelling_warning = True
spelling_word_list_filename = "spelling_wordlist.txt"
spelling_add_pypi_package_names = True
spelling_exclude_patterns = ["references.rst"]
spelling_filters = [
    "enchant.tokenize.URLFilter",
    "enchant.tokenize.EmailFilter",
    "enchant.tokenize.MentionFilter",
]

exclude_patterns = [
    "_build",
    "notebooks/README.rst",
    "release/changelog/*",
    "**.ipynb_checkpoints",
]


qualname_overrides = {
    'anndata._core.anndata.AnnData': 'anndata.AnnData',
}
# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.1/css/all.min.css",
    'custom.css',
]
html_logo = "_static/logo.png"
html_title = "SpaHDmap"

html_show_sphinx = False
html_show_sourcelink = True
html_copy_source = True
html_theme_options = {
    "source_repository": "https://github.com/sldyns/SpaHDmap/",
    "source_branch": "main",
    "source_directory": "docs/",
    "top_of_page_buttons": ["view", "edit"],
    "sidebar_hide_name": True,
    "light_css_variables": {
        "color-brand-primary": "#003262",
        "color-brand-content": "#003262",
        "admonition-font-size": "var(--font-size-small)",
        "admonition-title-font-size": "var(--font-size-small)",
    },
    "footer_icons": [
        {
            "name": "Xilab",
            "url": "https://www.math.pku.edu.cn/teachers/xirb/Homepage.html",
            "html": "",
            "class": "fa-sharp-duotone fa-solid fa-house",
        },

        {
            "name": "GitHub",
            "url": "https://github.com/sldyns/SpaHDmap",
            "html": "",
            "class": "fab fa-github",
        },
    ],
}
