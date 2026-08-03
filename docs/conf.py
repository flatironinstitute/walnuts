# Configuration file for the Sphinx documentation builder.

project = "Walnutpie"
import datetime

year = datetime.date.today().year
copyright = f"{year}, Walnutpie Developers"
author = "Walnutpie Developers"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

latex_engine = "xelatex"

extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "breathe",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = []

# html_favicon = "_static/image/favicon.ico"

html_show_sphinx = False

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/flatironinstitute/walnuts",
            "icon": "fab fa-github",
        },
    ],
    "use_edit_page_button": True,
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
}

html_context = {
    "github_user": "flatironinstitute",
    "github_repo": "walnuts",
    "github_version": "main",
    "doc_path": "docs",
}

# latex_logo = "_static/image/logo.pdf"

# -- Sphinx plugin configuration -------------------------------------------------

intersphinx_mapping = {
    "python": (
        "https://docs.python.org/3/",
        None,
    ),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "bridgestan": ("https://roualdes.us/bridgestan/latest/", None),
    "numba": ("https://numba.readthedocs.io/en/stable/", None),
}

breathe_projects = {"walnutpie": "_build/breathe/doxygen/walnutpie/xml/"}
breathe_projects_source = {
    "walnutpie": (
        "../include/",
        [
            "walnutpie.hpp",
            "walnutpie/api.hpp",
            "walnutpie/concepts.hpp",
            "walnutpie/config.hpp",
            "walnutpie/walnuts.hpp",
            "walnutpie/adaptive_walnuts.hpp",
        ],
    )
}
breathe_default_project = "walnutpie"
breathe_show_include = False

autoclass_content = "both"

nbsphinx_allow_errors = False
