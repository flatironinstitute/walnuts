# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Walnuts"
import datetime

year = datetime.date.today().year
copyright = f"{year}, Walnuts Developers"
author = "Walnuts Developers"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

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
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

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

intersphinx_mapping = {
    "python": (
        "https://docs.python.org/3/",
        None,
    ),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "bridgestan": ("https://roualdes.us/bridgestan/latest/", None),
}


breathe_projects = {"walnuts": "_build/breathe/doxygen/walnuts/xml/"}
breathe_projects_source = {
    "walnuts": (
        "../include/",
        [
            "walnuts.hpp",
            "walnuts/api.hpp",
            "walnuts/concepts.hpp",
            "walnuts/config.hpp",
            "walnuts/walnuts.hpp",
            "walnuts/adaptive_walnuts.hpp",
        ],
    )
}
breathe_default_project = "walnuts"
breathe_show_include = False

# doxygen doesn't like  __attribute and __declspec
# https://www.doxygen.nl/manual/preprocessing.html
# breathe_doxygen_config_options = {
#     "ENABLE_PREPROCESSING": "YES",
#     "MACRO_EXPANSION": "YES",
#     "EXPAND_ONLY_PREDEF": "YES",
#     "PREDEFINED": "WALNUTS_STRONG_INLINE=",
# }

autoclass_content = "both"
