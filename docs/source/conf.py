import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../../"))  # to access konfai/

project = "KonfAI"
author = "Valentin Boussot"
copyright = f"{datetime.now().year}, {author}"  # noqa: A001 - required by Sphinx

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_togglebutton",
    "sphinx_tabs.tabs",
]

templates_path = ["_templates"]
html_static_path = ["_static"]
exclude_patterns = ["_build", "build", "Thumbs.db", ".DS_Store"]

html_theme = "shibuya"
html_title = "KonfAI documentation"
html_logo = os.path.abspath("../../logo.png")
html_favicon = os.path.abspath("../../logo.png")
html_css_files = ["custom.css"]

myst_enable_extensions = [
    "deflist",
    "fieldlist",
    "colon_fence",
    "html_admonition",
    "html_image",
]
myst_heading_anchors = 3
suppress_warnings = [
    "sphinx_autodoc_typehints.local_function",
    "intersphinx.external",
]

autodoc_default_options = {
    "members": True,
    "private-members": False,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autosummary_generate = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}
