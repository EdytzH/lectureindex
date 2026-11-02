"""Bidirectional Streamlit component running Puter.js inside the user's browser.

Puter credentials and billing never enter the Python process. The frontend only
sends lecture text and a bounded candidate packet after an explicit user click.
"""
from pathlib import Path
import streamlit.components.v1 as components

_puter_component = components.declare_component(
    'bootcamp_puter_browser', path=str(Path(__file__).parent / 'puter_component'))


def puter_chat(request, key='puter_chat'):
    return _puter_component(request=request, key=key, default=None)
