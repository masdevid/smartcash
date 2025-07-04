"""
file_path: smartcash/ui/setup/colab/components/tips_panel.py
Deskripsi: Panel tips & requirements sederhana untuk modul Colab.
"""

from __future__ import annotations

import ipywidgets as widgets


def create_tips_requirements() -> widgets.HTML:
    """Return a minimal HTML widget berisi tips penggunaan Colab."""
    value = (
        "<ul>"
        "<li>Pastikan Google Drive ter-mount sebelum memulai setup.</li>"
        "<li>Jangan lupa cek quota penyimpanan Anda.</li>"
        "</ul>"
    )
    return widgets.HTML(value=value)
