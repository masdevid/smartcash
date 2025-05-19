"""
File: smartcash/ui/dataset/download/components/__init__.py
Deskripsi: Package untuk komponen UI download dataset
"""

from smartcash.ui.dataset.download.components.download_component import create_download_ui
from smartcash.ui.dataset.download.components.ui_creator import create_download_ui as create_ui

__all__ = ['create_download_ui', 'create_ui']
