"""
File: smartcash/ui/pretrained_model/handlers/download_handlers.py
Deskripsi: Handler untuk tombol download dan sinkronisasi model pretrained
"""

from typing import Dict, Any, Callable, Optional
from pathlib import Path
import time
from IPython.display import display, HTML

from smartcash.ui.utils.constants import ICONS, COLORS
from smartcash.common.environment import EnvironmentManager
from smartcash.common.logger import get_logger

logger = get_logger(__name__)

def handle_download_sync_button(b, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk tombol download dan sinkronisasi model pretrained.
    
    Args:
        b: Button widget yang dipicu
        ui_components: Dictionary berisi komponen UI
    """
    # Gunakan handler dari ui_handlers
    from smartcash.ui.model.handlers.ui_handlers import handle_download_sync_button as _handle_button
    return _handle_button(b, ui_components)

# Fungsi enable_button tidak diperlukan lagi karena kita tidak menggunakan threading

def process_download_sync(ui_components: Dict[str, Any]) -> None:
    """
    Memproses download dan sinkronisasi model pretrained.
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    # Gunakan implementasi dari process_orchestrator
    from smartcash.ui.model.services.process_orchestrator import process_download_sync as _process_download_sync
    return _process_download_sync(ui_components)
