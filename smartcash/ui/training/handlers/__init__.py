"""
File: smartcash/ui/training/handlers/__init__.py
Deskripsi: Inisialisasi modul handlers untuk UI training
"""

from smartcash.ui.training.handlers.setup_handler import setup_training_handlers
from smartcash.ui.training.handlers.training_info_handler import update_training_info
from smartcash.ui.training.handlers.training_execution_handler import run_training
from smartcash.ui.training.handlers.button_event_handlers import on_start_click, on_stop_click, register_button_handlers
from smartcash.ui.training.handlers.training_handler_utils import (
    get_training_status,
    set_training_status,
    update_ui_status,
    display_status_panel,
    ensure_ui_persistence,
    update_button_states
)

# Untuk kompatibilitas dengan kode lama
from smartcash.ui.training.handlers.setup_handler import setup_training_handlers as setup_training_button_handlers

__all__ = [
    'setup_training_handlers',
    'setup_training_button_handlers',  # Alias untuk kompatibilitas
    'update_training_info',
    'run_training',
    'on_start_click',
    'on_stop_click',
    'register_button_handlers',
    'get_training_status',
    'set_training_status',
    'update_ui_status',
    'display_status_panel',
    'ensure_ui_persistence',
    'update_button_states'
]
