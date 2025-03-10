"""
Cell 1.1 - Environment Configuration
Setup lingkungan kerja untuk project SmartCash.
"""

import sys
import os
from IPython.display import display
import ipywidgets as widgets

# Pastikan smartcash ada di path
if not any('smartcash' in p for p in sys.path):
    sys.path.append('.')

# Buat direktori smartcash jika belum ada
os.makedirs('smartcash/ui_components', exist_ok=True)
os.makedirs('smartcash/ui_handlers', exist_ok=True)

# Import komponen UI dan handler
from smartcash.ui_components.env_config import create_env_config_ui
from smartcash.ui_handlers.env_config import setup_env_handlers

# Buat dan setup UI
ui = create_env_config_ui()
ui = setup_env_handlers(ui)

# Tampilkan UI
display(ui['ui'])