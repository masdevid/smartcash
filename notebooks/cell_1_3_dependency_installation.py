"""
Cell 1.3 - Dependency Installation
Instalasi package yang diperlukan untuk project SmartCash.
"""

import sys
import atexit
from IPython.display import display

# Pastikan smartcash ada di path
if not any('smartcash' in p for p in sys.path):
    sys.path.append('.')

# Buat direktori smartcash jika belum ada
import os
os.makedirs('smartcash/ui_components', exist_ok=True)
os.makedirs('smartcash/ui_handlers', exist_ok=True)

# Import komponen UI dan handler
from smartcash.ui_components.dependency_installer import create_dependency_ui
from smartcash.ui_handlers.dependency_installer import setup_dependency_handlers

# Buat dan setup UI
ui_components = create_dependency_ui()
ui_components = setup_dependency_handlers(ui_components)

# Register cleanup untuk dijalankan saat notebook dihentikan
if 'cleanup' in ui_components and callable(ui_components['cleanup']):
    atexit.register(ui_components['cleanup'])

# Tampilkan UI
display(ui_components['ui'])