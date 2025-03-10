"""
Cell 1.3 - Dependency Installation
Instalasi package yang diperlukan untuk project SmartCash.
"""

import sys
from IPython.display import display
import os

# Pastikan smartcash ada di path
if not any('smartcash' in p for p in sys.path):
    sys.path.append('.')

# Buat direktori jika belum ada
os.makedirs('smartcash/ui_components', exist_ok=True)
os.makedirs('smartcash/ui_handlers', exist_ok=True)

# Import komponen UI dan handler
from smartcash.ui_components.dependency_installer import create_dependency_ui
from smartcash.ui_handlers.dependency_installer import setup_dependency_handlers

# Buat dan setup UI
ui = create_dependency_ui()
ui = setup_dependency_handlers(ui)

# Tampilkan UI
display(ui['ui'])