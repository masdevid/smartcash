"""
File: smartcash/ui/setup/cell_1_3_dependency_installer.py
Deskripsi: Cell instalasi dependencies untuk SmartCash dengan pendekatan modular
"""

import sys
if '.' not in sys.path: sys.path.append('.')

try:
    from smartcash.ui.setup.dependency_installer import setup_dependency_installer
    # Setup dependency installer
    ui_components = setup_dependency_installer()
    
    # Tampilkan UI - Pastikan hanya menampilkan widget UI, bukan dictionary UI components
    display(ui_components['ui'])

except ImportError as e: create_alert(e, 'error')