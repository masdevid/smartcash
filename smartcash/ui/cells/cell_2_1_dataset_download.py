"""
File: smartcash/ui/cells/cell_2_1_dataset_download.py
Deskripsi: Cell untuk download dataset SmartCash dengan kode minimal
"""

# Import dasar
import sys
if '.' not in sys.path: sys.path.append('.')

try:
    from smartcash.ui.setup.dependency_installer import setup_dependency_installer
    # Setup dependency installer
    ui_components = setup_dependency_installer()
    
    # Tampilkan UI
    display(ui_components['ui'])
except ImportError as e: create_alert(e, 'error')