"""
File: smartcash/ui/cells/cell_2_1_dataset_download.py
Deskripsi: Cell untuk download dataset SmartCash dengan kode minimal
"""

import sys
from IPython.display import display
if '.' not in sys.path: sys.path.append('.')

try:
    from smartcash.ui.dataset.dataset_download import setup_dataset_download
    # Setup dependency installer
    ui_components = setup_dataset_download()
    
    # Tampilkan UI
    display(ui_components['ui'])

except ImportError as e: create_alert(e, 'error')