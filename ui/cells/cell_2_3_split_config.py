"""
File: smartcash/ui/cells/cell_2_3_split_config.py
Deskripsi: Cell untuk konfigurasi pembagian dataset SmartCash dengan visualisasi yang disederhanakan
"""

import sys
from IPython.display import display
if '.' not in sys.path: sys.path.append('.')

try:
    from smartcash.ui.dataset.split_config import setup_split_config
    # Setup dependency installer
    ui_components = setup_split_config()
    
    # Tampilkan UI
    display(ui_components['ui'])

except ImportError as e: create_alert(e, 'error')