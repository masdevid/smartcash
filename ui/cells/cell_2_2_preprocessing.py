"""
File: smartcash/ui/cells/cell_2_2_preprocessing.py
Deskripsi: Cell minimal untuk preprocessing dataset SmartCash
"""

import sys
from IPython.display import display
if '.' not in sys.path: sys.path.append('.')

try:
    from smartcash.ui.dataset.preprocessing import setup_preprocessing
    # Setup dependency installer
    ui_components = setup_preprocessing()
    
    # Tampilkan UI
    display(ui_components['ui'])

except ImportError as e: create_alert(e, 'error')