"""
File: smartcash/ui/cells/cell_2_4_augmentation.py
Deskripsi: Cell untuk augmentasi dataset SmartCash dengan kode minimal
"""

import sys
from IPython.display import display
if '.' not in sys.path: sys.path.append('.')

try:
    from smartcash.ui.dataset.augmentation import setup_augmentation
    # Setup dependency installer
    ui_components = setup_augmentation()
    
    # Tampilkan UI
    display(ui_components['ui'])

except ImportError as e: create_alert(e, 'error')