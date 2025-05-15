"""
File: smartcash/ui/dataset/split/split_config_cell.py
Deskripsi: Cell untuk konfigurasi split dataset tanpa visualisasi
"""

import ipywidgets as widgets
from IPython.display import display
from typing import Dict, Any

from smartcash.ui.dataset.split.components.split_component import create_split_ui
from smartcash.ui.dataset.split.handlers.button_handlers import setup_button_handlers
from smartcash.ui.utils.ui_logger import create_direct_ui_logger
from smartcash.ui.utils.constants import ICONS

# Buat output widget untuk logger
output_widget = widgets.Output()

# Buat komponen UI
ui_components = create_split_ui()

# Setup UI logger
ui_components['output_log'] = output_widget
logger = create_direct_ui_logger(ui_components, 'split_config')
logger.debug(f"{ICONS['start']} Memulai setup Split Config UI")

# Setup button handlers
ui_components['logger'] = logger
ui_components = setup_button_handlers(ui_components)
logger.debug(f"{ICONS['handler']} Button handlers berhasil disetup")

# Tampilkan UI
display(ui_components['ui'])
logger.debug(f"{ICONS['success']} Split Config UI berhasil ditampilkan")
