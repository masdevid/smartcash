"""
File: smartcash/ui/dataset/split/split_config_cell.py
Deskripsi: Cell untuk konfigurasi split dataset tanpa visualisasi
"""

import ipywidgets as widgets
from IPython.display import display
from typing import Dict, Any

from smartcash.ui.dataset.split.components.split_component import create_split_ui
from smartcash.ui.dataset.split.handlers.button_handlers import setup_button_handlers
from smartcash.common.logger import get_logger
from smartcash.ui.utils.constants import ICONS

# Setup logger
logger = get_logger('split_config')
logger.info(f"{ICONS['start']} Memulai setup Split Config UI")

# Buat komponen UI
ui_components = create_split_ui()
logger.info(f"{ICONS['ui']} Komponen UI split config berhasil dibuat")

# Setup button handlers
ui_components['logger'] = logger
ui_components = setup_button_handlers(ui_components)
logger.info(f"{ICONS['handler']} Button handlers berhasil disetup")

# Tampilkan UI
display(ui_components['ui'])
logger.info(f"{ICONS['success']} Split Config UI berhasil ditampilkan")
