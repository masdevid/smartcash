"""
File: cell_2_1_dataset_download.py
Deskripsi: Cell untuk download dataset SmartCash dengan integrasi logging UI
"""

# Import komponen UI dari smartcash
from IPython.display import display
import sys

# Pastikan smartcash dalam path
if '.' not in sys.path:
    sys.path.append('.')

# Setup environment dan load config
from smartcash.ui.utils.cell_utils import setup_notebook_environment, setup_ui_component, display_ui
from smartcash.ui.utils.logging_utils import setup_ipython_logging
from smartcash.ui.dataset.dataset_download_handler import setup_dataset_download_handlers
import logging

# Setup environment dan load config
env, config = setup_notebook_environment(
    cell_name="dataset_download",
    config_path="configs/colab_config.yaml"
)

# Setup komponen UI dan handler
ui_components = setup_ui_component(env, config, "dataset_download")

# Setup logger yang terintegrasi dengan UI
logger = setup_ipython_logging(ui_components, "dataset_download", log_level=logging.INFO)
if logger:
    ui_components['logger'] = logger
    logger.info("🚀 Cell dataset_download diinisialisasi")

# Tambahkan dataset manager jika tersedia
try:
    from smartcash.dataset.manager import DatasetManager
    dataset_manager = DatasetManager(config=config, logger=logger)
    ui_components['dataset_manager'] = dataset_manager
    logger.info("✅ Dataset Manager berhasil diinisialisasi")
except ImportError as e:
    if logger:
        logger.warning(f"⚠️ Tidak dapat menggunakan DatasetManager: {str(e)}")
        logger.info("ℹ️ Beberapa fitur mungkin tidak tersedia")

# Lakukan validasi struktur dataset yang sudah ada
try:
    if 'validate_dataset_structure' in ui_components and callable(ui_components['validate_dataset_structure']):
        data_dir = config.get('data', {}).get('dir', 'data')
        ui_components['validate_dataset_structure'](data_dir)
except Exception as e:
    if logger:
        logger.warning(f"⚠️ Tidak dapat memvalidasi dataset: {str(e)}")

ui_components = setup_dataset_download_handlers(ui_components, env, config)
# Tampilkan UI
display_ui(ui_components)

# Pembersihan sumber daya dilakukan dengan:
# cleanup_resources(ui_components)