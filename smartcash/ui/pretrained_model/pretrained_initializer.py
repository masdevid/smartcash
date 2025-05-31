"""
File: smartcash/ui/pretrained_model/pretrained_initializer.py
Deskripsi: Inisialisasi UI dan logika bisnis untuk pretrained model dengan pendekatan DRY menggunakan CommonInitializer
"""

from typing import Dict, Any, List, Optional
from IPython.display import display

from smartcash.ui.utils.common_initializer import CommonInitializer
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.ui_logger_namespace import PRETRAINED_MODEL_LOGGER_NAMESPACE, KNOWN_NAMESPACES
from smartcash.common.environment import get_environment_manager
from smartcash.ui.pretrained_model.utils.logger_utils import get_module_logger

# Gunakan logger dari utils
MODULE_LOGGER_NAME = KNOWN_NAMESPACES[PRETRAINED_MODEL_LOGGER_NAMESPACE]
logger = get_module_logger()

class PretrainedModelInitializer(CommonInitializer):
    """Implementasi CommonInitializer untuk modul pretrained model"""
    
    def __init__(self):
        super().__init__(module_name=MODULE_LOGGER_NAME, logger_namespace=PRETRAINED_MODEL_LOGGER_NAMESPACE)
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Membuat komponen UI untuk model pretrained - one-liner style"""
        from smartcash.ui.pretrained_model.components.pretrained_components import create_pretrained_ui
        return {**create_pretrained_ui(), 'env': env, 'config': config}
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers khusus modul model - one-liner style"""
        from smartcash.ui.pretrained_model.handlers.setup_handlers import setup_model_handlers, setup_model_cleanup_handler
        return setup_model_cleanup_handler(setup_model_handlers(ui_components, env, config), self.module_name, config, env)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Mendapatkan konfigurasi default untuk model pretrained - one-liner style"""
        return {
            'models_dir': '/content/models',
            'drive_models_dir': '/content/drive/MyDrive/SmartCash/models',
            'models': {
                'yolov5s': {
                    'path': '/content/models/yolov5s.pt', 
                    'url': 'https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt', 
                    'size': 14*1024*1024
                },
                'efficientnet-b4': {
                    'path': '/content/models/efficientnet-b4_notop.h5', 
                    'url': 'https://storage.googleapis.com/keras-applications/efficientnet/efficientnet-b4_notop.h5', 
                    'size': 75*1024*1024
                }
            }
        }
    
    def _get_critical_components(self) -> List[str]:
        """Mendapatkan komponen kritis yang harus ada di UI - one-liner style"""
        return ['main_container', 'status', 'log', 'download_sync_button']
    
    def _get_return_value(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Custom return value untuk modul model - one-liner style"""
        display(ui_components.get('main_container')); return ui_components

# Helper functions untuk drive - disarankan dipindahkan ke utils di masa depan
is_drive_mounted = lambda: __import__('os').path.exists('/content/drive/MyDrive')

def mount_drive() -> tuple:
    """Memasang Google Drive - one-liner style"""
    try: __import__('google.colab').drive.mount('/content/drive'); return True, f"{ICONS.get('success', '✅')} Google Drive berhasil dipasang"
    except Exception as e: return False, f"{ICONS.get('error', '❌')} Gagal memasang Google Drive: {str(e)}"

# Fungsi entrypoint yang digunakan di cell notebook
def initialize_pretrained_model_ui() -> Dict[str, Any]:
    """Fungsi entrypoint untuk cell notebook - mempertahankan API publik agar backward compatible"""
    return PretrainedModelInitializer().initialize(get_environment_manager())

# Fungsi untuk menjalankan test case - one-liner style
run_tests = lambda: __import__('smartcash.ui.pretrained_model.tests.test_simple_download').ui.pretrained_model.tests.test_simple_download.run_all_tests()
