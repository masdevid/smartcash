"""
File: smartcash/ui/setup/dependency_installer/dependency_installer_initializer.py
Deskripsi: Fixed dependency installer menggunakan CommonInitializer pattern
"""

from typing import Dict, Any, List
from smartcash.ui.utils.common_initializer import CommonInitializer
from smartcash.ui.utils.ui_logger_namespace import DEPENDENCY_INSTALLER_LOGGER_NAMESPACE, KNOWN_NAMESPACES
MODULE_LOGGER_NAME = KNOWN_NAMESPACES[DEPENDENCY_INSTALLER_LOGGER_NAMESPACE]

# Import handlers dan components
from smartcash.ui.setup.dependency_installer.handlers.setup_handlers import setup_dependency_installer_handlers
from smartcash.ui.setup.dependency_installer.components.dependency_installer_component import create_dependency_installer_ui
from smartcash.common.config.manager import SimpleConfigManager as ConfigManager, get_config_manager as _get_config_manager
from smartcash.common.environment import get_environment_manager as _get_environment_manager


class DependencyInstallerInitializer(CommonInitializer):
    """Fixed dependency installer menggunakan CommonInitializer pattern"""
    
    def __init__(self):
        super().__init__(MODULE_LOGGER_NAME, DEPENDENCY_INSTALLER_LOGGER_NAMESPACE)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration untuk dependency installer"""
        return {
            'auto_install': False,
            'selected_packages': ['yolov5_req', 'smartcash_req', 'torch_req'],
            'custom_packages': '',
            'validate_after_install': True
        }
    
    def _get_critical_components(self) -> List[str]:
        """Critical component keys yang harus ada"""
        return ['ui', 'install_button', 'status']
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create UI components untuk dependency installer"""
        return create_dependency_installer_ui(env, config)
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers untuk dependency installer dengan penanganan error yang lebih baik"""
        try:
            # Tambahkan flag untuk mencegah log duplikat
            ui_components['dependency_installer_initialized'] = True
            
            # Setup handlers dengan penanganan error yang lebih baik
            ui_components = setup_dependency_installer_handlers(ui_components, env, config)
            ui_components['handlers_setup'] = True
            
            # Gunakan log_message jika tersedia untuk memastikan log hanya muncul di UI
            if 'log_message' in ui_components and callable(ui_components['log_message']):
                ui_components['log_message']("✅ Handlers berhasil disetup", "success")
        except Exception as e:
            # Tangkap error dan log dengan cara yang lebih baik
            error_message = f"❌ Handlers setup failed: {str(e)}"
            
            # Gunakan log_message jika tersedia untuk memastikan log hanya muncul di UI
            if 'log_message' in ui_components and callable(ui_components['log_message']):
                ui_components['log_message'](error_message, "error")
            else:
                # Fallback ke logger jika log_message tidak tersedia
                logger = ui_components.get('logger', self.logger)
                if logger:
                    logger.error(error_message)
            
            # Tandai setup gagal tapi jangan gagalkan inisialisasi
            ui_components['handlers_setup'] = False
        
        return ui_components
    
    def _additional_validation(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Validation untuk dependency installer"""
        button_keys = ['install_button']
        functional_buttons = [key for key in button_keys if ui_components.get(key) and hasattr(ui_components[key], 'on_click')]
        
        if not functional_buttons:
            return {'valid': False, 'message': 'Install button tidak functional'}
        
        return {'valid': True, 'functional_buttons': functional_buttons}
        
    def _get_return_value(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Override _get_return_value untuk mengembalikan ui_components dictionary, bukan hanya UI widget"""
        # Pastikan ui_components adalah dictionary dan bukan widget
        return ui_components


# Global instance
_dependency_installer_initializer = DependencyInstallerInitializer()
_config_manager = None

# Public API
def initialize_dependency_installer(env=None, config=None):
    """Inisialisasi dan tampilkan UI dependency installer"""
    from IPython.display import display, clear_output
    import builtins
    import sys
    from io import StringIO
    
    # Simpan stdout asli
    original_stdout = sys.stdout
    
    # Redirect stdout untuk mencegah pencetakan dictionary
    sys.stdout = StringIO()
    
    try:
        # Clear output terlebih dahulu untuk menghindari masalah rendering
        clear_output(wait=True)
        
        # Siapkan config dasar tanpa inisialisasi penuh
        config = config or {}
        config['suppress_logs'] = True  # Tambahkan flag untuk menekan log selama inisialisasi
        
        # Inisialisasi UI components dasar terlebih dahulu
        ui_components_basic = _dependency_installer_initializer._create_ui_components(config)
        
        # Tampilkan UI dasar terlebih dahulu
        if isinstance(ui_components_basic, dict) and 'ui' in ui_components_basic:
            display(ui_components_basic['ui'])
        
        # Sekarang lakukan inisialisasi penuh setelah UI ditampilkan
        ui_components = _dependency_installer_initializer.initialize(env=env, config=config)
        
        # Update UI jika berbeda dari yang dasar
        if isinstance(ui_components, dict) and 'ui' in ui_components and ui_components['ui'] != ui_components_basic.get('ui'):
            clear_output(wait=True)
            display(ui_components['ui'])
        
        # Aktifkan log setelah UI terender
        if isinstance(ui_components, dict):
            ui_components['suppress_logs'] = False
            
            # Setup reset log handler untuk tombol instalasi
            if 'install_button' in ui_components and hasattr(ui_components['install_button'], 'on_click'):
                original_handler = None
                for handler in ui_components['install_button']._click_handlers.callbacks:
                    original_handler = handler
                    break
                
                if original_handler:
                    # Hapus handler asli
                    ui_components['install_button']._click_handlers.callbacks.clear()
                    
                    # Tambahkan handler baru yang mereset log terlebih dahulu
                    def enhanced_click_handler(b):
                        # Reset log output
                        if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'clear_output'):
                            ui_components['log_output'].clear_output()
                        # Panggil handler asli
                        original_handler(b)
                    
                    ui_components['install_button'].on_click(enhanced_click_handler)
            
            # Log status inisialisasi jika tersedia
            if 'log_message' in ui_components and callable(ui_components['log_message']):
                ui_components['log_message']("✅ Dependency installer UI berhasil diinisialisasi", "success")
        else:
            # Fallback jika UI tidak dapat diinisialisasi
            print("❌ Error: Komponen UI dependency installer tidak dapat diinisialisasi dengan benar.")
    finally:
        # Kembalikan stdout asli
        sys.stdout = original_stdout
    
    # Kembalikan ui_components untuk digunakan jika diperlukan tanpa mencetak ke konsol
    class SuppressedReturn:
        def __init__(self, obj):
            self.obj = obj
        def __repr__(self):
            return ""
    
    return SuppressedReturn(ui_components).obj

def get_config_manager():
    """Mendapatkan instance ConfigManager untuk dependency installer dengan pendekatan singleton"""
    # Menggunakan fungsi get_config_manager yang sudah diimpor dari smartcash.common.config.manager
    return _get_config_manager()

def get_environment_manager():
    """Mendapatkan instance EnvironmentManager untuk dependency installer dengan pendekatan singleton"""
    # Menggunakan fungsi get_environment_manager yang sudah diimpor dari smartcash.common.environment
    return _get_environment_manager()