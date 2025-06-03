"""
File: smartcash/ui/setup/dependency_installer/dependency_installer_initializer.py
Deskripsi: Fixed dependency installer menggunakan CommonInitializer pattern dengan penanganan error yang lebih robust
"""

import logging
import traceback
from typing import Dict, Any, List, Optional
from IPython.display import display

from smartcash.ui.utils.common_initializer import CommonInitializer
from smartcash.ui.utils.ui_logger_namespace import DEPENDENCY_INSTALLER_LOGGER_NAMESPACE, KNOWN_NAMESPACES

# Gunakan namespace dari konstanta
MODULE_LOGGER_NAME = KNOWN_NAMESPACES[DEPENDENCY_INSTALLER_LOGGER_NAMESPACE]
from smartcash.ui.setup.dependency_installer.utils.validation_utils import validate_ui_components, get_default_config

class DependencyInstallerInitializer(CommonInitializer):
    """Initializer untuk dependency installer dengan pendekatan CommonInitializer pattern dan penanganan error yang lebih robust."""
    
    def __init__(self):
        """Initialize DependencyInstallerInitializer dengan logger setup menggunakan namespace."""
        super().__init__(MODULE_LOGGER_NAME, DEPENDENCY_INSTALLER_LOGGER_NAMESPACE)
        # Logger sudah disetup oleh parent class
    
    def _get_critical_components(self) -> List[str]:
        """Mendapatkan daftar komponen kritis yang harus ada dalam UI components."""
        return ['ui', 'install_button', 'status', 'log_output', 'progress_container', 'status_panel']
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Mendapatkan konfigurasi default untuk dependency installer."""
        from smartcash.ui.setup.dependency_installer.utils.validation_utils import get_default_config
        return get_default_config()
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Membuat UI components untuk dependency installer dengan penanganan error yang lebih baik."""
        from smartcash.ui.setup.dependency_installer.components.dependency_installer_component import create_dependency_installer_ui
        
        try:
            self.logger.info("Creating dependency installer UI components")
            ui_components = create_dependency_installer_ui(env=env or config.get('env'), config=config)
            # Tambahkan module name dan namespace untuk logging yang konsisten
            ui_components.update({
                'module_name': MODULE_LOGGER_NAME, 
                'logger_namespace': DEPENDENCY_INSTALLER_LOGGER_NAMESPACE,
                'dependency_installer_initialized': False
            })
            self.logger.info(f"UI components created successfully with {len(ui_components)} components")
            return ui_components
        except Exception as e:
            error_details = traceback.format_exc()
            self.logger.error(f"Failed to create UI components: {str(e)}\n{error_details}")
            return {'error': 'Failed to create UI components', 'details': str(e), 'traceback': error_details}
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers untuk dependency installer dengan penanganan error yang lebih baik."""
        from smartcash.ui.setup.dependency_installer.handlers.setup_handlers import setup_dependency_installer_handlers
        
        # Jika UI components sudah error, skip setup handlers
        if 'error' in ui_components:
            self.logger.error(f"Skipping handler setup due to previous error: {ui_components['error']}")
            return ui_components
        
        try:
            # Tandai sebagai diinisialisasi untuk mencegah log duplikat
            ui_components['dependency_installer_initialized'] = True
            
            self.logger.info("Setting up dependency installer handlers")
            result_components = setup_dependency_installer_handlers(ui_components, config)
            
            # Pastikan flag dipertahankan jika setup_dependency_installer_handlers mengembalikan dict baru
            if isinstance(result_components, dict):
                result_components['dependency_installer_initialized'] = True
                result_components['handlers_setup'] = True
                ui_components = result_components
            else:
                ui_components['handlers_setup'] = True
                
            self.logger.info("Handlers setup completed successfully")
            return ui_components
        except Exception as e:
            error_details = traceback.format_exc()
            self.logger.error(f"Failed to setup handlers: {str(e)}\n{error_details}")
            ui_components['handlers_setup'] = False
            
            # Log error ke UI jika log_message tersedia
            if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'append_log'):
                ui_components['log_output'].append_log(f"Error setting up handlers: {str(e)}", 'error')
            elif 'log_message' in ui_components and callable(ui_components['log_message']):
                ui_components['log_message'](f"❌ Error setting up handlers: {str(e)}", 'error')
            
            return ui_components
    
    def _validate_ui_components(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Validasi UI components untuk memastikan semua komponen kritis tersedia dengan penanganan error yang lebih baik."""
        try:
            # Jika UI components sudah error, skip validasi
            if 'error' in ui_components:
                self.logger.error(f"UI components validation skipped due to previous error: {ui_components['error']}")
                return ui_components
            
            # Validasi komponen kritis
            self.logger.info("Validating critical UI components")
            critical_components = self._get_critical_components()
            missing_components = []
            
            for component in critical_components:
                if component not in ui_components:
                    missing_components.append(component)
            
            if missing_components:
                error_msg = f"Missing critical components: {', '.join(missing_components)}\nAvailable components: {', '.join(ui_components.keys())}"
                self.logger.error(error_msg)
                return {'error': 'Failed to create UI components', 'details': error_msg, 'available_components': list(ui_components.keys())}
            
            self.logger.info("All critical components validated successfully")
            return validate_ui_components(ui_components)
        except Exception as e:
            error_details = traceback.format_exc()
            self.logger.error(f"Failed to validate UI components: {str(e)}\n{error_details}")
            return {'error': 'Failed to validate UI components', 'details': str(e), 'traceback': error_details}
    
    def _post_initialization_hook(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Hook yang dijalankan setelah inisialisasi selesai dengan penanganan error yang lebih baik."""
        # Jika UI components sudah error, tampilkan error dan skip post initialization
        if 'error' in ui_components:
            self.logger.error(f"Post initialization skipped due to previous error: {ui_components['error']}")
            
            # Buat widget error sederhana untuk ditampilkan
            try:
                import ipywidgets as widgets
                error_widget = widgets.HTML(
                    value=f"""
                    <div style="padding:10px; background-color:#f8d7da; color:#721c24; border-radius:4px; margin:10px 0;
                            border-left:4px solid #dc3545;">
                        <h3 style="margin-top:0;">❌ Error: {ui_components['error']}</h3>
                        <p>{ui_components.get('details', '')}</p>
                        <p>Silakan restart kernel dan coba lagi, atau hubungi developer untuk bantuan.</p>
                    </div>
                    """,
                    layout=widgets.Layout(width='100%')
                )
                display(error_widget)
            except Exception as display_error:
                self.logger.error(f"Failed to display error widget: {str(display_error)}")
            
            return ui_components
        
        try:
            self.logger.info("Running post initialization hook")
            
            # Display UI
            display(ui_components['ui'])
            self.logger.info("UI displayed successfully")
            
            # Log initialization success
            if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'append_log'):
                ui_components['log_output'].append_log("Dependency installer initialized successfully", 'success')
            
            # Run delayed analysis jika dikonfigurasi
            if config.get('delay_analysis', True) and 'run_delayed_analysis' in ui_components and ui_components.get('handlers_setup', False):
                self.logger.info("Running delayed analysis")
                ui_components['run_delayed_analysis']()
                
                # Sembunyikan progress container setelah analisis selesai jika tidak ada instalasi otomatis
                if not config.get('auto_install', False) and 'progress_container' in ui_components and hasattr(ui_components['progress_container'], 'layout'):
                    ui_components['progress_container'].layout.visibility = 'hidden'
            
            return ui_components
        except Exception as e:
            error_details = traceback.format_exc()
            self.logger.error(f"Error in post initialization: {str(e)}\n{error_details}")
            
            # Log error ke UI jika log_output tersedia
            if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'append_log'):
                ui_components['log_output'].append_log(f"Error in post initialization: {str(e)}", 'warning')
            elif 'log_message' in ui_components and callable(ui_components['log_message']):
                ui_components['log_message'](f"⚠️ Warning: {str(e)}", 'warning')
            
            return ui_components
        
    def _get_return_value(self, ui_components: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Mendapatkan nilai return yang akan ditampilkan di notebook dengan pendekatan one-liner"""
        # Langsung kembalikan ui_components tanpa silent wrapper
        return ui_components


def initialize_dependency_installer(config: Optional[Dict[str, Any]] = None, env=None) -> Dict[str, Any]:
    """Initialize dependency installer UI dan setup handlers dengan penanganan error yang lebih baik."""
    initializer = DependencyInstallerInitializer()
    config = config or get_default_config()
    
    try:
        # Clear output terlebih dahulu untuk menghindari masalah rendering
        from IPython.display import clear_output
        clear_output(wait=True)
        
        # Initialize UI components dan setup handlers
        ui_components = initializer.initialize(config, env)
        
        # Langsung kembalikan ui_components tanpa silent wrapper
        return ui_components
    except Exception as e:
        error_details = traceback.format_exc()
        initializer.logger.error(f"Unexpected error in initialize_dependency_installer: {str(e)}\n{error_details}")
        
        # Buat error components minimal yang dapat ditampilkan
        return {'error': 'Failed to initialize dependency installer', 'details': str(e)}

# Tambahkan alias untuk backward compatibility
from smartcash.common.config.manager import get_config_manager as _get_config_manager
from smartcash.common.environment import get_environment_manager as _get_environment_manager
get_config_manager, get_environment_manager = _get_config_manager, _get_environment_manager