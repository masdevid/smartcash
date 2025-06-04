"""
File: smartcash/ui/dataset/downloader/downloader_init.py
Deskripsi: Streamlined downloader initializer dengan reduced fallbacks
"""

from typing import Dict, Any, List, Union
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.dataset.downloader.handlers.downloader_config_handler import DownloaderConfigHandler
from smartcash.ui.dataset.downloader.components.main_ui import create_downloader_ui
from smartcash.ui.dataset.downloader.handlers.download_handler import setup_download_handlers
from smartcash.ui.dataset.downloader.handlers.validation_handler import validate_download_parameters
from smartcash.common.logger import get_logger

class DownloaderInitializer(CommonInitializer):
    """Downloader initializer dengan streamlined error handling."""
    
    def __init__(self):
        super().__init__('downloader', DownloaderConfigHandler, parent_module='dataset')
    
    def _create_ui_components(self, config: Dict[str, Any] = None, env=None, **kwargs) -> Dict[str, Any]:
        """Create UI components dengan essential validation dan auto-fix untuk komponen yang hilang."""
        try:
            # Pastikan config tidak None
            if config is None:
                from smartcash.ui.dataset.downloader.handlers.downloader_config_handler import DownloaderConfigHandler
                config_handler = DownloaderConfigHandler()
                config = config_handler.get_default_config()
                self.logger.info("🔧 Menggunakan default config karena config=None")
            
            # Coba buat UI components
            ui_components = create_downloader_ui(config, env)
            
            # Basic validation
            if not isinstance(ui_components, dict):
                self.logger.error("❌ Invalid UI components structure - bukan dictionary")
                return None
                
            if 'ui' not in ui_components:
                self.logger.warning("⚠️ Invalid UI components structure - tidak ada kunci 'ui'")
                # Coba tambahkan main_container sebagai ui jika ada
                if 'main_container' in ui_components:
                    self.logger.info("🔧 Menggunakan main_container sebagai ui")
                    ui_components['ui'] = ui_components['main_container']
                else:
                    return None
            
            # Pastikan semua komponen yang diperlukan tersedia
            self._ensure_required_components(ui_components, config)
            
            # Tambahkan logger
            ui_components['logger'] = self.logger
            self.logger.success("✅ UI components created successfully")
            return ui_components
            
        except Exception as e:
            self.logger.error(f"❌ UI creation failed: {str(e)}")
            import traceback
            self.logger.debug(f"Stack trace: {traceback.format_exc()}")
            return None  # Return None instead of re-raising to allow fallback UI
    
    def _ensure_required_components(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Pastikan semua komponen yang diperlukan tersedia."""
        required_components = ['workspace_field', 'project_field', 'version_field', 'api_key_field', 'download_button']
        missing_components = [comp for comp in required_components if comp not in ui_components]
        
        if missing_components:
            self.logger.warning(f"⚠️ Komponen yang diperlukan hilang: {', '.join(missing_components)}")
            
            # Coba tambahkan komponen yang hilang
            from smartcash.ui.dataset.downloader.components.form_fields import create_form_fields
            from smartcash.ui.dataset.downloader.components.action_buttons import create_action_buttons
            
            if 'workspace_field' in missing_components:
                ui_components['workspace_field'] = create_form_fields(config)['workspace_field']
            
            if 'project_field' in missing_components:
                ui_components['project_field'] = create_form_fields(config)['project_field']
            
            if 'version_field' in missing_components:
                ui_components['version_field'] = create_form_fields(config)['version_field']
            
            if 'api_key_field' in missing_components:
                ui_components['api_key_field'] = create_form_fields(config)['api_key_field']
            
            if 'download_button' in missing_components:
                ui_components['download_button'] = create_action_buttons()['download_button']
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers dengan validasi yang lebih baik."""
        # Essential validation only
        required_components = ['download_button', 'workspace_field', 'project_field']
        missing = [comp for comp in required_components if comp not in ui_components]
        
        if missing:
            self.logger.error(f"❌ Komponen kritis tidak ditemukan: {', '.join(missing)}")
            return {'success': False, 'message': f"Komponen kritis tidak ditemukan: {', '.join(missing)}"}
        
        try:
            self.logger.info("📝 Setting up downloader handlers...")
            
            # Setup handlers
            from smartcash.ui.dataset.downloader.handlers.download_handler import setup_download_handlers
            handlers_result = setup_download_handlers(ui_components, env, config)
            
            # Periksa hasil setup handlers
            if not handlers_result.get('download_handlers', False):
                error_msg = handlers_result.get('error', 'Handler setup failed')
                self.logger.error(f"❌ {error_msg}")
                return {'success': False, 'message': error_msg}
            
            self.logger.success("✅ Handlers setup completed")
            
            # Optional auto-validation
            results = {'success': True, 'message': "Handlers berhasil dikonfigurasi", 'valid': True}
            
            if config and config.get('auto_validate', False):
                try:
                    from smartcash.ui.dataset.downloader.handlers.validation_handler import validate_download_parameters
                    validation_result = validate_download_parameters(ui_components, include_api_test=False)
                    results['validation'] = validation_result
                    if validation_result['valid']:
                        self.logger.info("✅ Initial validation passed")
                    else:
                        self.logger.info(f"ℹ️ Validation notes: {validation_result.get('message', '')}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Initial validation skipped: {str(e)}")
            
            return results
                
        except Exception as e:
            self.logger.error(f"❌ Handler setup failed: {str(e)}")
            import traceback
            self.logger.debug(f"Stack trace: {traceback.format_exc()}")
            return {'success': False, 'message': f"Handler setup failed: {str(e)}"}
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default config with single fallback."""
        try:
            from smartcash.ui.dataset.downloader.handlers.defaults import DEFAULT_CONFIG
            return DEFAULT_CONFIG.copy()
        except ImportError:
            # Single fallback - minimal but functional
            return {
                '_base_': ['base_config'],
                'workspace': 'smartcash-wo2us',
                'project': 'rupiah-emisi-2022',
                'version': '3',
                'api_key': '',
                'output_format': 'yolov5pytorch',
                'validate_download': True,
                'progress_enabled': True,
                'retry_attempts': 3,
                'timeout_seconds': 30,
                'module_name': 'downloader',
                'version': '1.0.0'
            }
    
    def _get_critical_components(self) -> List[str]:
        """Essential components only."""
        return ['ui']
    
    def _validate_setup(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Validasi dengan auto-fix untuk komponen yang hilang."""
        try:
            # Basic validation
            if not isinstance(ui_components, dict):
                self.logger.error("❌ UI components bukan dictionary")
                return {'valid': False, 'message': "Invalid UI components structure - bukan dictionary"}
            
            # Pastikan ada kunci 'ui'
            if 'ui' not in ui_components:
                self.logger.warning("⚠️ UI components tidak memiliki kunci 'ui'")
                # Coba tambahkan main_container sebagai ui jika ada
                if 'main_container' in ui_components:
                    self.logger.info("🔧 Menggunakan main_container sebagai ui")
                    ui_components['ui'] = ui_components['main_container']
                else:
                    self.logger.error("❌ Tidak dapat menemukan main_container untuk dijadikan ui")
                    return {'valid': False, 'message': "Invalid UI components structure - tidak ada kunci 'ui' atau 'main_container'"}
            
            # Pastikan komponen kritis tersedia
            self._ensure_required_components(ui_components, {})
            
            # Periksa kembali komponen kritis setelah auto-fix
            critical_components = ['download_button', 'workspace_field', 'project_field', 'version_field']
            missing = [comp for comp in critical_components if comp not in ui_components]
            
            if missing:
                self.logger.warning(f"⚠️ Komponen kritis masih hilang setelah auto-fix: {', '.join(missing)}")
                # Buat komponen yang hilang
                from smartcash.ui.dataset.downloader.components.form_fields import create_form_fields
                from smartcash.ui.dataset.downloader.components.action_buttons import create_action_buttons
                
                form_components = create_form_fields({})
                action_components = create_action_buttons()
                
                for comp in missing:
                    if comp in form_components:
                        ui_components[comp] = form_components[comp]
                        self.logger.info(f"🔧 Menambahkan komponen {comp} dari form_components")
                    elif comp in action_components:
                        ui_components[comp] = action_components[comp]
                        self.logger.info(f"🔧 Menambahkan komponen {comp} dari action_components")
            
            # Ensure logger
            if 'logger' not in ui_components:
                ui_components['logger'] = self.logger
            
            # Ensure config_handler
            if 'config_handler' not in ui_components:
                from smartcash.ui.dataset.downloader.handlers.downloader_config_handler import DownloaderConfigHandler
                ui_components['config_handler'] = DownloaderConfigHandler()
                self.logger.info("🔧 Menambahkan config_handler default")
            
            self.logger.success("✅ Component validation passed")
            return {'valid': True, 'message': "Validation successful"}
            
        except Exception as e:
            self.logger.error(f"❌ Validation error: {str(e)}")
            import traceback
            self.logger.debug(f"Stack trace: {traceback.format_exc()}")
            return {'valid': False, 'message': f"Validation error: {str(e)}"}
    
    def _finalize_setup(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Minimal finalization."""
        # Ensure version is set
        ui_components['version'] = ui_components.get('version', '1.0.0')
        
        # Add module status
        ui_components['module_status'] = self.get_module_status()
        
        # Ensure logger is set
        if 'logger' not in ui_components:
            ui_components['logger'] = self.logger
        
        # Auto-detect API key (optional enhancement)
        try:
            if 'api_key_field' in ui_components and not ui_components['api_key_field'].value:
                from smartcash.common.environment import get_environment_manager
                env_manager = get_environment_manager()
                api_key = env_manager.get_roboflow_api_key()
                
                if api_key:
                    ui_components['api_key_field'].value = api_key
                    self.logger.info("🔑 API key auto-detected")
        except Exception:
            pass  # Silent fail for optional feature
        
        # Update supported formats (optional enhancement)
        try:
            if 'format_dropdown' in ui_components:
                from smartcash.dataset.roboflow.constants import SUPPORTED_FORMATS
                ui_components['format_dropdown'].options = SUPPORTED_FORMATS
        except Exception:
            pass  # Silent fail for optional feature
            
        self.logger.success("✅ Downloader setup completed")
        
    def _create_complete_components(self, widget) -> Dict[str, Any]:
        """Membuat dictionary komponen lengkap dari widget.
        
        Args:
            widget: Widget yang akan dibungkus dalam dictionary
            
        Returns:
            Dictionary berisi UI components lengkap
        """
        try:
            self.logger.info("🔧 Membuat komponen UI lengkap dari widget")
            
            # Import komponen yang diperlukan
            from smartcash.ui.dataset.downloader.components.form_fields import create_form_fields
            from smartcash.ui.dataset.downloader.components.action_buttons import create_action_buttons
            from smartcash.ui.components.progress_tracking import create_progress_tracking_container
            from smartcash.ui.dataset.downloader.handlers.downloader_config_handler import DownloaderConfigHandler
            
            # Buat komponen dasar
            config_handler = DownloaderConfigHandler()
            default_config = config_handler.get_default_config()
            
            # Buat komponen UI dasar
            form_components = create_form_fields(default_config)
            action_components = create_action_buttons()
            progress_components = create_progress_tracking_container()
            
            # Buat dictionary hasil dengan semua komponen yang diharapkan
            complete_result = {
                'ui': widget,  # Widget yang sudah dibuat
                'main_container': widget,
                'version': '1.0.0',
                # Form fields
                'workspace_field': form_components.get('workspace_field'),
                'project_field': form_components.get('project_field'),
                'version_field': form_components.get('version_field'),
                'api_key_field': form_components.get('api_key_field'),
                # Action buttons
                'download_button': action_components.get('download_button'),
                'validate_button': action_components.get('validate_button'),
                'quick_validate_button': action_components.get('quick_validate_button'),
                'save_button': action_components.get('save_button'),
                'reset_button': action_components.get('reset_button'),
                # Progress components
                'progress_bar': progress_components.get('progress_bar'),
                'progress_text': progress_components.get('progress_text'),
                'progress_container': progress_components.get('container'),
                # Config handler
                'config_handler': config_handler,
                'logger': self.logger
            }
            
            self.logger.success("✅ Berhasil membuat komponen UI lengkap")
            return complete_result
            
        except Exception as e:
            self.logger.error(f"❌ Gagal membuat komponen UI lengkap: {str(e)}")
            import traceback
            self.logger.debug(f"Stack trace: {traceback.format_exc()}")
            
            # Fallback ke dictionary sederhana
            return {'ui': widget, 'version': '1.0.0', 'logger': self.logger}

# Singleton instance
_downloader_initializer = DownloaderInitializer()

def initialize_downloader_ui(env=None, config=None, **kwargs):
    """Streamlined initialization dengan error handling yang lebih baik.
    
    Args:
        env: Environment manager instance
        config: Configuration dictionary atau False untuk config default
        **kwargs: Additional parameters
        
    Returns:
        Dictionary berisi UI components
    """
    from smartcash.common.logger import get_logger
    from smartcash.ui.utils.fallback_utils import create_fallback_ui
    
    logger = get_logger('downloader.initializer')
    logger.info(f"🔄 Initializing downloader UI dengan config: {config}")
    
    try:
        # Create initializer
        initializer = DownloaderInitializer()
        
        # Create UI components
        ui_result = initializer._create_ui_components(config)
        
        if ui_result is None:
            logger.error("❌ UI components creation failed")
            raise ValueError("Failed to create UI components")
        
        logger.success("✅ UI components created successfully")
        
        # Handle non-dictionary results (direct widget return)
        if not isinstance(ui_result, dict):
            logger.info(f"🔍 Result bukan dictionary, tipe: {type(ui_result)}")
            logger.info("🔧 Membungkus widget dalam dictionary dengan komponen lengkap")
            
            # Create complete components
            ui_components = initializer._create_complete_components(ui_result)
            logger.success("✅ Berhasil membuat UI components lengkap")
            return ui_components
        
        # Debug info untuk keys
        if isinstance(ui_result, dict):
            logger.info(f"🔍 Result keys: {', '.join(ui_result.keys())}")
            
            if 'ui' not in ui_result:
                logger.error("❌ UI component missing from initialization result")
                
                # Coba perbaiki result jika ada main_container
                if 'main_container' in ui_result:
                    logger.info("🔧 Menggunakan main_container sebagai ui")
                    ui_result['ui'] = ui_result['main_container']
                else:
                    return create_fallback_ui("Gagal membuat UI components untuk downloader: tidak ada UI component", "downloader")
            
            # Tambahkan versi
            ui_result['version'] = ui_result.get('downloader_version', '1.0.0')
            logger.success("✅ Downloader UI initialized successfully")
            return ui_result
        else:
            logger.error(f"❌ Invalid initialization result type: {type(ui_result)}")
            return create_fallback_ui(f"Gagal membuat UI components untuk downloader: tipe hasil {type(ui_result)}", "downloader")
        
    except Exception as e:
        logger.error(f"❌ Downloader initialization error: {str(e)}")
        import traceback
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return create_fallback_ui(f"Gagal membuat UI components untuk downloader: {str(e)}", "downloader")

def get_downloader_status():
    """Get current downloader status."""
    try:
        return _downloader_initializer.get_module_status()
    except Exception as e:
        return {
            'module_name': 'downloader',
            'initialized': False,
            'error': str(e),
            'status': 'error'
        }

def validate_downloader_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate downloader configuration."""
    try:
        from smartcash.ui.dataset.downloader.handlers.config_extractor import DownloaderConfigExtractor
        return DownloaderConfigExtractor.validate_extracted_config(config)
    except Exception as e:
        return {
            'valid': False,
            'errors': [f"Validation error: {str(e)}"],
            'warnings': []
        }