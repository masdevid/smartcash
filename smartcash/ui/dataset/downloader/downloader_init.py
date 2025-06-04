"""
File: smartcash/ui/dataset/downloader/downloader_init.py
Deskripsi: Streamlined downloader initializer dengan reduced fallbacks
"""

from typing import Dict, Any, List
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.dataset.downloader.handlers.downloader_config_handler import DownloaderConfigHandler
from smartcash.ui.dataset.downloader.components.main_ui import create_downloader_ui
from smartcash.ui.dataset.downloader.handlers.download_handler import setup_download_handlers
from smartcash.ui.dataset.downloader.handlers.validation_handler import validate_download_parameters

class DownloaderInitializer(CommonInitializer):
    """Downloader initializer dengan streamlined error handling."""
    
    def __init__(self):
        super().__init__('downloader', DownloaderConfigHandler, parent_module='dataset')
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create UI components dengan essential validation dan auto-fix untuk komponen yang hilang."""
        try:
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
        """Setup handlers with streamlined validation."""
        # Essential validation only
        required_components = ['download_button', 'workspace_field', 'project_field']
        missing = [comp for comp in required_components if comp not in ui_components]
        
        if missing:
            raise ValueError(f"Critical components missing: {', '.join(missing)}")
        
        try:
            self.logger.info("📝 Setting up downloader handlers...")
            
            # Setup handlers
            handlers_result = setup_download_handlers(ui_components, config, env)
            
            if not handlers_result.get('success', False):
                raise RuntimeError(handlers_result.get('message', 'Handler setup failed'))
            
            self.logger.success("✅ Handlers setup completed")
            
            # Optional auto-validation
            results = {'success': True, 'message': handlers_result.get('message'), 'valid': True}
            
            if config.get('auto_validate', False):
                try:
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
            raise
    
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
        """Streamlined validation."""
        # Parent validation
        base_result = super()._validate_setup(ui_components)
        if not base_result['valid']:
            return base_result
        
        # Essential checks only
        if not isinstance(ui_components, dict) or 'ui' not in ui_components:
            return {'valid': False, 'message': "Invalid UI components structure"}
        
        # Critical components check
        critical_missing = []
        if 'download_button' not in ui_components:
            critical_missing.append('download_button')
        
        form_fields = ['workspace_field', 'project_field', 'version_field']
        missing_form = [f for f in form_fields if f not in ui_components]
        if len(missing_form) > 1:  # Allow one missing field
            critical_missing.extend(missing_form)
        
        if critical_missing:
            return {
                'valid': False, 
                'message': f"Critical components missing: {', '.join(critical_missing)}"
            }
        
        # Ensure logger
        if 'logger' not in ui_components:
            ui_components['logger'] = self.logger
        
        self.logger.success("✅ Component validation passed")
        return {'valid': True, 'message': "Validation successful"}
    
    def _finalize_setup(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Minimal finalization."""
        # Ensure logger
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

# Singleton instance
_downloader_initializer = DownloaderInitializer()

def initialize_downloader_ui(env=None, config=None, **kwargs):
    """Streamlined initialization dengan error handling yang lebih baik."""
    from smartcash.common.logger import get_logger
    from smartcash.ui.utils.fallback_utils import create_fallback_ui
    
    logger = get_logger('downloader.initializer')
    
    try:
        # Optional config validation
        if config is not None:
            try:
                from smartcash.ui.dataset.downloader.handlers.config_extractor import DownloaderConfigExtractor
                validation = DownloaderConfigExtractor.validate_extracted_config(config)
                if not validation.get('valid', False):
                    logger.warning(f"⚠️ Config issues detected, using defaults as fallback")
                    config = None
            except Exception as e:
                logger.info(f"ℹ️ Config validation skipped: {str(e)}")
        
        # Initialize dengan debug info
        logger.info(f"🔄 Initializing downloader UI dengan config: {config is not None}")
        result = _downloader_initializer.initialize(env=env, config=config, **kwargs)
        
        # Debug info
        if result is None:
            logger.error("❌ Initialization result is None")
            return create_fallback_ui("Gagal membuat UI components untuk downloader: hasil inisialisasi None", "downloader")
            
        # Basic result validation dengan debug info
        if not isinstance(result, dict):
            logger.info(f"🔍 Result bukan dictionary, tipe: {type(result)}")
            # Jika result adalah widget, bungkus dalam dictionary dengan semua komponen yang diharapkan
            import ipywidgets as widgets
            if isinstance(result, widgets.Widget):
                logger.info("🔧 Membungkus widget dalam dictionary dengan komponen lengkap")
                # Buat dictionary dengan semua komponen yang diharapkan
                from smartcash.ui.dataset.downloader.components.form_fields import create_form_fields
                from smartcash.ui.dataset.downloader.components.action_buttons import create_action_buttons
                from smartcash.ui.components.progress_tracking import create_progress_tracking_container
                
                # Buat komponen dasar
                default_config = {}
                try:
                    from smartcash.ui.dataset.downloader.handlers.downloader_config_handler import DownloaderConfigHandler
                    config_handler = DownloaderConfigHandler()
                    default_config = config_handler.get_default_config()
                except Exception as e:
                    logger.warning(f"⚠️ Gagal mendapatkan default config: {str(e)}")
                
                # Buat komponen UI dasar
                try:
                    form_components = create_form_fields(default_config)
                    action_components = create_action_buttons()
                    progress_components = create_progress_tracking_container()
                    
                    # Buat dictionary hasil dengan semua komponen yang diharapkan
                    complete_result = {
                        'ui': result,  # Widget yang sudah dibuat
                        'main_container': result,
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
                        'config_handler': config_handler if 'config_handler' in locals() else None
                    }
                    
                    logger.success("✅ Berhasil membuat UI components lengkap")
                    return complete_result
                    
                except Exception as e:
                    logger.error(f"❌ Gagal membuat komponen UI dasar: {str(e)}")
                    # Fallback ke dictionary sederhana
                    return {'ui': result, 'version': '1.0.0'}
            else:
                logger.error(f"❌ Invalid initialization result type: {type(result)}")
                return create_fallback_ui(f"Gagal membuat UI components untuk downloader: tipe hasil {type(result)}", "downloader")
        
        # Debug info untuk keys
        logger.info(f"🔍 Result keys: {', '.join(result.keys())}")
            
        if 'ui' not in result:
            logger.error("❌ UI component missing from initialization result")
            
            # Coba perbaiki result jika ada main_container
            if 'main_container' in result:
                logger.info("🔧 Menggunakan main_container sebagai ui")
                result['ui'] = result['main_container']
            else:
                return create_fallback_ui("Gagal membuat UI components untuk downloader: tidak ada UI component", "downloader")
        
        # Tambahkan versi
        result['version'] = result.get('downloader_version', '1.0.0')
        logger.success("✅ Downloader UI initialized successfully")
        return result
        
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