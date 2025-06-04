"""
File: smartcash/ui/dataset/downloader/downloader_init.py
Deskripsi: Complete downloader initializer dengan proper config inheritance dan error handling
"""

from typing import Dict, Any, List
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.dataset.downloader.handlers.downloader_config_handler import DownloaderConfigHandler
from smartcash.ui.dataset.downloader.components.main_ui import create_downloader_ui
from smartcash.ui.dataset.downloader.handlers.download_handler import setup_download_handlers
from smartcash.ui.dataset.downloader.handlers.validation_handler import validate_download_parameters
from smartcash.ui.dataset.downloader.handlers.progress_handler import setup_progress_handlers

class DownloaderInitializer(CommonInitializer):
    """Downloader initializer dengan CommonInitializer pattern dan config inheritance."""
    
    def __init__(self):
        super().__init__('downloader', DownloaderConfigHandler, parent_module='dataset')
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create UI components dengan responsive layout."""
        try:
            return create_downloader_ui(config, env)
        except Exception as e:
            self.logger.error(f"❌ UI creation error: {str(e)}")
            # Fallback UI
            import ipywidgets as widgets
            return {
                'ui': widgets.VBox([
                    widgets.HTML(f"<h3>⚠️ Downloader (Error Mode)</h3>"),
                    widgets.HTML(f"<p style='color:red;'>Error: {str(e)}</p>")
                ])
            }
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers dengan progress callback integration."""
        try:
            results = {}
            
            # Setup progress handlers dengan callback
            progress_result = setup_progress_handlers(ui_components)
            results.update(progress_result)
            
            # Setup download handlers dengan validation dan progress
            download_result = setup_download_handlers(ui_components, env, config)
            results.update(download_result)
            
            # Setup initial validation jika diminta
            if config.get('auto_validate', False):
                try:
                    validation_result = validate_download_parameters(ui_components, include_api_test=False)
                    if validation_result['valid']:
                        self.logger.info("✅ Initial validation passed")
                    else:
                        self.logger.warning(f"⚠️ Initial validation issues: {'; '.join(validation_result.get('errors', []))}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Initial validation error: {str(e)}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Handler setup error: {str(e)}")
            return {'handlers_error': str(e)}
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default config dari defaults.py dengan inheritance support."""
        try:
            from smartcash.ui.dataset.downloader.handlers.defaults import DEFAULT_CONFIG
            return DEFAULT_CONFIG.copy()
        except ImportError:
            self.logger.warning("⚠️ defaults.py tidak ditemukan, menggunakan fallback config")
            return self._fallback_config()
    
    def _fallback_config(self) -> Dict[str, Any]:
        """Fallback config jika defaults.py tidak ada."""
        return {
            '_base_': ['base_config'],  # Inherit dari base_config.yaml
            
            # Dataset identification
            'workspace': 'smartcash-wo2us',
            'project': 'rupiah-emisi-2022',
            'version': '3',
            'api_key': '',
            
            # Download options
            'output_format': 'yolov5pytorch',
            'validate_download': True,
            'organize_dataset': True,
            'backup_existing': False,
            
            # Progress options
            'progress_enabled': True,
            'show_detailed_progress': False,
            
            # Performance options
            'retry_attempts': 3,
            'timeout_seconds': 30,
            'chunk_size_kb': 8,
            
            # Metadata
            'module_name': 'downloader',
            'version': '1.0.0',
            'created_by': 'SmartCash Dataset Downloader'
        }
    
    def _get_critical_components(self) -> List[str]:
        """Critical components yang harus ada."""
        return [
            'ui', 'download_button', 'validate_button', 'workspace_field',
            'project_field', 'version_field', 'api_key_field'
        ]
    
    def _validate_setup(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced validation dengan downloader-specific checks."""
        # Base validation
        base_result = super()._validate_setup(ui_components)
        
        if not base_result['valid']:
            return base_result
        
        # Downloader-specific validation
        additional_checks = []
        
        # Check progress components
        if 'tracker' not in ui_components and 'progress_manager' not in ui_components:
            additional_checks.append("Progress tracking components")
        
        # Check form fields
        form_fields = ['workspace_field', 'project_field', 'version_field', 'api_key_field']
        missing_fields = [field for field in form_fields if field not in ui_components]
        if missing_fields:
            additional_checks.extend(missing_fields)
        
        if additional_checks:
            return {
                'valid': False,
                'message': f"Downloader validation failed: {', '.join(additional_checks)}"
            }
        
        return {'valid': True, 'message': "Downloader validation passed"}
    
    def _finalize_setup(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Enhanced finalize dengan downloader-specific setup."""
        super()._finalize_setup(ui_components, config)
        
        # Auto-detect API key jika tersedia
        try:
            from smartcash.ui.dataset.downloader.handlers.config_updater import DownloaderConfigUpdater
            DownloaderConfigUpdater.update_ui_from_environment(ui_components)
        except Exception as e:
            self.logger.warning(f"⚠️ Environment update warning: {str(e)}")
        
        # Set downloader-specific metadata
        ui_components.update({
            'downloader_version': '1.0.0',
            'supported_formats': ['yolov5pytorch', 'yolov8', 'coco', 'createml'],
            'api_provider': 'roboflow'
        })

# Singleton instance dan public API
_downloader_initializer = DownloaderInitializer()

def initialize_downloader_ui(env=None, config=None, **kwargs):
    """Public API untuk initialize downloader UI dengan comprehensive error handling."""
    try:
        return _downloader_initializer.initialize(env=env, config=config, **kwargs)
    except Exception as e:
        # Fallback error UI
        import ipywidgets as widgets
        from smartcash.common.logger import get_logger
        
        logger = get_logger('downloader.init_fallback')
        logger.error(f"❌ Critical downloader initialization error: {str(e)}")
        
        error_ui = widgets.VBox([
            widgets.HTML("<h3 style='color:red;'>⚠️ Downloader Initialization Failed</h3>"),
            widgets.HTML(f"<p><strong>Error:</strong> {str(e)}</p>"),
            widgets.HTML("<p><em>💡 Try restarting the cell atau check dependencies</em></p>")
        ], layout=widgets.Layout(padding='20px', border='1px solid red', border_radius='5px'))
        
        return {'ui': error_ui, 'error': str(e), 'fallback_mode': True}

def get_downloader_status():
    """Get downloader status untuk debugging."""
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
    """Validate downloader config tanpa UI."""
    try:
        from smartcash.ui.dataset.downloader.handlers.config_extractor import DownloaderConfigExtractor
        return DownloaderConfigExtractor.validate_extracted_config(config)
    except Exception as e:
        return {
            'valid': False,
            'errors': [f"Config validation error: {str(e)}"],
            'warnings': []
        }