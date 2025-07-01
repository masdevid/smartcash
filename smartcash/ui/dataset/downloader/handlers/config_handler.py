"""
File: smartcash/ui/dataset/downloader/handlers/config_handler.py
Deskripsi: Fixed config handler dengan proper reset functionality dan error handling
"""

from typing import Dict, Any, Optional, List, Callable
from functools import wraps

from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.ui.utils.ui_logger import get_module_logger
from smartcash.ui.handlers.error_handler import create_error_response, handle_ui_errors
from smartcash.ui.dataset.downloader.handlers.config_extractor import extract_downloader_config
from smartcash.ui.dataset.downloader.handlers.config_updater import update_downloader_ui, validate_ui_inputs
from smartcash.ui.dataset.downloader.utils.colab_secrets import set_api_key_to_config, get_api_key_from_secrets
from smartcash.common.config.manager import get_config_manager


class DownloaderConfigHandler(ConfigHandler):
    """Fixed config handler dengan proper reset dan API key handling"""
    
    def __init__(self, module_name: str = 'downloader', parent_module: str = 'dataset'):
        super().__init__(module_name, parent_module)
        self.config_filename = 'dataset_config.yaml'
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dari downloader UI components"""
        return extract_downloader_config(ui_components)
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dari config"""
        update_downloader_ui(ui_components, config)
    
    @handle_ui_errors(error_component_title="Config Loading Error")
    def load_config(self, config_filename: str = None) -> Dict[str, Any]:
        """Load config dengan API key auto-detection dan fail-fast error handling"""
        # Create error context for better tracing
        ctx = ErrorContext(
            component="DownloaderConfigHandler.load_config",
            operation="load_config"
        )
        
        # Use ErrorHandler for consistent error handling
        handler = ErrorHandler(
            context=ctx,
            logger=self.logger
        )
        
        filename = config_filename or self.config_filename
        
        # Load dari file - fail fast if file access issues
        config = self.config_manager.load_config(filename)
        
        if not config:
            self.logger.info(f"📂 File {filename} tidak ditemukan, menggunakan default")
            config = self.get_default_config()
            
            # Save default ke file untuk pertama kali
            self.config_manager.save_config(config, filename)
            self.logger.info(f"💾 Default config tersimpan ke {filename}")
        
        # Ensure required sections exist with fail-fast validation
        if not isinstance(config, dict):
            error_msg = f"Invalid config format: expected dict, got {type(config)}"
            handler.handle_error(message=error_msg)
            return self.get_default_config()
            
        # Ensure downloader section exists
        if 'downloader' not in config:
            config['downloader'] = {}
            
        # Ensure basic structure exists
        if 'basic' not in config['downloader']:
            default_config = self.get_default_config()
            config['downloader']['basic'] = default_config.get('downloader', {}).get('basic', {})
            
        # Ensure advanced structure exists
        if 'advanced' not in config['downloader']:
            default_config = self.get_default_config()
            config['downloader']['advanced'] = default_config.get('downloader', {}).get('advanced', {})
        
        # Auto-detect dan set API key dari Colab secrets
        try:
            config = set_api_key_to_config(config, force_refresh=False)
        except Exception as e:
            error_msg = f"Error setting API key: {str(e)}"
            handler.handle_error(error=e, message=error_msg, log_level="warning")
            # Continue with existing config
        
        return config
    
    def save_config(self, ui_components: Dict[str, Any], config_filename: str = None, update_shared: bool = True) -> Dict[str, Any]:
        """Save config dengan merge strategy untuk dataset_config.yaml
        
        Returns:
            Dict[str, Any]: Dictionary dengan format {'status': str, 'error': Optional[str]}
        """
        try:
            filename = config_filename or self.config_filename
            
            # Extract current config dari UI
            current_config = self.extract_config(ui_components)
            
            # Validate sebelum save
            validation = self.validate_config(current_config)
            if not validation['valid']:
                error_msg = f"Config tidak valid: {'; '.join(validation['errors'])}"
                self.logger.error(f"❌ {error_msg}")
                return {'status': 'error', 'error': error_msg}
            
            # Load existing config untuk merge
            existing_config = self.config_manager.load_config(filename) or {}
            
            # Merge dengan existing config
            merged_config = self._merge_downloader_config(existing_config, current_config)
            
            # Save ke file
            self.config_manager.save_config(merged_config, filename)
            
            # Update shared config if enabled
            if update_shared and self.use_shared_config and self.shared_manager:
                from smartcash.ui.config_cell.managers.shared_config_manager import broadcast_config_update
                broadcast_config_update(self.parent_module, self.module_name, current_config)
                self.logger.debug(f"Shared config updated for {self.full_module_name}")
            
            # Update local state
            self._config_state.update(current_config)
            
            # Execute callbacks
            self._notify_callbacks(current_config, 'save')
            
            # Handle success
            self._handle_save_success(ui_components, current_config)
            
            self.logger.info(f"💾 Config saved to {filename}")
            return {'status': 'success'}
            
        except Exception as e:
            error_msg = f"Error saving config: {str(e)}"
            self.logger.error(f"❌ {error_msg}", exc_info=True)
            
            # Handle failure
            self._handle_save_failure(ui_components, error_msg)
            
            return {'status': 'error', 'error': error_msg}
    
    def reset_config(self, ui_components: Dict[str, Any], config_filename: str = None) -> Dict[str, Any]:
        """Enhanced reset config dengan proper UI update dan error handling
        
        Returns:
            Dict[str, Any]: Dictionary dengan format {'status': str, 'error': Optional[str]}
        """
        try:
            # Get default config
            default_config = self.get_default_config()
            
            # Preserve current API key if available
            try:
                current_api_key = getattr(ui_components.get('api_key_input'), 'value', '').strip()
                if current_api_key:
                    # Ensure nested structure exists
                    if 'data' not in default_config:
                        default_config['data'] = {}
                    if 'roboflow' not in default_config['data']:
                        default_config['data']['roboflow'] = {}
                    
                    default_config['data']['roboflow']['api_key'] = current_api_key
                    self.logger.debug("Preserved API key during reset")
            except Exception as e:
                self.logger.debug(f"Could not preserve API key: {str(e)}")
            
            # Update UI with default config
            self.update_ui(ui_components, default_config)
            
            # Reset progress indicators
            self._reset_progress_indicators(ui_components)
            
            # Save default config if filename provided
            if config_filename:
                filename = config_filename
                self.config_manager.save_config(default_config, filename)
                self.logger.info(f"💾 Default config saved to {filename}")
            
            # Update local state
            self._config_state.update(default_config)
            
            # Execute callbacks
            self._notify_callbacks(default_config, 'reset')
            
            # Handle success
            self._handle_reset_success(ui_components, default_config)
            
            self.logger.info("🔄 Config reset to defaults")
            return {'status': 'success'}
            
        except Exception as e:
            error_msg = f"Error resetting config: {str(e)}"
            self.logger.error(f"❌ {error_msg}", exc_info=True)
            
            # Handle failure
            self._handle_reset_failure(ui_components, error_msg)
            
            return {'status': 'error', 'error': error_msg}
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced validation dengan comprehensive checks untuk downloader"""
        errors = []
        warnings = []
        
        # Check required sections
        if 'data' not in config:
            errors.append("Missing 'data' section")
        else:
            data_config = config['data']
            
            # Check data source
            if 'source' not in data_config:
                errors.append("Missing 'data.source'")
            elif data_config['source'] != 'roboflow':
                warnings.append(f"Unsupported data source: {data_config['source']}")
            
            # Check Roboflow config
            if 'roboflow' not in data_config:
                errors.append("Missing 'data.roboflow' section")
            else:
                roboflow_config = data_config['roboflow']
                
                # Check required Roboflow fields
                for field in ['workspace', 'project', 'version', 'api_key']:
                    if field not in roboflow_config:
                        errors.append(f"Missing 'data.roboflow.{field}'")
                    elif not roboflow_config[field] and field != 'api_key':  # API key can be empty if using Colab secrets
                        errors.append(f"Empty 'data.roboflow.{field}'")
        
        # Check download section
        if 'download' not in config:
            errors.append("Missing 'download' section")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def _merge_downloader_config(self, existing: Dict[str, Any], new_downloader: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced merge downloader config dengan existing dataset_config.yaml"""
        # Create a deep copy to avoid modifying the original
        result = existing.copy()
        
        # Ensure downloader section exists
        if 'downloader' not in result:
            result['downloader'] = {}
        
        # Update metadata
        if 'config_version' in new_downloader:
            result['config_version'] = new_downloader['config_version']
        
        if 'updated_at' in new_downloader:
            result['updated_at'] = new_downloader['updated_at']
        
        # Update data section
        if 'data' in new_downloader:
            if 'data' not in result:
                result['data'] = {}
            
            # Update source
            if 'source' in new_downloader['data']:
                result['data']['source'] = new_downloader['data']['source']
            
            # Update roboflow config
            if 'roboflow' in new_downloader['data']:
                if 'roboflow' not in result['data']:
                    result['data']['roboflow'] = {}
                
                for key, value in new_downloader['data']['roboflow'].items():
                    result['data']['roboflow'][key] = value
            
            # Update file naming
            if 'file_naming' in new_downloader['data']:
                if 'file_naming' not in result['data']:
                    result['data']['file_naming'] = {}
                
                for key, value in new_downloader['data']['file_naming'].items():
                    result['data']['file_naming'][key] = value
        
        # Update other sections
        for section in ['download', 'uuid_renaming', 'validation', 'cleanup']:
            if section in new_downloader:
                result[section] = new_downloader[section]
        
        return result
    
    def extract_config_from_ui(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Wrapper untuk extract config dari UI dengan validation"""
        try:
            return self.extract_config(ui_components)
        except Exception as e:
            self.logger.error(f"❌ Error extracting config from UI: {str(e)}", exc_info=True)
            return self.get_default_config()
    
    def get_api_key_status(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Get API key status dengan auto-detection info"""
        try:
            from smartcash.ui.dataset.downloader.utils.colab_secrets import validate_api_key
            
            # Check current UI value
            current_key = getattr(ui_components.get('api_key_input'), 'value', '').strip()
            
            # Check Colab secrets
            detected_key = get_api_key_from_secrets()
            
            if detected_key:
                validation = validate_api_key(detected_key)
                return {
                    'source': 'colab_secret',
                    'valid': validation['valid'],
                    'message': f"Auto-detect dari Colab: {validation['message']}",
                    'key_preview': f"{detected_key[:4]}...{detected_key[-4:]}" if len(detected_key) > 8 else '****'
                }
            elif current_key:
                validation = validate_api_key(current_key)
                return {
                    'source': 'manual_input',
                    'valid': validation['valid'],
                    'message': f"Manual input: {validation['message']}",
                    'key_preview': f"{current_key[:4]}...{current_key[-4:]}" if len(current_key) > 8 else '****'
                }
            else:
                return {
                    'source': 'not_provided',
                    'valid': False,
                    'message': 'API key belum diisi',
                    'key_preview': '****'
                }
                
        except Exception as e:
            return {
                'source': 'error',
                'valid': False,
                'message': f"Error checking API key: {str(e)}",
                'key_preview': '****'
            }
    
    @handle_ui_errors(error_component_title="Default Config Error")
    def get_default_config(self) -> Dict[str, Any]:
        """Get default config dengan optimal workers dan fail-fast error handling"""
        # Create error context for better tracing
        ctx = ErrorContext(
            component="DownloaderConfigHandler.get_default_config",
            operation="get_defaults"
        )
        
        # Use ErrorHandler for consistent error handling
        handler = ErrorHandler(
            context=ctx,
            logger=self.logger
        )
        
        try:
            # Import with fail-fast error handling
            try:
                from smartcash.ui.dataset.downloader.handlers.defaults import get_default_downloader_config
                from smartcash.common.worker_utils import get_worker_counts_for_operations
            except ImportError as e:
                error_msg = f"Failed to import required modules: {str(e)}"
                handler.handle_error(error=e, message=error_msg)
                # Don't return here - let it fail in the next step for better error context
            
            # Get default config with fail-fast error handling
            default_config = get_default_downloader_config()
            
            # Validate default config structure
            if not isinstance(default_config, dict):
                error_msg = f"Invalid default config format: expected dict, got {type(default_config)}"
                handler.handle_error(message=error_msg)
                raise ValueError(error_msg)
                
            # Update with optimal workers using the centralized worker count function
            worker_counts = get_worker_counts_for_operations()
            
            # Apply worker counts with safe dict access
            if 'download' in default_config:
                default_config['download']['max_workers'] = worker_counts.get('download', 4)
                
            if 'uuid_renaming' in default_config:
                default_config['uuid_renaming']['parallel_workers'] = worker_counts.get('uuid_renaming', 2)
            
            # Add missing sections if not present with safe defaults
            if 'validation' not in default_config:
                default_config['validation'] = {
                    'enabled': True,
                    'parallel_workers': worker_counts.get('validation', 2)
                }
            
            if 'cleanup' not in default_config:
                default_config['cleanup'] = {
                    'auto_cleanup_downloads': False,
                    'parallel_workers': worker_counts.get('download', 4)  # Use same as download
                }
            
            return default_config
            
        except Exception as e:
            error_msg = f"Failed to get default config: {str(e)}"
            handler.handle_error(error=e, message=error_msg)
            
            # Instead of having a fallback here, raise the exception to be handled by the decorator
            # This follows the fail-fast approach
            raise
    
    # These methods are already implemented in the parent class and don't need custom behavior
    # Removing unnecessary overrides to follow DRY principle
    
    @handle_ui_errors(error_component_title="Progress Reset Error")
    def _reset_progress_indicators(self, ui_components: Dict[str, Any]) -> None:
        """Reset progress indicators in UI components with fail-fast error handling"""
        # Create error context for better tracing
        ctx = ErrorContext(
            component="DownloaderConfigHandler._reset_progress_indicators",
            operation="reset_progress"
        )
        
        # Use ErrorHandler for consistent error handling
        handler = ErrorHandler(
            context=ctx,
            logger=self.logger,
            log_level="debug"  # Use debug level for UI widget errors
        )
        
        # Validate input
        if not ui_components or not isinstance(ui_components, dict):
            error_msg = "Invalid UI components provided"
            handler.handle_error(message=error_msg)
            return
        
        # Find progress indicators with fail-fast validation
        progress_keys = [key for key in ui_components if isinstance(key, str) and 'progress' in key.lower()]
        
        for key in progress_keys:
            try:
                widget = ui_components[key]
                
                # Hide widget with safe attribute access
                if hasattr(widget, 'layout'):
                    widget.layout.visibility = 'hidden'
                    widget.layout.display = 'none'
                
                # Reset value with safe attribute access
                if hasattr(widget, 'value'):
                    widget.value = 0
                
                # Reset progress tracker with safe method call
                if hasattr(widget, 'reset') and callable(widget.reset):
                    widget.reset()
            except Exception as e:
                error_msg = f"Error resetting {key}: {str(e)}"
                handler.handle_error(error=e, message=error_msg, log_level="debug")
                # Continue with other widgets - don't fail the entire operation
