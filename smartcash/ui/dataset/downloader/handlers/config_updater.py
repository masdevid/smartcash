"""
File: smartcash/ui/dataset/downloader/handlers/config_updater.py
Deskripsi: Update UI components dari loaded config dengan safe value setting
"""

from typing import Dict, Any
from smartcash.common.logger import get_logger

class DownloaderConfigUpdater:
    """Static class untuk update UI components dari config."""
    
    @staticmethod
    def update_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI components dari config dengan safe setting."""
        logger = get_logger('downloader.config_updater')
        
        try:
            # Dataset identification fields
            field_mappings = {
                'workspace': 'workspace_field',
                'project': 'project_field', 
                'version': 'version_field',
                'api_key': 'api_key_field'
            }
            
            # Update text fields
            for config_key, ui_key in field_mappings.items():
                DownloaderConfigUpdater._update_text_field(ui_components, ui_key, config.get(config_key, ''))
            
            # Update dropdown fields
            DownloaderConfigUpdater._update_dropdown(ui_components, 'format_dropdown', config.get('output_format', 'yolov5pytorch'))
            
            # Update checkbox fields
            checkbox_mappings = {
                'validate_download': 'validate_checkbox',
                'organize_dataset': 'organize_checkbox',
                'backup_existing': 'backup_checkbox',
                'progress_enabled': 'progress_checkbox',
                'show_detailed_progress': 'detailed_progress_checkbox'
            }
            
            for config_key, ui_key in checkbox_mappings.items():
                DownloaderConfigUpdater._update_checkbox(ui_components, ui_key, config.get(config_key, False))
            
            # Update numeric fields
            numeric_mappings = {
                'retry_attempts': 'retry_field',
                'timeout_seconds': 'timeout_field', 
                'chunk_size_kb': 'chunk_size_field'
            }
            
            for config_key, ui_key in numeric_mappings.items():
                DownloaderConfigUpdater._update_numeric_field(ui_components, ui_key, config.get(config_key, 0))
            
            logger.debug(f"✅ UI updated dari config dengan {len(config)} parameters")
            
        except Exception as e:
            logger.error(f"❌ UI update error: {str(e)}")
    
    @staticmethod
    def _update_text_field(ui_components: Dict[str, Any], ui_key: str, value: str) -> None:
        """Update text field dengan safe value setting."""
        try:
            widget = ui_components.get(ui_key)
            if widget and hasattr(widget, 'value'):
                widget.value = str(value or '')
        except Exception:
            pass
    
    @staticmethod
    def _update_checkbox(ui_components: Dict[str, Any], ui_key: str, value: bool) -> None:
        """Update checkbox dengan safe value setting."""
        try:
            widget = ui_components.get(ui_key)
            if widget and hasattr(widget, 'value'):
                widget.value = bool(value)
        except Exception:
            pass
    
    @staticmethod
    def _update_dropdown(ui_components: Dict[str, Any], ui_key: str, value: str) -> None:
        """Update dropdown dengan safe value setting."""
        try:
            widget = ui_components.get(ui_key)
            if widget and hasattr(widget, 'value') and hasattr(widget, 'options'):
                if value in widget.options:
                    widget.value = value
                elif widget.options:
                    widget.value = widget.options[0]  # Default ke option pertama
        except Exception:
            pass
    
    @staticmethod
    def _update_numeric_field(ui_components: Dict[str, Any], ui_key: str, value: int) -> None:
        """Update numeric field dengan safe value setting dan validation."""
        try:
            widget = ui_components.get(ui_key)
            if widget and hasattr(widget, 'value'):
                # Validate range berdasarkan field type
                validated_value = DownloaderConfigUpdater._validate_numeric_value(ui_key, value)
                widget.value = validated_value
        except Exception:
            pass
    
    @staticmethod
    def _validate_numeric_value(field_key: str, value: int) -> int:
        """Validate numeric value berdasarkan field constraints."""
        validation_rules = {
            'retry_field': (1, 10, 3),      # (min, max, default)
            'timeout_field': (10, 300, 30),
            'chunk_size_field': (1, 64, 8)
        }
        
        if field_key in validation_rules:
            min_val, max_val, default_val = validation_rules[field_key]
            if not isinstance(value, (int, float)) or value < min_val or value > max_val:
                return default_val
            return int(value)
        
        return max(0, int(value or 0))
    
    @staticmethod
    def update_ui_from_environment(ui_components: Dict[str, Any]) -> None:
        """Update UI dari environment variables dan detected values."""
        logger = get_logger('downloader.config_updater')
        
        try:
            # Auto-detect API key dari environment
            api_key = DownloaderConfigUpdater._detect_api_key()
            if api_key and not ui_components.get('api_key_field', {}).get('value'):
                DownloaderConfigUpdater._update_text_field(ui_components, 'api_key_field', api_key)
                logger.info("🔑 API key terdeteksi dari environment")
            
            # Update status indicators
            DownloaderConfigUpdater._update_status_indicators(ui_components)
            
        except Exception as e:
            logger.warning(f"⚠️ Environment update error: {str(e)}")
    
    @staticmethod
    def _detect_api_key() -> str:
        """Detect API key dari environment variables."""
        import os
        
        # Check environment variables
        env_keys = ['ROBOFLOW_API_KEY', 'ROBOFLOW_KEY', 'RF_API_KEY']
        for env_key in env_keys:
            api_key = os.environ.get(env_key, '').strip()
            if api_key and len(api_key) > 10:
                return api_key
        
        # Check Google Colab userdata
        try:
            from google.colab import userdata
            for key_name in ['ROBOFLOW_API_KEY', 'roboflow_api_key', 'API_KEY']:
                try:
                    api_key = userdata.get(key_name, '').strip()
                    if api_key and len(api_key) > 10:
                        return api_key
                except Exception:
                    continue
        except ImportError:
            pass
        
        return ''
    
    @staticmethod
    def _update_status_indicators(ui_components: Dict[str, Any]) -> None:
        """Update status indicators berdasarkan current state."""
        # Update environment status
        try:
            from smartcash.common.environment import get_environment_manager
            env_manager = get_environment_manager()
            
            status_text = "🔗 Drive Connected" if env_manager.is_drive_mounted else "⚠️ Local Storage"
            if 'env_status' in ui_components:
                ui_components['env_status'].value = f"<span style='color: {'green' if env_manager.is_drive_mounted else 'orange'};'>{status_text}</span>"
        except Exception:
            pass
    
    @staticmethod
    def reset_ui_to_defaults(ui_components: Dict[str, Any]) -> None:
        """Reset UI ke default values."""
        try:
            from smartcash.ui.dataset.downloader.handlers.defaults import DEFAULT_CONFIG
            DownloaderConfigUpdater.update_ui(ui_components, DEFAULT_CONFIG)
        except ImportError:
            # Fallback reset
            DownloaderConfigUpdater.update_ui(ui_components, {
                'workspace': 'smartcash-wo2us',
                'project': 'rupiah-emisi-2022', 
                'version': '3',
                'output_format': 'yolov5pytorch',
                'validate_download': True,
                'organize_dataset': True,
                'backup_existing': False,
                'retry_attempts': 3,
                'timeout_seconds': 30,
                'chunk_size_kb': 8
            })