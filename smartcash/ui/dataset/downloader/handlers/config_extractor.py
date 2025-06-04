"""
File: smartcash/ui/dataset/downloader/handlers/config_extractor.py
Deskripsi: Ekstraksi konfigurasi dari UI components dengan validation dan clean values
"""

from typing import Dict, Any
from smartcash.common.logger import get_logger

class DownloaderConfigExtractor:
    """Static class untuk extract config dari UI components."""
    
    @staticmethod
    def extract_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dari UI components dengan validation."""
        logger = get_logger('downloader.config_extractor')
        
        try:
            config = {
                # Dataset identification
                'workspace': DownloaderConfigExtractor._extract_text(ui_components, 'workspace_field'),
                'project': DownloaderConfigExtractor._extract_text(ui_components, 'project_field'),
                'version': DownloaderConfigExtractor._extract_text(ui_components, 'version_field'),
                'api_key': DownloaderConfigExtractor._extract_text(ui_components, 'api_key_field'),
                
                # Download options
                'output_format': DownloaderConfigExtractor._extract_dropdown(ui_components, 'format_dropdown', 'yolov5pytorch'),
                'validate_download': DownloaderConfigExtractor._extract_checkbox(ui_components, 'validate_checkbox', True),
                'organize_dataset': DownloaderConfigExtractor._extract_checkbox(ui_components, 'organize_checkbox', True),
                'backup_existing': DownloaderConfigExtractor._extract_checkbox(ui_components, 'backup_checkbox', False),
                
                # Progress options
                'progress_enabled': DownloaderConfigExtractor._extract_checkbox(ui_components, 'progress_checkbox', True),
                'show_detailed_progress': DownloaderConfigExtractor._extract_checkbox(ui_components, 'detailed_progress_checkbox', False),
                
                # Advanced options
                'retry_attempts': DownloaderConfigExtractor._extract_int(ui_components, 'retry_field', 3),
                'timeout_seconds': DownloaderConfigExtractor._extract_int(ui_components, 'timeout_field', 30),
                'chunk_size_kb': DownloaderConfigExtractor._extract_int(ui_components, 'chunk_size_field', 8)
            }
            
            # Clean dan validate values
            config = DownloaderConfigExtractor._clean_config(config)
            
            logger.debug(f"📋 Config extracted: {len(config)} parameters")
            return config
            
        except Exception as e:
            logger.error(f"❌ Config extraction error: {str(e)}")
            return DownloaderConfigExtractor._get_fallback_config()
    
    @staticmethod
    def _extract_text(ui_components: Dict[str, Any], key: str, default: str = '') -> str:
        """Extract text value dengan cleaning."""
        try:
            widget = ui_components.get(key)
            if widget and hasattr(widget, 'value'):
                value = str(widget.value or '').strip()
                return value if value else default
        except Exception:
            pass
        return default
    
    @staticmethod
    def _extract_checkbox(ui_components: Dict[str, Any], key: str, default: bool = False) -> bool:
        """Extract checkbox value."""
        try:
            widget = ui_components.get(key)
            if widget and hasattr(widget, 'value'):
                return bool(widget.value)
        except Exception:
            pass
        return default
    
    @staticmethod
    def _extract_dropdown(ui_components: Dict[str, Any], key: str, default: str = '') -> str:
        """Extract dropdown value."""
        try:
            widget = ui_components.get(key)
            if widget and hasattr(widget, 'value'):
                return str(widget.value or default)
        except Exception:
            pass
        return default
    
    @staticmethod
    def _extract_int(ui_components: Dict[str, Any], key: str, default: int = 0) -> int:
        """Extract integer value dengan validation."""
        try:
            widget = ui_components.get(key)
            if widget and hasattr(widget, 'value'):
                return max(0, int(float(widget.value or default)))
        except (ValueError, TypeError):
            pass
        return default
    
    @staticmethod
    def _clean_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Clean dan normalize config values."""
        cleaned = config.copy()
        
        # Clean workspace dan project names
        for key in ['workspace', 'project']:
            if cleaned.get(key):
                cleaned[key] = cleaned[key].lower().replace(' ', '-').replace('_', '-')
        
        # Normalize version
        if cleaned.get('version'):
            cleaned['version'] = str(cleaned['version']).strip()
        
        # Validate numeric values
        cleaned['retry_attempts'] = max(1, min(10, cleaned.get('retry_attempts', 3)))
        cleaned['timeout_seconds'] = max(10, min(300, cleaned.get('timeout_seconds', 30)))
        cleaned['chunk_size_kb'] = max(1, min(64, cleaned.get('chunk_size_kb', 8)))
        
        return cleaned
    
    @staticmethod
    def _get_fallback_config() -> Dict[str, Any]:
        """Fallback config jika extraction gagal."""
        return {
            'workspace': 'smartcash-wo2us',
            'project': 'rupiah-emisi-2022',
            'version': '3',
            'api_key': '',
            'output_format': 'yolov5pytorch',
            'validate_download': True,
            'organize_dataset': True,
            'backup_existing': False,
            'progress_enabled': True,
            'show_detailed_progress': False,
            'retry_attempts': 3,
            'timeout_seconds': 30,
            'chunk_size_kb': 8
        }
    
    @staticmethod
    def validate_extracted_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate extracted config dan return validation result."""
        errors, warnings = [], []
        
        # Required fields validation
        required_fields = ['workspace', 'project', 'version']
        for field in required_fields:
            if not config.get(field):
                errors.append(f"{field.title()} tidak boleh kosong")
        
        # API key validation (optional tapi warning jika kosong)
        if not config.get('api_key'):
            warnings.append("API key kosong - akan mencoba dari environment")
        
        # Format validation
        valid_formats = ['yolov5pytorch', 'yolov8', 'coco', 'createml']
        if config.get('output_format') not in valid_formats:
            warnings.append(f"Format tidak dikenali: {config.get('output_format')}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'config': config
        }