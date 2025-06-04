"""
File: smartcash/ui/dataset/downloader/handlers/downloader_config_handler.py
Deskripsi: ConfigHandler untuk downloader module dengan integration ke ConfigExtractor/Updater
"""

from typing import Dict, Any
from smartcash.ui.handlers.config_handlers import ConfigHandler
from .config_extractor import DownloaderConfigExtractor
from .config_updater import DownloaderConfigUpdater
from .defaults import DEFAULT_CONFIG

class DownloaderConfigHandler(ConfigHandler):
    """ConfigHandler untuk downloader dengan integration ke extractor/updater pattern."""
    
    def __init__(self, module_name: str = 'downloader', parent_module: str = 'dataset'):
        super().__init__(module_name)
        self.parent_module = parent_module
        
        # Set config filename untuk load_config
        self.config_filename = f"{module_name}_config.yaml"
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config menggunakan DownloaderConfigExtractor."""
        return DownloaderConfigExtractor.extract_config(ui_components)
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI menggunakan DownloaderConfigUpdater."""
        return DownloaderConfigUpdater.update_ui(ui_components, config)
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default config dari defaults.py."""
        return DEFAULT_CONFIG.copy()
    
    def after_save_success(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Override dengan downloader-specific success handling."""
        super().after_save_success(ui_components, config)
        
        # Update environment indicators
        DownloaderConfigUpdater.update_ui_from_environment(ui_components)
        
        # Log saved parameters
        self.logger.info(f"📋 Saved downloader config:")
        self.logger.info(f"   • Dataset: {config.get('workspace')}/{config.get('project')}:{config.get('version')}")
        self.logger.info(f"   • Format: {config.get('output_format')}")
        self.logger.info(f"   • Options: validate={config.get('validate_download')}, organize={config.get('organize_dataset')}")
    
    def after_reset_success(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Override dengan downloader-specific reset handling."""
        super().after_reset_success(ui_components, config)
        
        # Auto-detect API key after reset
        DownloaderConfigUpdater.update_ui_from_environment(ui_components)
        
        self.logger.info(f"🔄 Reset to default config: {config.get('workspace')}/{config.get('project')}")
    
    def load_config(self, config_name: str = None, use_base_config: bool = True) -> Dict[str, Any]:
        """Load config dengan downloader-specific handling."""
        config_name = config_name or self.config_filename or 'downloader_config'
        
        # Gunakan parent method untuk loading
        loaded_config = super().load_config(config_name, use_base_config)
        
        # Merge dengan defaults jika config kosong
        if not loaded_config or len(loaded_config) < 3:
            default_config = self.get_default_config()
            loaded_config.update({k: v for k, v in default_config.items() if k not in loaded_config})
        
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate config dengan DownloaderConfigExtractor validation."""
        return DownloaderConfigExtractor.validate_extracted_config(config)