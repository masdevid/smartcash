"""
File: smartcash/ui/dataset/split/handlers/config_handler.py
Deskripsi: Configuration handler untuk dataset split dengan centralized error handling
"""

from typing import Dict, Any, Optional, Tuple
from smartcash.ui.core.handlers.config_handler import ConfigHandler
from smartcash.ui.core.errors.handlers import handle_ui_errors
from .base_split_handler import BaseSplitHandler
from .config_extractor import extract_split_config
from .config_updater import update_split_ui, reset_ui_to_defaults
from .defaults import get_default_split_config

class SplitConfigHandler(ConfigHandler, BaseSplitHandler):
    """Handler untuk dataset split configuration dengan centralized error handling.
    
    Handler ini mengintegrasikan dengan parent component untuk:
    - Logging terpusat melalui BaseHandler
    - Error handling konsisten
    - Config management melalui ConfigHandler
    """
    
    def __init__(self, ui_components: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize split config handler.
        
        Args:
            ui_components: Optional UI components dictionary
            config: Optional initial configuration
            **kwargs: Additional arguments passed to parent classes
        """
        # Initialize ConfigHandler with module name
        ConfigHandler.__init__(
            self, 
            module_name="split", 
            persistence_enabled=True,
            **kwargs
        )
        
        # Initialize BaseSplitHandler with UI components
        BaseSplitHandler.__init__(
            self,
            ui_components=ui_components,
            **kwargs
        )
        
        # Set initial config if provided
        if config:
            self.set_config(config)
    
    @handle_ui_errors(error_component_title="Extract Config Error", log_error=True)
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract configuration dari UI components.
        
        Args:
            ui_components: Dictionary of UI components
            
        Returns:
            Dictionary containing extracted configuration
        """
        config = extract_split_config(ui_components)
        self.logger.info("✅ Berhasil extract konfigurasi split")
        return config
    
    @handle_ui_errors(error_component_title="Update UI Error", log_error=True)
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI components dengan konfigurasi.
        
        Args:
            ui_components: Dictionary of UI components
            config: Configuration to apply
        """
        update_split_ui(ui_components, config)
        self.logger.info("✅ UI berhasil di-update dengan konfigurasi")
    
    @handle_ui_errors(error_component_title="Reset UI Error", log_error=True)
    def reset_ui(self, ui_components: Dict[str, Any]) -> None:
        """Reset UI components ke nilai default.
        
        Args:
            ui_components: Dictionary of UI components to reset
        """
        reset_ui_to_defaults(ui_components)
        self.logger.info("🔄 UI berhasil di-reset ke nilai default")
    
    @handle_ui_errors(error_component_title="Validate Config Error", log_error=True)
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate konfigurasi split.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            ratios = config.get('data', {}).get('split_ratios', {})
            train = ratios.get('train', 0)
            valid = ratios.get('valid', 0)
            test = ratios.get('test', 0)
            
            # Validasi ratio sum = 1.0
            total = train + valid + test
            if abs(total - 1.0) > 0.001:  # Allow small floating point error
                return False, f"Split ratios harus berjumlah 1.0 (saat ini: {total:.3f})"
            
            # Validasi path configuration
            paths = config.get('dataset', {}).get('paths', {})
            if not paths.get('base_dir'):
                return False, "Base directory tidak boleh kosong"
                
            self.logger.debug(f"✅ Validasi sukses - Train: {train}, Valid: {valid}, Test: {test}")
            return True, ""
            
        except Exception as e:
            error_msg = f"Validation error: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            return False, error_msg
    
    @handle_ui_errors(error_component_title="Save Config Error", log_error=True)
    def save_config(self, ui_components: Dict[str, Any]) -> bool:
        """Save konfigurasi dengan status update.
        
        Args:
            ui_components: UI components untuk extract config
            
        Returns:
            True jika berhasil save
        """
        # Extract current config
        config = self.extract_config(ui_components)
        
        # Validate sebelum save
        is_valid, error_msg = self.validate_config(config)
        if not is_valid:
            raise ValueError(error_msg)
        
        # Save menggunakan parent method
        result = super().save_config(ui_components)
        
        if result:
            self.logger.info("💾 Konfigurasi berhasil disimpan")
            
        return result
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Get default configuration untuk handler ini.
        
        Returns:
            Default configuration dictionary
        """
        return get_default_split_config()
