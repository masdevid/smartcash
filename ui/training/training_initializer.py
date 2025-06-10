"""
File: smartcash/ui/training/training_initializer.py
Deskripsi: Training UI module dengan progress tracking dan live metrics display
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from IPython.display import display
from smartcash.ui.initializers.blank_initializer import BlankInitializer
from smartcash.ui.training.components.ui_components import create_training_ui
from smartcash.ui.training.handlers.training_handlers import setup_training_handlers
from smartcash.common.config_loader import load_config

class TrainingInitializer(BlankInitializer):
    """Training UI module dengan progress tracking dan live metrics"""
    
    def __init__(self):
        super().__init__(
            module_name='training',
            title='🚀 Training',
            description='Training model dengan progress tracking dan live metrics'
        )
    
    def initialize(self, **kwargs) -> Dict[str, Any]:
        """Initialize training UI components"""
        try:
            # Load config from training_config.yaml
            config = self._load_training_config()
            
            # Create UI components
            ui_components = create_training_ui(config)
            
            # Setup handlers
            self._setup_handlers(ui_components, config)
            
            # Display the UI
            display(ui_components['ui'])
            
            return ui_components
            
        except Exception as e:
            error_msg = f"❌ Gagal menginisialisasi training UI: {str(e)}"
            print(error_msg)
            return {'error': error_msg}
    
    def _load_training_config(self) -> Dict[str, Any]:
        """Load training config with cascade inheritance"""
        try:
            return load_config('training_config.yaml')
        except Exception as e:
            print(f"⚠️ Gagal memuat konfigurasi training: {str(e)}")
            return {}
    
    def _setup_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Setup semua handler yang diperlukan"""
        try:
            setup_training_handlers(ui_components, config)
        except Exception as e:
            print(f"⚠️ Gagal setup handlers: {str(e)}")

# Global instance for easy access
_training_initializer = TrainingInitializer()

def initialize_training_ui(**kwargs) -> Dict[str, Any]:
    """Factory function untuk menginisialisasi training UI"""
    return _training_initializer.initialize(**kwargs)
