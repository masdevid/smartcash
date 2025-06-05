"""
File: smartcash/ui/training/handlers/config_handler.py
Deskripsi: Config handler untuk training dengan YAML integration dan one-liner style code
"""

from typing import Dict, Any
import ipywidgets as widgets
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.ui.training.handlers.defaults import get_default_training_config

class TrainingConfigHandler(ConfigHandler):
    """Config handler untuk training dengan YAML integration dan one-liner style code"""
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dari UI components dengan one-liner style"""
        # Menggunakan fungsi extract_config_from_ui dari defaults.py
        from smartcash.ui.training.handlers.defaults import extract_config_from_ui
        
        # Definisikan extraction map yang sesuai dengan struktur UI
        extraction_map = {
            'model': ['model_type', 'backbone_selector', 'confidence_threshold', 'iou_threshold'],
            'training': ['batch_size', 'epochs', 'learning_rate', 'optimizer']
        }
        
        # Extract config dengan one-liner
        return extract_config_from_ui(ui_components, extraction_map)
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dari config dengan one-liner style"""
        # Menggunakan fungsi update_ui_from_config dari defaults.py
        from smartcash.ui.training.handlers.defaults import update_ui_from_config
        
        # Definisikan ui_map yang sesuai dengan struktur UI
        ui_map = {
            'model_type': ('type', None, 'model'),
            'backbone_selector': ('backbone', None, 'model'),
            'confidence_threshold': ('confidence', None, 'model'),
            'iou_threshold': ('iou_threshold', None, 'model'),
            'batch_size': ('batch_size', None, 'training'),
            'epochs': ('epochs', None, 'training'),
            'learning_rate': ('learning_rate', None, 'training'),
            'optimizer': ('optimizer', None, 'training')
        }
        
        # Update UI dengan one-liner
        update_ui_from_config(ui_components, config, ui_map)
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default config untuk training dengan struktur yang lebih jelas"""
        # Menggunakan fungsi get_default_training_config dari defaults.py
        from smartcash.ui.training.handlers.defaults import get_default_training_config
        return get_default_training_config()
