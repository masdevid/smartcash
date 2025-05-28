"""
File: smartcash/ui/training/training_init.py
Deskripsi: Main initializer untuk training UI menggunakan CommonInitializer dengan simplified handlers
"""

from typing import Dict, Any, List
from smartcash.ui.utils.common_initializer import CommonInitializer
from smartcash.ui.utils.ui_factory_utils import create_processing_ui


class TrainingInitializer(CommonInitializer):
    """Training UI initializer dengan consolidated components dan handlers"""
    
    def __init__(self):
        super().__init__('training', 'smartcash.ui.training')
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create UI components menggunakan consolidated approach"""
        from smartcash.ui.training.components.training_form import create_training_form
        from smartcash.ui.training.components.training_layout import create_training_layout
        
        # Create form components dengan config
        form_components = create_training_form(config)
        
        # Create layout arrangement
        layout_components = create_training_layout(form_components)
        
        return layout_components
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers menggunakan consolidated approach"""
        from smartcash.ui.training.handlers.training_handlers import setup_all_training_handlers
        
        # Setup semua handlers dalam satu function call
        ui_components = setup_all_training_handlers(ui_components, config, env)
        
        return ui_components
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default training configuration"""
        return {
            'training': {
                'backbone': 'efficientnet_b4',
                'epochs': 100,
                'batch_size': 16,
                'learning_rate': 0.001,
                'image_size': 640,
                'save_checkpoints': True,
                'use_tensorboard': True,
                'use_mixed_precision': True,
                'use_ema': False,
                'optimizer': 'Adam'
            },
            'paths': {
                'checkpoint_dir': 'runs/train/weights',
                'tensorboard_dir': 'runs/tensorboard'
            }
        }
    
    def _get_critical_components(self) -> List[str]:
        """Critical components yang harus ada"""
        return ['main_container', 'start_button', 'stop_button', 'status_panel', 'progress_container', 'log_output']


def initialize_training_ui(env=None, config=None, **kwargs):
    """Factory function untuk training UI dengan auto-display"""
    return create_processing_ui(TrainingInitializer, 'training', 'smartcash.ui.training', env, config, **kwargs)