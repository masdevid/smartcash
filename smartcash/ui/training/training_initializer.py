"""smartcash/ui/training/training_initializer.py

Initializer untuk modul UI Training yang mewarisi ConfigCellInitializer.
"""

from typing import Dict, Any
from smartcash.ui.initializers.config_cell_initializer import ConfigCellInitializer
from smartcash.common.config.manager import get_config_manager
from smartcash.ui.strategy.handlers.config_handler import StrategyConfigHandler

class TrainingInitializer(ConfigCellInitializer):
    """Initializer khusus untuk UI Training"""
    
    def __init__(self, config_manager=None, env_manager=None):
        super().__init__(
            module_name='training',
            config_filename='training_config.yaml',
            config_handler_class=None,
            parent_module=None
        )
        self.config_manager = config_manager or get_config_manager()
        self.env_manager = env_manager
    
    def _load_cascading_config(self) -> Dict[str, Any]:
        """Load config dengan cascading inheritance"""
        # Urutan inheritance: base -> preprocessing -> augmentation -> model -> backbone -> hyperparameters -> training
        inheritance_chain = [
            'base_config',
            'preprocessing_config',
            'augmentation_config',
            'model_config',
            'backbone_config',
            'hyperparameters_config',
            'training_config'
        ]
        
        # Merge configs dalam urutan inheritance
        merged_config = {}
        for config_name in inheritance_chain:
            try:
                config = self.config_manager.get_config(config_name)
                merged_config.update(config)
            except Exception as e:
                print(f"⚠️ Gagal load config {config_name}: {str(e)}")
        
        return merged_config
    
    def setup_ui(self):
        """Menyiapkan komponen UI training"""
        from smartcash.ui.training.components import training_component
        self.config = self._load_cascading_config()
        self.ui_components = training_component.create_training_ui(self.config)
        return self.ui_components
    
    def setup_handlers(self, ui_components):
        """Menyiapkan handler untuk UI training"""
        from smartcash.ui.training.handlers import training_handlers
        training_handlers.setup_handlers(
            ui_components,
            self.config_manager,
            self.env_manager
        )


def initialize_training_config(env=None, config=None, parent_callbacks=None, **kwargs):
    """
    Factory function untuk training config cell dengan cascading inheritance
    
    Args:
        env: Environment manager instance
        config: Override config values (akan di-merge dengan cascading config)
        parent_callbacks: Callbacks untuk parent modules
        **kwargs: Additional arguments
        
    Returns:
        UI components dengan config yang sudah di-cascade
    """
    # Jika config tidak disediakan atau minimal, biarkan initializer load cascading config
    if not config:
        config = None  # Let initializer handle cascading loading
    
    return create_config_cell(
        TrainingInitializer, 
        'training', 
        'training_config', 
        env=env, 
        config=config, 
        config_handler_class=StrategyConfigHandler,
        parent_callbacks=parent_callbacks,
        **kwargs
    )
