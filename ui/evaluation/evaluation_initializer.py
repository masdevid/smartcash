"""
File: smartcash/ui/evaluation/evaluation_initializer.py
Deskripsi: Initializer untuk model evaluation dengan checkpoint selection dan testing pada data raw
"""

from typing import Dict, Any, List, Optional, Type

from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.evaluation.components.evaluation_form import create_evaluation_form
from smartcash.ui.evaluation.components.evaluation_layout import create_evaluation_layout
from smartcash.ui.evaluation.handlers.checkpoint_handler import setup_checkpoint_handlers
from smartcash.ui.evaluation.handlers.evaluation_handler import setup_evaluation_handlers
from smartcash.ui.evaluation.handlers.metrics_handler import setup_metrics_handlers
from smartcash.ui.evaluation.handlers.config_handler import EvaluationConfigHandler

class EvaluationInitializer(CommonInitializer):
    """Initializer untuk model evaluation dengan Common Initializer pattern"""
    
    def __init__(self):
        super().__init__(
            module_name='evaluation',
            config_handler_class=EvaluationConfigHandler,
        )
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Buat komponen UI untuk evaluation dengan one-liner style"""
        form_components = create_evaluation_form(config)
        layout_components = create_evaluation_layout(form_components, config)
        
        # Merge semua components dengan one-liner
        return {**form_components, **layout_components, 
                'config': config, 'env': env}
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers untuk checkpoint selection, evaluation, dan metrics"""
        from smartcash.ui.evaluation.handlers.scenario_handler import setup_scenario_handlers
        ui_components = setup_evaluation_handlers(ui_components, config, env)
        ui_components = setup_checkpoint_handlers(ui_components, config, env)
        ui_components = setup_scenario_handlers(ui_components, config, env)
        ui_components = setup_metrics_handlers(ui_components, config, env)
        return ui_components
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default config untuk evaluation"""
        from smartcash.ui.evaluation.handlers.defaults import get_default_evaluation_config
        return get_default_evaluation_config()
    
    def _get_critical_components(self) -> List[str]:
        """Komponen critical untuk evaluation"""
        return ['main_container', 'checkpoint_selector', 'evaluate_button', 
                'metrics_table', 'log_output', 'progress_container']
     
__evaluation_initializer = EvaluationInitializer()

def initialize_evaluation_ui(env=None, config=None, **kwargs) -> Dict[str, Any]:
    """Factory function untuk initialize evaluation UI dengan auto-display dan parent module support"""
    return __evaluation_initializer.initialize(env=env, config=config, **kwargs)