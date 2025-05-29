"""
File: smartcash/ui/training/training_init.py
Deskripsi: Training initializer dengan direct model integration tanpa bridge service
"""

from typing import Dict, Any, List
from IPython.display import display

from smartcash.ui.utils.common_initializer import CommonInitializer
from smartcash.ui.utils.logging_utils import suppress_all_outputs


class TrainingInitializer(CommonInitializer):
    """Training UI initializer dengan direct model manager integration"""
    
    def __init__(self, module_name: str, logger_namespace: str):
        super().__init__(module_name, logger_namespace)
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create UI components dengan model service integration"""
        from smartcash.ui.training.components.training_form import create_training_form
        from smartcash.ui.training.components.training_layout import create_training_layout
        
        # Create form components dengan config
        form_components = create_training_form(config)
        
        # Create layout arrangement
        layout_components = create_training_layout(form_components)
        
        return layout_components
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers dengan direct model integration"""
        from smartcash.ui.training.handlers.training_handlers import setup_all_training_handlers
        
        # Initialize model services directly
        ui_components = self._initialize_model_services(ui_components, config)
        
        # Setup handlers dengan model integration
        ui_components = setup_all_training_handlers(ui_components, config, env)
        
        return ui_components
    
    def _initialize_model_services(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize model services dengan progress tracking integration"""
        try:
            from smartcash.model.manager import ModelManager
            from smartcash.model.manager_checkpoint import ModelCheckpointManager
            from smartcash.model.services.training_service import ModelTrainingService
            from smartcash.model.services.evaluation.core_evaluation_service import EvaluationService
            
            # Progress tracking untuk initialization
            from smartcash.ui.training.utils.training_progress_utils import update_model_loading_progress
            
            logger = ui_components.get('logger')
            
            # Step 1: Parse config
            update_model_loading_progress(ui_components, 1, 4, "📋 Parsing configuration...")
            training_config = config.get('training', {})
            model_type = training_config.get('model_type', 'efficient_optimized')
            checkpoint_dir = config.get('paths', {}).get('checkpoint_dir', 'runs/train/checkpoints')
            
            model_config = {
                'backbone': training_config.get('backbone', 'efficientnet_b4'),
                'detection_layers': training_config.get('detection_layers', ['banknote', 'nominal']),
                'num_classes': training_config.get('num_classes', 7),
                'img_size': (training_config.get('image_size', 640), training_config.get('image_size', 640)),
                'use_attention': config.get('model_optimization', {}).get('use_attention', True),
                'use_residual': config.get('model_optimization', {}).get('use_residual', True),
                'use_ciou': config.get('model_optimization', {}).get('use_ciou', True)
            }
            
            # Step 2: Create model manager
            update_model_loading_progress(ui_components, 2, 4, f"🧠 Creating {model_type} model manager...")
            model_manager = ModelManager(config=model_config, model_type=model_type, logger=logger)
            
            # Step 3: Create checkpoint manager dengan progress callback
            update_model_loading_progress(ui_components, 3, 4, "💾 Setting up checkpoint manager...")
            
            # Create progress callback untuk checkpoint operations
            def checkpoint_progress_callback(current, total, message):
                from smartcash.ui.training.utils.training_progress_utils import update_checkpoint_progress
                update_checkpoint_progress(ui_components, current, total, message)
            
            checkpoint_manager = ModelCheckpointManager(
                model_manager=model_manager,
                checkpoint_dir=checkpoint_dir,
                logger=logger,
                progress_callback=checkpoint_progress_callback
            )
            
            # Set checkpoint manager ke model manager untuk integration
            model_manager.checkpoint_manager = checkpoint_manager
            
            # Step 4: Create services
            update_model_loading_progress(ui_components, 4, 4, "🚀 Initializing training services...")
            training_service = ModelTrainingService(model_manager, config)
            evaluation_service = EvaluationService(config=config, logger=logger)
            
            # Update UI components
            ui_components.update({
                'model_manager': model_manager,
                'checkpoint_manager': checkpoint_manager,
                'training_service': training_service,
                'evaluation_service': evaluation_service
            })
            
            logger and logger.success(f"✅ Model services initialized: {model_type}")
            
        except Exception as e:
            logger = ui_components.get('logger')
            logger and logger.error(f"❌ Model services initialization error: {str(e)}")
            
            # Fallback services
            ui_components.update({
                'model_manager': None,
                'checkpoint_manager': None,
                'training_service': None,
                'evaluation_service': None
            })
        
        return ui_components
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default training configuration dengan model integration"""
        return {
            'training': {
                'model_type': 'efficient_optimized',  # EfficientNet-B4 dengan FeatureAdapter
                'backbone': 'efficientnet_b4',
                'epochs': 100,
                'batch_size': 16,
                'learning_rate': 0.001,
                'image_size': 640,
                'save_checkpoints': True,
                'use_tensorboard': True,
                'use_mixed_precision': True,
                'optimizer': 'Adam',
                'detection_layers': ['banknote', 'nominal'],
                'num_classes': 7,
                'weight_decay': 0.0005
            },
            'paths': {
                'data_yaml': 'data/currency_dataset.yaml',
                'checkpoint_dir': 'runs/train/checkpoints',
                'tensorboard_dir': 'runs/tensorboard'
            },
            'model_optimization': {
                'use_attention': True,
                'use_residual': True,
                'use_ciou': True
            }
        }
    
    def _get_critical_components(self) -> List[str]:
        """Critical components yang harus ada"""
        return ['main_container', 'start_button', 'stop_button', 'status_panel', 'progress_container', 'log_output']


def initialize_training_ui(env=None, config=None, **kwargs):
    """Factory function untuk training UI dengan direct model integration"""
    suppress_all_outputs()
    
    try:
        # Create initializer
        initializer = TrainingInitializer('training', 'smartcash.ui.training')
        
        # Initialize UI
        result = initializer.initialize(env, config, **kwargs)
        
        # Auto-display
        if hasattr(result, 'children') or hasattr(result, 'layout'):
            display(result)
        elif isinstance(result, dict) and 'ui' in result:
            display(result['ui'])
        
        return result
        
    except Exception as e:
        from smartcash.ui.utils.fallback_utils import create_fallback_ui
        fallback = create_fallback_ui(f"Training init error: {str(e)}", 'training')
        display(fallback.get('ui', fallback))
        return fallback


# One-liner factory untuk compatibility
create_training_ui = lambda env=None, config=None, **kw: initialize_training_ui(env, config, **kw)