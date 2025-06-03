"""
File: smartcash/ui/training/training_init.py
Deskripsi: Enhanced training initializer dengan YAML config integration dan simplified architecture
"""

from typing import Dict, Any, List
from IPython.display import display
from smartcash.ui.utils.common_initializer import CommonInitializer
from smartcash.ui.utils.logging_utils import suppress_backend_logs
from smartcash.ui.utils.ui_logger_namespace import TRAINING_LOGGER_NAMESPACE, KNOWN_NAMESPACES
from smartcash.common.config.manager import get_config_manager

MODULE_LOGGER_NAME = KNOWN_NAMESPACES[TRAINING_LOGGER_NAMESPACE]

class TrainingInitializer(CommonInitializer):
    """Enhanced training initializer dengan YAML config integration"""
    
    def __init__(self):
        super().__init__(MODULE_LOGGER_NAME, TRAINING_LOGGER_NAMESPACE)
        self.config_update_callbacks = []
        self._ui_displayed = False
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create UI components dengan YAML config integration"""
        try:
            from smartcash.ui.training.components.training_form import create_training_form
            from smartcash.ui.training.components.training_layout import create_training_layout
            
            # Merge config dengan YAML configs
            yaml_config = self._load_yaml_configs()
            merged_config = {**self._get_default_config(), **yaml_config, **config}
            
            # Create form dengan merged config
            form_components = create_training_form(merged_config)
            
            # Create layout
            layout_components = create_training_layout(form_components)
            
            # Setup config callback
            self._setup_yaml_config_callback(layout_components)
            
            return layout_components
            
        except Exception as e:
            self.logger.error(f"❌ Error creating UI components: {str(e)}")
            return self._create_minimal_fallback(str(e))
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers dengan model integration"""
        try:
            from smartcash.ui.training.handlers.training_handlers import setup_all_training_handlers
            
            # Initialize model services dengan YAML config
            ui_components = self._initialize_model_services_yaml(ui_components, config)
            
            # Setup handlers
            ui_components = setup_all_training_handlers(ui_components, config, env)
            
            return ui_components
            
        except Exception as e:
            self.logger.error(f"❌ Error setting up handlers: {str(e)}")
            return ui_components
    
    def _initialize_model_services_yaml(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize model services dengan YAML config parameters"""
        try:
            # Extract YAML config parameters
            model_config = config.get('model', {})
            training_config = config.get('training', {})
            
            model_type = model_config.get('type', 'efficient_basic')
            backbone = model_config.get('backbone', 'efficientnet_b4')
            layer_mode = config.get('training_utils', {}).get('layer_mode', 'single')
            
            self.logger.info(f"🔧 Initializing model services dengan YAML config:")
            self.logger.info(f"   • Model: {model_type}")
            self.logger.info(f"   • Backbone: {backbone}")
            self.logger.info(f"   • Layer Mode: {layer_mode}")
            
            # Create model manager dengan YAML params
            model_manager = self._create_yaml_model_manager(config)
            
            # Create training manager
            training_manager = self._create_yaml_training_manager(ui_components, config, model_manager)
            
            # Register services
            training_manager and model_manager and training_manager.register_model_manager(model_manager) and training_manager.register_config(config)
            
            ui_components.update({
                'model_manager': model_manager,
                'training_manager': training_manager,
                'yaml_config': config
            })
            
            self.logger.success(f"✅ Model services initialized dengan {model_type}")
            
        except Exception as e:
            self.logger.error(f"❌ Model services error: {str(e)}")
            
        return ui_components
    
    def _create_yaml_model_manager(self, config: Dict[str, Any]):
        """Create model manager dengan YAML config dan Drive pretrained models"""
        try:
            from smartcash.model.manager import ModelManager
            
            # Extract parameters dari YAML config
            model_config = config.get('model', {})
            training_utils = config.get('training_utils', {})
            
            # Validasi pretrained models dari Drive
            pretrained_models_path = config.get('pretrained_models_path', '/content/drive/MyDrive/SmartCash/models')
            self._validate_pretrained_models(pretrained_models_path)
            
            model_manager = ModelManager(
                config=config,
                model_type=model_config.get('type', 'efficient_basic'),
                layer_mode=training_utils.get('layer_mode', 'single'),
                detection_layers=['banknote'],  # Default untuk single layer
                pretrained_models_path=pretrained_models_path,
                logger=self.logger
            )
            
            return model_manager
            
        except Exception as e:
            self.logger.error(f"❌ YAML model manager error: {str(e)}")
            return None
    
    def _validate_pretrained_models(self, models_path: str) -> None:
        """Validasi pretrained models di Drive sesuai model_constants.py"""
        try:
            from smartcash.ui.pretrained_model.constants.model_constants import MODEL_CONFIGS
            from pathlib import Path
            
            models_dir = Path(models_path)
            if not models_dir.exists():
                self.logger.warning(f"⚠️ Pretrained models directory tidak ditemukan: {models_path}")
                return
            
            # Check model files sesuai model_constants.py
            missing_models = []
            for model_key, model_info in MODEL_CONFIGS.items():
                model_file = models_dir / model_info['filename']
                if not model_file.exists():
                    missing_models.append(model_info['filename'])
            
            if missing_models:
                self.logger.warning(f"⚠️ Missing pretrained models: {', '.join(missing_models)}")
            else:
                self.logger.success(f"✅ All pretrained models tersedia di {models_path}")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Error validating pretrained models: {str(e)}")
    
    def _create_yaml_training_manager(self, ui_components: Dict[str, Any], config: Dict[str, Any], model_manager):
        """Create training manager dengan YAML config"""
        try:
            if not model_manager:
                self.logger.warning("⚠️ Model manager tidak tersedia")
                return None
            
            # Simple training manager yang delegate ke model service
            class YAMLTrainingManager:
                def __init__(self, ui_components, config, model_manager, logger):
                    self.ui_components = ui_components
                    self.config = config
                    self.model_manager = model_manager
                    self.logger = logger
                    self.is_training = False
                
                def register_model_manager(self, manager): self.model_manager = manager
                def register_config(self, config): self.config.update(config)
                
                def start_training(self, training_config):
                    """Start training dengan YAML config"""
                    self.is_training = True
                    self.logger.info(f"🚀 Starting training dengan YAML config...")
                    # TODO: Delegate ke actual training service
                    
                def stop_training(self):
                    """Stop training"""
                    self.is_training = False
                    self.logger.info("🛑 Training stopped")
                
                def reset_metrics(self):
                    """Reset metrics"""
                    self.logger.info("🔄 Metrics reset")
                
                def get_training_status(self):
                    """Get training status"""
                    return {'is_training': self.is_training, 'config': self.config}
            
            return YAMLTrainingManager(ui_components, config, model_manager, self.logger)
            
        except Exception as e:
            self.logger.error(f"❌ YAML training manager error: {str(e)}")
            return None
    
    def _setup_yaml_config_callback(self, ui_components: Dict[str, Any]):
        """Setup callback untuk YAML config updates"""
        def yaml_config_update_callback(new_config: Dict[str, Any]):
            """Handle YAML config updates"""
            try:
                # Merge dengan existing YAML configs
                yaml_configs = self._load_yaml_configs()
                merged_config = {**yaml_configs, **new_config}
                
                # Update form components
                from smartcash.ui.training.components.training_form import update_config_tabs_in_form
                update_config_tabs_in_form(ui_components, merged_config)
                
                self.logger.info("🔄 YAML configuration updated")
                
            except Exception as e:
                self.logger.warning(f"⚠️ YAML config callback error: {str(e)}")
        
        self.config_update_callbacks.append(yaml_config_update_callback)
        ui_components['config_update_callback'] = yaml_config_update_callback
    
    def _load_yaml_configs(self) -> Dict[str, Any]:
        """Load dan merge semua YAML configs dari Google Drive dengan inheritance"""
        try:
            # Set config manager untuk Drive path
            config_manager = get_config_manager(base_dir='/content/drive/MyDrive/SmartCash')
            
            # Load configs sesuai inheritance order dari YAML _base_
            base_config = config_manager.get_config('base_config') or {}
            hyperparams = config_manager.get_config('hyperparameters_config') or {}
            model_config = config_manager.get_config('model_config') or {}
            training_config = config_manager.get_config('training_config') or {}
            
            # Merge dengan inheritance order (sesuai _base_ dalam YAML)
            merged = {**base_config}
            
            # Merge hyperparameters (inherits from base)
            merged.update(hyperparams)
            
            # Merge model config (inherits from base + hyperparams)
            merged.update(model_config)
            
            # Merge training config (inherits from all above)
            merged.update(training_config)
            
            # Add Drive paths untuk pretrained models
            merged['pretrained_models_path'] = '/content/drive/MyDrive/SmartCash/models'
            
            self.logger.debug(f"📄 Loaded YAML configs dari Drive: {len(merged)} parameters")
            return merged
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error loading Drive YAML configs: {str(e)}")
            return {}
    
    def _create_minimal_fallback(self, error_msg: str) -> Dict[str, Any]:
        """Create minimal fallback components"""
        from smartcash.ui.utils.fallback_utils import create_fallback_ui
        
        fallback = create_fallback_ui(f"Training init error: {error_msg}", 'training')
        
        # Add minimal training components
        fallback.update({
            'start_button': __import__('ipywidgets').Button(description="🚀 Training", disabled=True),
            'stop_button': __import__('ipywidgets').Button(description="⏹️ Stop", disabled=True),
            'config_tabs': __import__('ipywidgets').HTML("Config tidak tersedia"),
            'log_output': __import__('ipywidgets').Output(),
            'chart_output': __import__('ipywidgets').Output(),
            'config': {}
        })
        
        return fallback
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default config sesuai YAML structure"""
        return {
            'model': {
                'type': 'efficient_basic',
                'backbone': 'efficientnet_b4',
                'backbone_pretrained': True,
                'confidence': 0.25,
                'iou_threshold': 0.45,
                'use_attention': False,
                'use_residual': False,
                'use_ciou': False
            },
            'training': {},
            'hyperparameters': {},
            'epochs': 100,
            'batch_size': 16,
            'learning_rate': 0.001,
            'weight_decay': 0.0005,
            'optimizer': 'Adam',
            'scheduler': 'cosine',
            'early_stopping': {'enabled': True, 'patience': 15},
            'save_best': {'enabled': True},
            'multi_scale': True,
            'training_utils': {
                'experiment_name': 'efficientnet_b4_training',
                'checkpoint_dir': '/content/runs/train/checkpoints',
                'mixed_precision': True,
                'layer_mode': 'single'
            },
            'validation': {
                'frequency': 1,
                'iou_thres': 0.6,
                'conf_thres': 0.001
            },
            'pretrained_models_path': '/content/drive/MyDrive/SmartCash/models'
        }
    
    def _get_critical_components(self) -> List[str]:
        """Critical components untuk training UI"""
        return ['main_container', 'start_button', 'stop_button', 'config_tabs', 'log_output']
    
    def trigger_config_update(self, new_config: Dict[str, Any]):
        """Trigger YAML config update callbacks"""
        [callback(new_config) for callback in self.config_update_callbacks]

# Global instance
_training_initializer = TrainingInitializer()

def initialize_training_ui(env=None, config=None, **kwargs):
    """Factory function untuk training UI dengan YAML config integration"""
    suppress_backend_logs()
    
    try:
        # Use global instance untuk prevent duplication
        initializer = _training_initializer
        
        # Check duplicate display
        if initializer._ui_displayed:
            return {'message': 'Training UI sudah ditampilkan', 'status': 'duplicate'}
        
        # Initialize dengan YAML integration
        result = initializer.initialize(env, config, **kwargs)
        
        # Auto-display dengan duplicate prevention
        if result and isinstance(result, dict):
            ui_widget = result.get('ui') or result.get('main_container')
            if ui_widget and not getattr(ui_widget, '_displayed', False):
                display(ui_widget)
                ui_widget._displayed = True
                initializer._ui_displayed = True
        
        return result
        
    except Exception as e:
        from smartcash.ui.utils.fallback_utils import create_fallback_ui
        fallback = create_fallback_ui(f"Training init error: {str(e)}", 'training')
        display(fallback.get('ui', fallback))
        return fallback

# One-liner factory untuk compatibility
create_training_ui = lambda env=None, config=None, **kw: initialize_training_ui(env, config, **kw)
get_training_initializer = lambda: _training_initializer
reset_training_display = lambda: setattr(_training_initializer, '_ui_displayed', False)