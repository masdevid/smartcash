"""
File: smartcash/ui/strategy/handlers/config_handler.py
Deskripsi: Config handler untuk strategy khusus (bukan hyperparameters)
"""

from typing import Dict, Any
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.ui.strategy.handlers.defaults import get_default_strategy_config
from smartcash.ui.utils.fallback_utils import show_status_safe


class StrategyConfigHandler(ConfigHandler):
    """Config handler untuk strategy dengan cascading inheritance dan fokus non-hyperparameters"""
    
    def load_config(self, config_name: str = None, use_base_config: bool = True) -> Dict[str, Any]:
        """Load config dengan cascading inheritance"""
        return self._load_cascading_inheritance()
    
    def _load_cascading_inheritance(self) -> Dict[str, Any]:
        """Load config dengan cascading inheritance sesuai urutan"""
        inheritance_chain = [
            'base_config',
            'preprocessing_config', 
            'augmentation_config',
            'model_config',
            'backbone_config',
            'hyperparameters_config',
            'training_config'
        ]
        
        merged_config = {}
        for config_name in inheritance_chain:
            try:
                config = self.config_manager.get_config(config_name)
                if config:
                    config.pop('_base_', None)
                    merged_config = self._deep_merge_dicts(merged_config, config)
                    self.logger.debug(f"🔗 Loaded {config_name}")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load {config_name}: {str(e)}")
        
        return merged_config if merged_config else self.get_default_config()
    
    def _deep_merge_dicts(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge dictionaries"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_dicts(result[key], value)
            else:
                result[key] = value
        return result
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config khusus strategy (bukan hyperparameters)"""
        get_val = lambda key, default: getattr(ui_components.get(key), 'value', default) if key in ui_components else default
        
        return {
            # Validation strategy
            'validation': {
                'frequency': get_val('val_frequency_slider', 1),
                'iou_thres': get_val('iou_thres_slider', 0.6),
                'conf_thres': get_val('conf_thres_slider', 0.001),
                'max_detections': get_val('max_detections_slider', 300)
            },
            
            # Training utilities
            'training_utils': {
                'experiment_name': get_val('experiment_name_text', 'efficientnet_b4_training'),
                'checkpoint_dir': get_val('checkpoint_dir_text', '/content/runs/train/checkpoints'),
                'tensorboard': get_val('tensorboard_checkbox', True),
                'log_metrics_every': get_val('log_metrics_slider', 10),
                'visualize_batch_every': get_val('visualize_batch_slider', 100),
                'gradient_clipping': get_val('gradient_clipping_slider', 1.0),
                'layer_mode': get_val('layer_mode_dropdown', 'single')
            },
            
            # Multi-scale training
            'multi_scale': {
                'enabled': get_val('multi_scale_checkbox', True),
                'img_size_min': get_val('img_size_min_slider', 320),
                'img_size_max': get_val('img_size_max_slider', 640),
                'step_size': 32
            }
        }
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dari config strategy"""
        set_val = lambda key, value: setattr(ui_components[key], 'value', value) if key in ui_components and hasattr(ui_components[key], 'value') else None
        
        validation = config.get('validation', {})
        utils = config.get('training_utils', {})
        multi_scale = config.get('multi_scale', {})
        
        # Update mapping dengan one-liner
        field_mappings = [
            # Validation
            ('val_frequency_slider', validation.get('frequency', 1)),
            ('iou_thres_slider', validation.get('iou_thres', 0.6)),
            ('conf_thres_slider', validation.get('conf_thres', 0.001)),
            ('max_detections_slider', validation.get('max_detections', 300)),
            
            # Utils
            ('experiment_name_text', utils.get('experiment_name', 'efficientnet_b4_training')),
            ('checkpoint_dir_text', utils.get('checkpoint_dir', '/content/runs/train/checkpoints')),
            ('tensorboard_checkbox', utils.get('tensorboard', True)),
            ('log_metrics_slider', utils.get('log_metrics_every', 10)),
            ('visualize_batch_slider', utils.get('visualize_batch_every', 100)),
            ('gradient_clipping_slider', utils.get('gradient_clipping', 1.0)),
            ('layer_mode_dropdown', utils.get('layer_mode', 'single')),
            
            # Multi-scale
            ('multi_scale_checkbox', multi_scale.get('enabled', True)),
            ('img_size_min_slider', multi_scale.get('img_size_min', 320)),
            ('img_size_max_slider', multi_scale.get('img_size_max', 640))
        ]
        
        [set_val(component_key, value) for component_key, value in field_mappings]
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default strategy configuration"""
        return get_default_strategy_config()
    
    def after_save_success(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Status informatif setelah save berhasil"""
        show_status_safe("✅ Konfigurasi strategi training berhasil disimpan", "success", ui_components)
        self.logger.success("💾 Strategy config saved successfully")
    
    def after_reset_success(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Status informatif setelah reset berhasil"""
        show_status_safe("🔄 Konfigurasi strategi training berhasil direset", "success", ui_components)
        self.logger.success("🔄 Strategy config reset to defaults")