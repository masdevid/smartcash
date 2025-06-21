"""
File: smartcash/ui/strategy/handlers/config_handler.py
Deskripsi: Config handler untuk strategy dengan cascading inheritance
"""

from typing import Dict, Any, Optional
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.common.logger import get_logger

logger = get_logger(__name__)


class StrategyConfigHandler(ConfigHandler):
    """Config handler untuk strategy dengan cascading inheritance"""
    
    def __init__(self, module_name: str = 'strategy', 
                 config_filename: str = 'strategy_config.yaml',
                 parent_module: Optional[str] = None):
        super().__init__(module_name, config_filename, parent_module)
        self.config_type = 'strategy'
    
    def get_default_config(self) -> Dict[str, Any]:
        """Dapatkan default configuration"""
        return {
            'training': {
                'enabled': True,
                'batch_size': 16,
                'epochs': 100,
                'optimizer': 'AdamW',
                'learning_rate': 0.001,
            },
            'validation': {
                'enabled': True,
                'val_split': 0.2
            }
        }
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validasi konfigurasi strategy"""
        return self._validate_config_structure(config)
    
    def _validate_config_structure(self, config: Dict[str, Any]) -> bool:
        """Validasi struktur konfigurasi"""
        if not isinstance(config, dict):
            logger.warning("Config harus berupa dictionary")
            return False
            
        # Validasi section yang diperlukan
        required_sections = ['training', 'validation']
        for section in required_sections:
            if section not in config:
                logger.warning(f"Section '{section}' tidak ditemukan dalam konfigurasi")
                return False
        
        # Validasi parameter training
        training = config.get('training', {})
        if not isinstance(training, dict):
            logger.warning("Section 'training' harus berupa dictionary")
            return False
            
        # Validasi parameter validation
        validation = config.get('validation', {})
        if not isinstance(validation, dict):
            logger.warning("Section 'validation' harus berupa dictionary")
            return False
            
        return True
    
    def merge_with_model_config(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge strategy config dengan model config"""
        if not isinstance(model_config, dict):
            return self.config
            
        merged_config = self.config.copy()
        
        # Merge training parameters
        if 'training' in model_config:
            training = model_config['training']
            if isinstance(training, dict):
                merged_config['training'].update({
                    'batch_size': training.get('batch_size', merged_config['training']['batch_size']),
                    'epochs': training.get('epochs', merged_config['training']['epochs']),
                    'learning_rate': training.get('learning_rate', merged_config['training']['learning_rate'])
                })
        
        return merged_config
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Optimized config extraction dengan efficient mapping"""
        get_val = lambda key, default: getattr(ui_components.get(key), 'value', default) if key in ui_components else default
        
        # Strategy-specific extraction dengan nested structure
        return {
            'validation': {
                'frequency': get_val('val_frequency_slider', 1),
                'iou_thres': get_val('iou_thres_slider', 0.6),
                'conf_thres': get_val('conf_thres_slider', 0.001),
                'max_detections': get_val('max_detections_slider', 300)
            },
            'training_utils': {
                'experiment_name': get_val('experiment_name_text', self._get_dynamic_experiment_name(ui_components)),
                'checkpoint_dir': get_val('checkpoint_dir_text', '/content/runs/train/checkpoints'),
                'tensorboard': get_val('tensorboard_checkbox', True),
                'log_metrics_every': get_val('log_metrics_slider', 10),
                'visualize_batch_every': get_val('visualize_batch_slider', 100),
                'gradient_clipping': get_val('gradient_clipping_slider', 1.0),
                'layer_mode': get_val('layer_mode_dropdown', 'single')
            },
            'multi_scale': {
                'enabled': get_val('multi_scale_checkbox', True),
                'img_size_min': get_val('img_size_min_slider', 320),
                'img_size_max': get_val('img_size_max_slider', 640),
                'step_size': 32
            }
        }
    
    def _get_dynamic_experiment_name(self, ui_components: Dict[str, Any]) -> str:
        """Generate dynamic experiment name berdasarkan model_type dan layer_mode"""
        layer_mode = getattr(ui_components.get('layer_mode_dropdown'), 'value', 'single')
        # Default model_type dari cascading config atau fallback
        model_type = getattr(ui_components, 'model_type', 'efficient_optimized')
        return f"{model_type}_{layer_mode}"
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Optimized UI update dengan batch operations"""
        # Extract nested configs dengan safe defaults
        validation = config.get('validation', {})
        utils = config.get('training_utils', {})
        multi_scale = config.get('multi_scale', {})
        
        # Batch update mapping untuk efficiency
        update_mappings = [
            # Validation mappings
            ('val_frequency_slider', validation.get('frequency', 1)),
            ('iou_thres_slider', validation.get('iou_thres', 0.6)),
            ('conf_thres_slider', validation.get('conf_thres', 0.001)),
            ('max_detections_slider', validation.get('max_detections', 300)),
            
            # Utils mappings
            ('experiment_name_text', utils.get('experiment_name', self._get_fallback_experiment_name(config))),
            ('checkpoint_dir_text', utils.get('checkpoint_dir', '/content/runs/train/checkpoints')),
            ('tensorboard_checkbox', utils.get('tensorboard', True)),
            ('log_metrics_slider', utils.get('log_metrics_every', 10)),
            ('visualize_batch_slider', utils.get('visualize_batch_every', 100)),
            ('gradient_clipping_slider', utils.get('gradient_clipping', 1.0)),
            ('layer_mode_dropdown', utils.get('layer_mode', 'single')),
            
            # Multi-scale mappings
            ('multi_scale_checkbox', multi_scale.get('enabled', True)),
            ('img_size_min_slider', multi_scale.get('img_size_min', 320)),
            ('img_size_max_slider', multi_scale.get('img_size_max', 640))
        ]
        
        # One-liner batch update
        [setattr(ui_components[key], 'value', value) 
         for key, value in update_mappings 
         if key in ui_components and hasattr(ui_components[key], 'value')]
    
    def _get_fallback_experiment_name(self, config: Dict[str, Any]) -> str:
        """Fallback experiment name dari config"""
        model_type = config.get('model', {}).get('model_type', 'efficient_optimized')
        layer_mode = config.get('training_utils', {}).get('layer_mode', 'single')
        return f"{model_type}_{layer_mode}"
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default config dengan optimasi caching"""
        if self._default_config is None:
            from .defaults import get_default_strategy_config
            self._default_config = get_default_strategy_config()
        return self._default_config
    
    def after_save_success(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Auto-refresh summary dengan optimized timestamp handling"""
        self._refresh_summary_with_timestamp(ui_components, config, "💾 Disimpan")
        self.logger.success("💾 Strategy config saved successfully")
    
    def after_reset_success(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Auto-refresh summary dengan optimized timestamp handling"""
        self._refresh_summary_with_timestamp(ui_components, config, "🔄 Direset")
        self.logger.success("🔄 Strategy config reset to defaults")
    
    def _refresh_summary_with_timestamp(self, ui_components: Dict[str, Any], config: Dict[str, Any], action: str) -> None:
        """Optimized summary refresh dengan timestamp"""
        if 'summary_card' not in ui_components:
            return
        
        try:
            from smartcash.ui.strategy.components.ui_layout import update_summary_card
            import datetime
            
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            formatted_timestamp = f"{action} {timestamp}"
            
            update_summary_card(ui_components, config, formatted_timestamp)
            
        except ImportError:
            self.logger.warning("⚠️ Could not update summary card")
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced config validation untuk strategy parameters"""
        errors = []
        
        # Validation checks dengan one-liner pattern
        validation_checks = [
            (config.get('validation', {}).get('frequency', 1) < 1, "Validation frequency must be >= 1"),
            (not (0.1 <= config.get('validation', {}).get('iou_thres', 0.6) <= 0.9), "IoU threshold must be between 0.1-0.9"),
            (config.get('multi_scale', {}).get('img_size_min', 320) >= config.get('multi_scale', {}).get('img_size_max', 640), "Min image size must be < max image size"),
            (not config.get('training_utils', {}).get('experiment_name', '').strip(), "Experiment name cannot be empty")
        ]
        
        errors.extend([msg for condition, msg in validation_checks if condition])
        
        return {'valid': not errors, 'errors': errors}
    
    def get_config_summary(self, config: Dict[str, Any]) -> str:
        """Optimized config summary untuk display"""
        validation_count = len(config.get('validation', {}))
        utils_count = len(config.get('training_utils', {}))
        multi_scale_enabled = config.get('multi_scale', {}).get('enabled', False)
        
        return f"📊 Strategy: {validation_count} validation, {utils_count} utils, multi-scale: {'✅' if multi_scale_enabled else '❌'}"