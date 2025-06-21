"""
File: smartcash/ui/hyperparameters/handlers/config_handler.py
Deskripsi: Handler untuk konfigurasi hyperparameters dengan inheritance dari BaseConfigHandler
"""

from typing import Dict, Any, Optional
from smartcash.ui.handlers.config_handlers import BaseConfigHandler
from smartcash.common.logger import get_logger
from smartcash.ui.utils.fallback_utils import try_operation_safe

logger = get_logger(__name__)


class HyperparametersConfigHandler(BaseConfigHandler):
    """Handler untuk konfigurasi hyperparameters dengan inheritance dari BaseConfigHandler"""
    
    def __init__(self, module_name: str = 'hyperparameters', 
                 config_filename: str = 'hyperparameters_config.yaml'):
        super().__init__(module_name, config_filename)
        self.config_type = 'hyperparameters'
    
    def get_default_config(self) -> Dict[str, Any]:
        """Ambil default configuration untuk hyperparameters"""
        from .defaults import get_default_hyperparameters_config
        return get_default_hyperparameters_config()
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validasi konfigurasi hyperparameters"""
        return try_operation_safe(
            operation=lambda: self._validate_config_structure(config),
            fallback_value=False,
            logger=logger,
            operation_name="validating hyperparameters config"
        )
    
    def _validate_config_structure(self, config: Dict[str, Any]) -> bool:
        """Internal validation logic"""
        if not isinstance(config, dict):
            logger.warning("Config harus berupa dictionary")
            return False
            
        # Validasi section yang diperlukan
        required_sections = [
            'training', 
            'optimizer', 
            'scheduler', 
            'loss', 
            'early_stopping', 
            'checkpoint'
        ]
        
        for section in required_sections:
            if section not in config:
                logger.warning(f"Section '{section}' tidak ditemukan dalam konfigurasi")
                return False
        
        # Validasi parameter training
        training = config.get('training', {})
        if not isinstance(training, dict):
            logger.warning("Section 'training' harus berupa dictionary")
            return False
            
        # Validasi parameter training
        if not self._validate_training_params(training):
            return False
            
        # Validasi parameter optimizer
        optimizer = config.get('optimizer', {})
        if not isinstance(optimizer, dict):
            logger.warning("Section 'optimizer' harus berupa dictionary")
            return False
            
        if not self._validate_optimizer_params(optimizer):
            return False
        
        return True
    
    def _validate_training_params(self, training: Dict[str, Any]) -> bool:
        """Validasi parameter training"""
        try:
            epochs = training.get('epochs', 100)
            if not isinstance(epochs, int) or epochs < 1:
                logger.error("Invalid epochs value")
                return False
            
            batch_size = training.get('batch_size', 16)
            if not isinstance(batch_size, int) or batch_size < 1:
                logger.error("Invalid batch_size value")
                return False
            
            learning_rate = training.get('learning_rate', 0.01)
            if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
                logger.error("Invalid learning_rate value")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error validating training params: {e}")
            return False
    
    def _validate_optimizer_params(self, optimizer: Dict[str, Any]) -> bool:
        """Validasi parameter optimizer"""
        try:
            name = optimizer.get('name', 'AdamW')
            if name not in ['AdamW', 'SGD', 'Adam']:
                logger.error(f"Unsupported optimizer: {name}")
                return False
            
            weight_decay = optimizer.get('weight_decay', 0.0001)
            if not isinstance(weight_decay, (int, float)) or weight_decay < 0:
                logger.error("Invalid weight_decay value")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error validating optimizer params: {e}")
            return False
    
    def merge_with_model_config(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge hyperparameters config dengan model config"""
        return try_operation_safe(
            operation=lambda: self._merge_configs(model_config),
            fallback_value=self.config,
            logger=logger,
            operation_name="merging hyperparameters with model config"
        )
    
    def _merge_configs(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Internal merge logic"""
        merged_config = self.config.copy()
        
        # Merge training parameters dari model config
        if 'training' in model_config:
            model_training = model_config['training']
            merged_config['training'].update({
                'epochs': model_training.get('epochs', merged_config['training']['epochs']),
                'batch_size': model_training.get('batch_size', merged_config['training']['batch_size']),
                'learning_rate': model_training.get('learning_rate', merged_config['training']['learning_rate'])
            })
        
        # Merge optimizer parameters
        if 'training' in model_config and 'optimizer' in model_config['training']:
            optimizer_name = model_config['training']['optimizer']
            merged_config['optimizer']['name'] = optimizer_name
        
        # Merge scheduler parameters
        if 'training' in model_config and 'scheduler' in model_config['training']:
            scheduler_name = model_config['training']['scheduler']
            merged_config['scheduler']['name'] = scheduler_name
        
        return merged_config
    
    def extract_training_params(self) -> Dict[str, Any]:
        """Extract parameter training untuk backend model"""
        return try_operation_safe(
            operation=self._extract_training_params_internal,
            fallback_value={},
            logger=logger,
            operation_name="extracting training parameters"
        )
    
    def _extract_training_params_internal(self) -> Dict[str, Any]:
        """Internal method untuk extract training parameters"""
        return {
            'epochs': self.config['training']['epochs'],
            'batch_size': self.config['training']['batch_size'],
            'learning_rate': self.config['training']['learning_rate'],
            'image_size': self.config['training'].get('image_size', 640),
            'weight_decay': self.config['optimizer'].get('weight_decay', 0.0005),
            'optimizer': self.config['optimizer'].get('name', 'SGD'),
            'scheduler': self.config['scheduler'].get('name', 'cosine'),
            'early_stopping': self.config['early_stopping'].get('enabled', True),
            'patience': self.config['early_stopping'].get('patience', 15)
        }
    
    def update_from_widgets(self, widgets_dict: Dict[str, Any]) -> bool:
        """Update config dari widget values"""
        return try_operation_safe(
            operation=lambda: self._update_config_from_widgets(widgets_dict),
            fallback_value=False,
            logger=logger,
            operation_name="updating config from widgets"
        )
    
    def _update_config_from_widgets(self, widgets_dict: Dict[str, Any]) -> bool:
        """Internal update logic"""
        if not isinstance(widgets_dict, dict):
            logger.warning("Widgets dictionary tidak valid")
            return False
        
        # Update training parameters
        widget_mappings = [
            ('epochs_slider', 'training', 'epochs'),
            ('batch_size_slider', 'training', 'batch_size'),
            ('learning_rate_slider', 'training', 'learning_rate'),
            ('image_size_slider', 'training', 'image_size'),
            ('optimizer_dropdown', 'optimizer', 'name'),
            ('weight_decay_slider', 'optimizer', 'weight_decay'),
            ('momentum_slider', 'optimizer', 'momentum'),
            ('scheduler_dropdown', 'scheduler', 'name'),
            ('warmup_epochs_slider', 'scheduler', 'warmup_epochs'),
            ('early_stopping_checkbox', 'early_stopping', 'enabled'),
            ('patience_slider', 'early_stopping', 'patience')
        ]
        
        try:
            for widget_key, section, param in widget_mappings:
                if widget_key in widgets_dict and hasattr(widgets_dict[widget_key], 'value'):
                    if section not in self.config:
                        self.config[section] = {}
                    self.config[section][param] = widgets_dict[widget_key].value
            
            return True
            
        except Exception as e:
            logger.error(f"Gagal update config dari widgets: {e}")
            return False