#!/usr/bin/env python3
"""
Configuration Builder for Training Pipeline

This module handles configuration setup and validation for the training pipeline,
ensuring consistent and valid configurations across different training modes.
"""

from pathlib import Path
from typing import Dict, Any

from smartcash.common.logger import get_logger

logger = get_logger(__name__)


class ConfigurationBuilder:
    """Builds and validates training configurations."""
    
    def __init__(self, session_id: str):
        """
        Initialize configuration builder.
        
        Args:
            session_id: Training session identifier
        """
        self.session_id = session_id
    
    def build_training_config(self, **kwargs) -> Dict[str, Any]:
        """
        Build comprehensive training configuration.
        
        Args:
            **kwargs: Configuration parameters
            
        Returns:
            Complete training configuration dictionary
        """
        # Base configuration with enhanced debug tracking
        debug_map_value = kwargs.get('debug_map', False)
        
        config = {
            'backbone': kwargs.get('backbone', 'cspdarknet'),
            'pretrained': kwargs.get('pretrained', True),
            'training_mode': kwargs.get('training_mode', 'two_phase'),
            'phase_1_epochs': kwargs.get('phase_1_epochs', 1),
            'phase_2_epochs': kwargs.get('phase_2_epochs', 1),
            'checkpoint_dir': Path(kwargs.get('checkpoint_dir', 'data/checkpoints')),
            'force_cpu': kwargs.get('force_cpu', False),
            'session_id': self.session_id,
            'debug_map': debug_map_value,
            'start_phase': kwargs.get('start_phase', 1)
        }
        
        # Enhanced debug flag tracking
        if debug_map_value:
            logger.info(f"🐛 ConfigurationBuilder: debug_map={debug_map_value} received from kwargs")
            logger.info(f"🐛 ConfigurationBuilder: debug_map stored in config={config.get('debug_map')}")
        
        # Add validation metrics configuration from command line args
        validation_config = kwargs.get('validation_metrics_config', {})
        # Validation metrics always use hierarchical approach (YOLOv5 + per-layer)
        logger.info(f"📊 Validation metrics: Using hierarchical validation (YOLOv5 + per-layer)")
        
        # Model configuration
        config['model'] = self._build_model_config(kwargs.get('model', {}), config)
        
        # Training phases configuration
        config['training_phases'] = self._build_phases_config(config)
        
        # Training configuration
        config['training'] = self._build_training_params(config)
        
        # Paths configuration
        config['paths'] = self._build_paths_config(config)
        
        # Device configuration
        config['device'] = self._build_device_config(config)
        
        # Apply training overrides
        config = self._apply_overrides(config, **kwargs)
        
        # Final debug flag verification
        if debug_map_value:
            logger.info(f"🐛 ConfigurationBuilder: Final config debug_map={config.get('debug_map')}")
        
        logger.info(f"🔧 Training configuration built: {config['backbone']} | yolov5")
        return config
    
    def get_phase_specific_model_config(self, base_config: Dict[str, Any], phase_num: int) -> Dict[str, Any]:
        """
        Get model configuration specific to a training phase.
        
        Args:
            base_config: Base training configuration
            phase_num: Phase number (1 or 2)
            
        Returns:
            Phase-specific model configuration
        """
        model_config = base_config['model'].copy()
        training_mode = base_config.get('training_mode', 'two_phase')
        
        if training_mode == 'two_phase':
            if phase_num == 1:
                # Phase 1: Single-layer, denomination detection only (classes 0-6)
                model_config.update({
                    'layer_mode': 'single',
                    'detection_layers': ['layer_1'],
                    'num_classes': 7,
                    'freeze_backbone': True
                })
                logger.info("🔧 Phase 1 model config: single-layer, 7 classes, frozen backbone")
            elif phase_num == 2:
                # Phase 2: Multi-layer, all detection tasks (classes 0-16)
                model_config.update({
                    'layer_mode': 'multi', 
                    'detection_layers': ['layer_1', 'layer_2', 'layer_3'],
                    'num_classes': 17,
                    'freeze_backbone': False
                })
                logger.info("🔧 Phase 2 model config: multi-layer, 17 classes, unfrozen backbone")
        
        return model_config
    
    def _build_model_config(self, model_config: Dict[str, Any], base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Build model-specific configuration."""
        # CRITICAL: Phase-aware model configuration for single→multi approach
        training_mode = base_config.get('training_mode', 'two_phase')
        
        if training_mode == 'two_phase':
            # For two-phase training: Use default phase-aware configuration
            # Phase-specific configs will be applied during phase setup
            default_layer_mode = 'single'  # Phase 1 default
            default_detection_layers = ['layer_1']  # Phase 1 default 
            default_num_classes = 7  # Phase 1 default (Layer 1 only)
        else:
            # For single-phase training: Use multi-layer configuration
            default_layer_mode = 'multi'
            default_detection_layers = ['layer_1', 'layer_2', 'layer_3']
            default_num_classes = 17  # All layers
            
        return {
            'model_name': model_config.get('model_name', 'smartcash_yolov5_integrated'),
            'backbone': base_config['backbone'],
            'pretrained': base_config['pretrained'],
            'layer_mode': model_config.get('layer_mode', default_layer_mode),
            'detection_layers': model_config.get('detection_layers', default_detection_layers),
            'num_classes': model_config.get('num_classes', default_num_classes),
            'img_size': model_config.get('img_size', 640),
            'feature_optimization': model_config.get('feature_optimization', {'enabled': True})
        }
    
    def _build_phases_config(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Build training phases configuration."""
        # Note: base_config parameter kept for interface consistency
        return {
            'phase_1': {
                'learning_rate': 0.001,
                'learning_rates': {
                    'head': 0.001,  # Default head LR for phase 1
                    'backbone': 0.0001  # Default backbone LR (lower)
                },
                'freeze_backbone': True,
                'layer_mode': 'single',
                'description': 'Frozen backbone training'
            },
            'phase_2': {
                'learning_rate': 0.0001,
                'learning_rates': {
                    'head': 0.0001,  # Default head LR for phase 2
                    'backbone': 0.00001  # Default backbone LR (even lower for fine-tuning)
                },
                'freeze_backbone': False,
                'layer_mode': 'multi',
                'description': 'Fine-tuning with multi-layer'
            }
        }
    
    def _build_training_params(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Build training parameters configuration."""
        return {
            'mixed_precision': False,  # Disable for CPU
            'batch_size': 8,
            'num_workers': 0 if base_config['force_cpu'] else 4,
            'pin_memory': False,
            'training_mode': base_config['training_mode'],
            'early_stopping': {
                'enabled': True,
                'patience': 15,  # Default patience (can be overridden by user args)
                'min_delta': 0.001,
                'monitor': 'val_accuracy'
            },
            'validation': {
                'use_hierarchical_validation': True
            }
        }
    
    def _build_paths_config(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Build paths configuration."""
        return {
            'checkpoints': str(base_config['checkpoint_dir']),
            'visualization': 'data/visualization',
            'logs': 'data/logs'
        }
    
    def _build_device_config(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Build device configuration compatible with platform presets."""
        if base_config.get('force_cpu', False):
            return {
                'device': 'cpu',
                'mixed_precision': False,
                'memory_fraction': 1.0,
                'allow_tf32': False
            }
        else:
            return {
                'device': 'auto',  # Will be auto-detected by platform presets
                'mixed_precision': True,
                'memory_fraction': 0.8,
                'allow_tf32': True
            }
    
    def _apply_overrides(self, config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Apply configuration overrides from kwargs."""
        try:
            from smartcash.model.training.utils.setup_utils import apply_training_overrides
            return apply_training_overrides(config, **kwargs)
        except ImportError:
            logger.warning("Setup utils not available, skipping overrides")
            return config
    
    def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """
        Validate training configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if configuration is valid
        """
        required_keys = ['backbone', 'training_mode', 'model', 'training', 'paths']
        
        for key in required_keys:
            if key not in config:
                logger.error(f"Missing required configuration key: {key}")
                return False
        
        # Validate training mode
        if config['training_mode'] not in ['single_phase', 'two_phase']:
            logger.error(f"Invalid training mode: {config['training_mode']}")
            return False
        
        # Validate epochs
        if config['phase_1_epochs'] <= 0:
            logger.error("Phase 1 epochs must be positive")
            return False
        
        if config['training_mode'] == 'two_phase' and config['phase_2_epochs'] <= 0:
            logger.error("Phase 2 epochs must be positive for two-phase training")
            return False
        
        logger.info("✅ Configuration validation passed")
        return True