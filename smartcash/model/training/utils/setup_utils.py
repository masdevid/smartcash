#!/usr/bin/env python3
"""
Setup and configuration utilities for the unified training pipeline.

This module handles environment setup, configuration management, and preparation logic.
"""

from pathlib import Path
from typing import Dict, Any
from smartcash.common.logger import get_logger
from smartcash.model.training.platform_presets import get_platform_config, setup_platform_optimizations

logger = get_logger(__name__)


def prepare_training_environment(
    backbone: str, 
    phase_1_epochs: int, 
    phase_2_epochs: int, 
    checkpoint_dir: str, 
    force_cpu: bool = False, 
    **kwargs
) -> Dict[str, Any]:
    """
    Prepare training environment and configuration.
    
    Args:
        backbone: Model backbone name
        phase_1_epochs: Number of epochs for phase 1
        phase_2_epochs: Number of epochs for phase 2
        checkpoint_dir: Directory for checkpoint management
        force_cpu: Force CPU usage instead of auto-detecting GPU/MPS
        **kwargs: Additional configuration overrides
        
    Returns:
        Dictionary containing preparation results and configuration
    """
    try:
        # Step 1: Platform detection and optimization
        logger.info("🔧 Detecting platform and applying optimizations")
        setup_platform_optimizations()
        
        # Step 2: Load platform-optimized configuration
        logger.info("⚙️ Loading platform-optimized configuration")
        config = get_platform_config(backbone, phase_1_epochs, phase_2_epochs)
        
        # Step 3: Apply force_cpu if specified
        if force_cpu:
            config = apply_force_cpu_configuration(config)
        
        # Step 4: Apply custom overrides
        if kwargs:
            config = apply_configuration_overrides(config, **kwargs)
        
        # Step 5: Setup checkpoint directory
        logger.info("📁 Setting up checkpoint management")
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        config['paths']['checkpoints'] = str(checkpoint_path)
        
        return {
            'success': True,
            'config': config,
            'checkpoint_dir': str(checkpoint_path),
            'platform_info': config['platform_info']
        }
        
    except Exception as e:
        logger.error(f"Error in environment preparation: {str(e)}")
        return {'success': False, 'error': f"Preparation failed: {str(e)}"}


def apply_force_cpu_configuration(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply force CPU configuration settings.
    
    Args:
        config: Training configuration dictionary
        
    Returns:
        Updated configuration with CPU settings
    """
    logger.info("🖥️ Force CPU mode enabled - disabling GPU/MPS detection")
    config['device']['device'] = 'cpu'
    config['device']['auto_detect'] = False
    
    # Configure data loader settings for CPU mode to prevent semaphore leaks
    if 'data' not in config:
        config['data'] = {}
    if 'training' not in config:
        config['training'] = {}
    if 'data' not in config['training']:
        config['training']['data'] = {}
        
    # Force single worker on CPU to prevent semaphore leaks
    config['data']['num_workers'] = 0
    config['training']['data']['num_workers'] = 0
    config['data']['pin_memory'] = False
    config['training']['data']['pin_memory'] = False
    
    # Set multiprocessing context to avoid semaphore issues
    import torch.multiprocessing as mp
    if hasattr(mp, 'set_start_method'):
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Method already set
    
    if 'platform_info' in config:
        config['platform_info']['force_cpu'] = True
        
    logger.info("🖥️ CPU mode configured: num_workers=0, pin_memory=False")
    return config


def apply_configuration_overrides(config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Apply custom configuration overrides.
    
    Args:
        config: Base configuration dictionary
        **kwargs: Configuration overrides
        
    Returns:
        Updated configuration with overrides applied
    """
    # Apply specific training parameter overrides
    training_overrides = {}
    
    # Loss type configuration
    if 'loss_type' in kwargs and kwargs['loss_type']:
        training_overrides['loss'] = training_overrides.get('loss', {})
        training_overrides['loss']['type'] = kwargs['loss_type']
    
    # Batch size configuration
    if 'batch_size' in kwargs and kwargs['batch_size'] is not None:
        training_overrides['data'] = training_overrides.get('data', {})
        training_overrides['data']['batch_size'] = kwargs['batch_size']
    
    # Learning rate configurations for training phases
    if any(lr_key in kwargs for lr_key in ['head_lr_p1', 'head_lr_p2', 'backbone_lr']):
        training_overrides['training_phases'] = training_overrides.get('training_phases', {})
        
        if 'head_lr_p1' in kwargs:
            training_overrides['training_phases']['phase_1'] = training_overrides['training_phases'].get('phase_1', {})
            training_overrides['training_phases']['phase_1']['learning_rates'] = training_overrides['training_phases']['phase_1'].get('learning_rates', {})
            training_overrides['training_phases']['phase_1']['learning_rates']['head'] = kwargs['head_lr_p1']
        
        if 'head_lr_p2' in kwargs:
            training_overrides['training_phases']['phase_2'] = training_overrides['training_phases'].get('phase_2', {})
            training_overrides['training_phases']['phase_2']['learning_rates'] = training_overrides['training_phases']['phase_2'].get('learning_rates', {})
            training_overrides['training_phases']['phase_2']['learning_rates']['head'] = kwargs['head_lr_p2']
        
        if 'backbone_lr' in kwargs:
            # Apply backbone learning rate to both phases
            for phase in ['phase_1', 'phase_2']:
                training_overrides['training_phases'][phase] = training_overrides['training_phases'].get(phase, {})
                training_overrides['training_phases'][phase]['learning_rates'] = training_overrides['training_phases'][phase].get('learning_rates', {})
                training_overrides['training_phases'][phase]['learning_rates']['backbone'] = kwargs['backbone_lr']
    
    # Early stopping configuration overrides
    early_stopping_keys = ['early_stopping_enabled', 'early_stopping_patience', 'early_stopping_metric', 
                         'early_stopping_mode', 'early_stopping_min_delta']
    if any(es_key in kwargs for es_key in early_stopping_keys):
        training_overrides['training'] = training_overrides.get('training', {})
        training_overrides['training']['early_stopping'] = training_overrides['training'].get('early_stopping', {})
        
        # Map parameters to config structure
        if 'early_stopping_enabled' in kwargs:
            training_overrides['training']['early_stopping']['enabled'] = kwargs['early_stopping_enabled']
        if 'early_stopping_patience' in kwargs:
            training_overrides['training']['early_stopping']['patience'] = kwargs['early_stopping_patience']
        if 'early_stopping_metric' in kwargs:
            training_overrides['training']['early_stopping']['metric'] = kwargs['early_stopping_metric']
        if 'early_stopping_mode' in kwargs:
            training_overrides['training']['early_stopping']['mode'] = kwargs['early_stopping_mode']
        if 'early_stopping_min_delta' in kwargs:
            training_overrides['training']['early_stopping']['min_delta'] = kwargs['early_stopping_min_delta']
    
    # Apply training overrides to main config
    if training_overrides:
        config = deep_merge_dict(config, training_overrides)
    
    # Apply any other custom configuration overrides
    excluded_keys = ['loss_type', 'batch_size', 'head_lr_p1', 'head_lr_p2', 'backbone_lr'] + early_stopping_keys
    for key, value in kwargs.items():
        if key not in excluded_keys and key in config:
            if isinstance(config[key], dict) and isinstance(value, dict):
                config[key].update(value)
            else:
                config[key] = value
    
    return config


def deep_merge_dict(base_dict: dict, override_dict: dict) -> dict:
    """
    Deep merge two dictionaries, with override_dict taking precedence.
    
    Args:
        base_dict: Base dictionary
        override_dict: Override dictionary
        
    Returns:
        Merged dictionary
    """
    result = base_dict.copy()
    
    for key, value in override_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dict(result[key], value)
        else:
            result[key] = value
    
    return result


def configure_single_phase_settings(config: Dict[str, Any], layer_mode: str) -> Dict[str, Any]:
    """
    Configure settings specific to single-phase training.
    
    Args:
        config: Training configuration
        layer_mode: Layer mode ('single' or 'multi')
        
    Returns:
        Updated configuration for single-phase training
    """
    # Configure loss type based on layer mode (model builder stays in multi mode)
    # Ensure the configuration structure exists
    if 'training' not in config:
        config['training'] = {}
    if 'loss' not in config['training']:
        config['training']['loss'] = {}
    if 'type' not in config['training']['loss']:
        config['training']['loss']['type'] = 'uncertainty_multi_task'
    
    if layer_mode == 'single':
        # Single layer detection - force standard YOLO loss
        original_loss_type = config['training']['loss']['type']
        config['training']['loss']['type'] = 'standard'
        logger.info(f"🎯 Single layer mode: using standard YOLO loss (was {original_loss_type})")
    else:
        # Multi layer detection - use dynamic/uncertainty loss
        if config['training']['loss']['type'] not in ['uncertainty_multi_task', 'weighted_multi_task']:
            original_loss_type = config['training']['loss']['type']
            config['training']['loss']['type'] = 'uncertainty_multi_task'
            logger.info(f"🎯 Multi layer mode: using uncertainty_multi_task loss (was {original_loss_type})")
        else:
            logger.info(f"🎯 Multi layer mode: using {config['training']['loss']['type']} loss")
    
    return config