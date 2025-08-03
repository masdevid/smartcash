#!/usr/bin/env python3
"""
Setup and configuration utilities for the unified training pipeline.

This module handles environment setup, configuration management, and preparation logic.
"""

import torch
from pathlib import Path
from typing import Dict, Any
from smartcash.common.logger import get_logger
from smartcash.model.training.platform_presets import get_platform_config, setup_platform_optimizations, setup_platform_optimizations_with_device

logger = get_logger(__name__)


def prepare_training_environment(
    backbone: str, 
    pretrained: bool,
    phase_1_epochs: int, 
    phase_2_epochs: int, 
    checkpoint_dir: str, 
    force_cpu: bool = False,
    training_mode: str = 'two_phase',
    **kwargs
) -> Dict[str, Any]:
    """
    Prepare training environment and configuration.
    
    Args:
        backbone: Model backbone name
        pretrained: Use pretrained weights for backbone
        phase_1_epochs: Number of epochs for phase 1
        phase_2_epochs: Number of epochs for phase 2
        checkpoint_dir: Directory for checkpoint management
        force_cpu: Force CPU usage instead of auto-detecting GPU/MPS
        training_mode: Training mode ('single_phase' or 'two_phase')
        **kwargs: Additional configuration overrides
        
    Returns:
        Dictionary containing preparation results and configuration
    """
    try:
        # Step 1: Load platform-optimized configuration
        logger.debug("⚙️ Loading platform-optimized configuration")
        config = get_platform_config(backbone, phase_1_epochs, phase_2_epochs)
        
        # Step 2: Set pretrained flag and training mode in configuration
        if 'model' in config:
            config['model']['pretrained'] = pretrained
        
        if 'training' not in config:
            config['training'] = {}
        config['training']['training_mode'] = training_mode
        logger.info(f"🏗️ Training setup: {backbone} (pretrained={pretrained}, mode={training_mode})")
        
        # Step 3: Apply force_cpu if specified
        if force_cpu:
            config = apply_force_cpu_configuration(config)
        
        # Step 4: Platform detection and optimization (with device awareness)
        logger.debug("🔧 Detecting platform and applying optimizations")
        target_device = None
        if force_cpu:
            target_device = torch.device('cpu')
        setup_platform_optimizations_with_device(target_device)
        
        # Step 4: Apply custom overrides
        if kwargs:
            config = apply_configuration_overrides(config, **kwargs)
        
        # Step 5: Setup checkpoint directory
        logger.debug("📁 Setting up checkpoint management")
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
    logger.debug("🖥️ Force CPU mode enabled - disabling GPU/MPS detection")
    
    # Ensure device configuration structure exists
    if 'device' not in config:
        config['device'] = {}
    
    # Force CPU device configuration
    config['device']['device'] = 'cpu'
    config['device']['mixed_precision'] = False  # Disable mixed precision for CPU
    config['device']['memory_fraction'] = 1.0    # Use full memory for CPU
    config['device']['allow_tf32'] = False       # Not applicable for CPU
    
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
        
    logger.debug("🖥️ CPU mode configured: num_workers=0, pin_memory=False")
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
    
    # Early stopping configuration overrides (including phase-specific)
    early_stopping_keys = ['early_stopping_enabled', 'early_stopping_phase_1_enabled', 'early_stopping_phase_2_enabled',
                         'early_stopping_patience', 'early_stopping_metric', 
                         'early_stopping_mode', 'early_stopping_min_delta']
    if any(es_key in kwargs for es_key in early_stopping_keys) or ('patience' in kwargs and kwargs['patience'] is not None):
        training_overrides['training'] = training_overrides.get('training', {})
        training_overrides['training']['early_stopping'] = training_overrides['training'].get('early_stopping', {})
        
        # Map parameters to config structure
        if 'early_stopping_enabled' in kwargs:
            training_overrides['training']['early_stopping']['enabled'] = kwargs['early_stopping_enabled']
        if 'early_stopping_phase_1_enabled' in kwargs:
            training_overrides['training']['early_stopping']['phase_1_enabled'] = kwargs['early_stopping_phase_1_enabled']
        if 'early_stopping_phase_2_enabled' in kwargs:
            training_overrides['training']['early_stopping']['phase_2_enabled'] = kwargs['early_stopping_phase_2_enabled']
        if 'early_stopping_patience' in kwargs:
            training_overrides['training']['early_stopping']['patience'] = kwargs['early_stopping_patience']
        if 'patience' in kwargs and kwargs['patience'] is not None:
             training_overrides['training']['early_stopping']['patience'] = kwargs['patience']
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


def apply_training_overrides(config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Apply training parameter overrides from command line arguments to configuration.
    
    Args:
        config: Base training configuration
        **kwargs: Command line arguments and overrides
        
    Returns:
        Updated configuration with command line overrides applied
    """
    # Early stopping overrides (especially patience)
    if 'patience' in kwargs and kwargs['patience'] is not None:
        if 'training' not in config:
            config['training'] = {}
        if 'early_stopping' not in config['training']:
            config['training']['early_stopping'] = {}
        config['training']['early_stopping']['patience'] = kwargs['patience']
        logger.debug(f"🎯 Early stopping patience override: {kwargs['patience']}")
    
    # Phase-specific epoch overrides
    if 'phase1_epochs' in kwargs and kwargs['phase1_epochs'] is not None:
        if 'training_phases' not in config:
            config['training_phases'] = {}
        if 'phase_1' not in config['training_phases']:
            config['training_phases']['phase_1'] = {}
        config['training_phases']['phase_1']['epochs'] = kwargs['phase1_epochs']
        logger.debug(f"🎯 Phase 1 epochs override: {kwargs['phase1_epochs']}")
    
    if 'phase2_epochs' in kwargs and kwargs['phase2_epochs'] is not None:
        if 'training_phases' not in config:
            config['training_phases'] = {}
        if 'phase_2' not in config['training_phases']:
            config['training_phases']['phase_2'] = {}
        config['training_phases']['phase_2']['epochs'] = kwargs['phase2_epochs']
        logger.debug(f"🎯 Phase 2 epochs override: {kwargs['phase2_epochs']}")
    
    # Training mode override
    if 'training_mode' in kwargs and kwargs['training_mode']:
        if 'training' not in config:
            config['training'] = {}
        config['training']['training_mode'] = kwargs['training_mode']
        logger.debug(f"🎯 Training mode override: {kwargs['training_mode']}")
    
    # Backbone override
    if 'backbone' in kwargs and kwargs['backbone']:
        if 'model' not in config:
            config['model'] = {}
        config['model']['backbone'] = kwargs['backbone']
        logger.debug(f"🎯 Backbone override: {kwargs['backbone']}")
    
    # Pretrained override
    if 'pretrained' in kwargs and kwargs['pretrained'] is not None:
        if 'model' not in config:
            config['model'] = {}
        config['model']['pretrained'] = kwargs['pretrained']
        logger.debug(f"🎯 Pretrained override: {kwargs['pretrained']}")
    
    # Verbose override
    if 'verbose' in kwargs and kwargs['verbose'] is not None:
        if 'training' not in config:
            config['training'] = {}
        config['training']['verbose'] = kwargs['verbose']
        logger.debug(f"🎯 Verbose override: {kwargs['verbose']}")
    
    # Force CPU override
    if 'force_cpu' in kwargs and kwargs['force_cpu']:
        config = apply_force_cpu_configuration(config)
        logger.debug("🎯 Force CPU override applied")
    
    # Single layer mode override
    if 'single_layer_mode' in kwargs and kwargs['single_layer_mode']:
        if 'model' not in config:
            config['model'] = {}
        config['model']['layer_mode'] = kwargs['single_layer_mode']
        logger.debug(f"🎯 Single layer mode override: {kwargs['single_layer_mode']}")
    
    # Validation metrics configuration override
    if 'validation_metrics_config' in kwargs and kwargs['validation_metrics_config']:
        validation_config = kwargs['validation_metrics_config']
        
        # Add to training.validation section
        if 'training' not in config:
            config['training'] = {}
        if 'validation' not in config['training']:
            config['training']['validation'] = {}
        
        # Apply validation metrics settings (always hierarchical)
        config['training']['validation']['use_hierarchical_validation'] = True
        
        logger.info(f"🎯 Validation metrics: Using hierarchical validation (YOLOv5 + per-layer)")
    
    # Resume training configuration overrides
    if 'resume_checkpoint' in kwargs and kwargs['resume_checkpoint']:
        config['resume'] = True
        config['resume_checkpoint'] = kwargs['resume_checkpoint']
        logger.debug(f"🎯 Resume checkpoint override: {kwargs['resume_checkpoint']}")
    
    if 'resume_optimizer_state' in kwargs and kwargs['resume_optimizer_state']:
        config['resume_optimizer_state'] = kwargs['resume_optimizer_state']
        logger.debug(f"🎯 Resume optimizer state override: {kwargs['resume_optimizer_state']}")
    
    if 'resume_scheduler_state' in kwargs and kwargs['resume_scheduler_state']:
        config['resume_scheduler_state'] = kwargs['resume_scheduler_state']
        logger.debug(f"🎯 Resume scheduler state override: {kwargs['resume_scheduler_state']}")
    
    if 'resume_epoch' in kwargs and kwargs['resume_epoch'] is not None:
        config['resume_epoch'] = kwargs['resume_epoch']
        logger.debug(f"🎯 Resume epoch override: {kwargs['resume_epoch']}")
    
    return config


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
        logger.debug(f"🎯 Single layer mode: using standard YOLO loss (was {original_loss_type})")
    else:
        # Multi layer detection - use dynamic/uncertainty loss
        if config['training']['loss']['type'] not in ['uncertainty_multi_task', 'weighted_multi_task']:
            original_loss_type = config['training']['loss']['type']
            config['training']['loss']['type'] = 'uncertainty_multi_task'
            logger.debug(f"🎯 Multi layer mode: using uncertainty_multi_task loss (was {original_loss_type})")
        else:
            logger.debug(f"🎯 Multi layer mode: using {config['training']['loss']['type']} loss")
    
    return config