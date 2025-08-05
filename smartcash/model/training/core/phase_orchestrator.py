#!/usr/bin/env python3
"""
Phase orchestration for the unified training pipeline.

This module handles phase management, configuration setup, and high-level
training coordination between phases.
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable

from smartcash.common.logger import get_logger
from smartcash.model.training.data_loader_factory import DataLoaderFactory
from smartcash.model.training.utils.metrics_history import create_metrics_recorder
from smartcash.model.training.optimizer_factory import OptimizerFactory
from smartcash.model.training.loss_manager import LossManager
from smartcash.model.training.utils.early_stopping import create_early_stopping, create_phase_specific_early_stopping

logger = get_logger(__name__)


class PhaseOrchestrator:
    """Manages training phase setup, configuration, and high-level coordination."""
    
    def __init__(self, model, model_api, config, progress_tracker):
        """
        Initialize phase orchestrator.
        
        Args:
            model: PyTorch model
            model_api: Model API instance
            config: Training configuration
            progress_tracker: Progress tracking instance
        """
        self.model = model
        self.model_api = model_api
        self.config = config
        self.progress_tracker = progress_tracker
        
        # Training state
        self._is_single_phase = False
    
    def _propagate_phase_to_children(self, module, phase_num: int):
        """
        Propagate current_phase to all child modules that can use it.
        
        Args:
            module: PyTorch module to propagate phase to
            phase_num: Phase number to set
        """
        for name, child in module.named_children():
            # Set current_phase on child modules
            child.current_phase = phase_num
            logger.debug(f"Set current_phase={phase_num} on {name} ({type(child).__name__})")
            
            # Recursively propagate to grandchildren
            self._propagate_phase_to_children(child, phase_num)
    
    def setup_phase(self, phase_num: int, epochs: int, save_best_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Set up training phase with all required components.
        
        Args:
            phase_num: Training phase number (1 or 2)
            epochs: Total number of epochs to train for
            save_best_path: Path to save the best model checkpoint.
            
        Returns:
            Dictionary containing all setup components
        """
        try:
            phase_config = self.config['training_phases'][f'phase_{phase_num}']
            
            # Log memory optimization recommendations if available
            self._log_memory_optimization_info()
            
            # Set up data loaders
            data_factory = DataLoaderFactory(self.config)
            train_loader = data_factory.create_train_loader()
            val_loader = data_factory.create_val_loader()
            
            # Set up loss manager with phase awareness
            loss_manager = LossManager(self.config)
            loss_manager.set_current_phase(phase_num)
            
            # Configure model phase
            self._configure_model_phase(phase_num)
            
            # Set up optimizer and scheduler
            optimizer_factory = OptimizerFactory(self.config)
            base_lr = phase_config.get('learning_rate', 0.001)
            
            # Log learning rate configuration for this phase
            self._log_learning_rate_configuration(phase_num, base_lr, phase_config)
            
            optimizer = optimizer_factory.create_optimizer(self.model, base_lr)
            scheduler = optimizer_factory.create_scheduler(optimizer, total_epochs=epochs)
            
            # Log scheduler configuration for this phase
            self._log_scheduler_configuration(phase_num, scheduler, optimizer_factory)
            
            # Set up mixed precision scaler
            scaler = None
            if self.config['training']['mixed_precision']:
                import torch
                scaler = torch.amp.GradScaler('cuda')
            
            # Set up metrics recorder for JSON-based metrics tracking with structured naming
            # Debug: Log config contents for resume troubleshooting
            logger.debug(f"Phase orchestrator config type: {type(self.config)}")
            if isinstance(self.config, dict):
                config_keys = list(self.config.keys())
                logger.debug(f"Config dict keys: {config_keys}")
                # Show resume-related keys specifically
                resume_related = {k: v for k, v in self.config.items() if 'resume' in k.lower()}
                logger.debug(f"Resume-related config: {resume_related}")
            
            # Handle both dict and object-style config access
            if isinstance(self.config, dict):
                backbone = self.config.get('backbone', 'unknown')
                data_name = self.config.get('data_name', 'data')
                # Check both 'resume' and 'resume_checkpoint' keys for resume mode
                resume_value = self.config.get('resume', False)
                resume_checkpoint_value = self.config.get('resume_checkpoint')
                resume_mode = resume_value or bool(resume_checkpoint_value)
                logger.debug(f"Config dict - resume: {resume_value}, resume_checkpoint: {resume_checkpoint_value}, final resume_mode: {resume_mode}")
            else:
                backbone = getattr(self.config, 'backbone', 'unknown')
                data_name = getattr(self.config, 'data_name', 'data') 
                # Check both 'resume' and 'resume_checkpoint' attributes for resume mode
                resume_value = getattr(self.config, 'resume', False)
                resume_checkpoint_value = getattr(self.config, 'resume_checkpoint', None)
                resume_mode = resume_value or bool(resume_checkpoint_value)
                logger.debug(f"Config object - resume: {resume_value}, resume_checkpoint: {resume_checkpoint_value}, final resume_mode: {resume_mode}")
            
            metrics_recorder = create_metrics_recorder(
                output_dir="logs/training",
                backbone=backbone,
                data_name=data_name, 
                phase_num=phase_num,
                resume_mode=resume_mode
            )
            
            # Set up early stopping
            early_stopping = self.setup_early_stopping(phase_num, save_best_path=save_best_path)
            
            return {
                'train_loader': train_loader,
                'val_loader': val_loader,
                'loss_manager': loss_manager,
                'optimizer': optimizer,
                'scheduler': scheduler,
                'scaler': scaler,
                'metrics_recorder': metrics_recorder,
                'early_stopping': early_stopping,
                'phase_config': phase_config
            }
            
        except Exception as e:
            logger.error(f"Error setting up phase {phase_num}: {str(e)}", exc_info=True)
            raise
    
    def _configure_model_phase(self, phase_num: int):
        """Configure model for specific training phase."""
        # Set current phase on model for layer mode control
        self._set_model_phase(phase_num)
        
        # Configure layer mode based on phase and configuration
        training_mode = self.config.get('training_mode', 'two_phase')
        
        if training_mode == 'single_phase':
            # Single-phase mode: respect the configured layer mode
            single_layer_mode = self.config.get('model', {}).get('layer_mode', 'multi')
            if single_layer_mode == 'single':
                self.model.force_single_layer = True
                logger.info(f"🎯 Single-phase mode: forcing single layer output")
            else:
                self.model.force_single_layer = False
                logger.info(f"🎯 Single-phase mode: using multi-layer output")
            
            # Handle backbone freezing for single-phase mode
            single_freeze_backbone = self.config.get('single_phase_freeze_backbone', False)
            self._configure_backbone_freezing(single_freeze_backbone, phase_num)
            
        elif training_mode == 'two_phase':
            # Two-phase mode: Phase 1 = single layer + frozen backbone, Phase 2 = multi layer + unfrozen backbone
            if phase_num == 1:
                logger.info(f"🎯 Phase {phase_num}: single layer mode (layer_1 only)")
                # Phase 1: Freeze backbone
                self._configure_backbone_freezing(freeze=True, phase_num=phase_num)
            else:
                logger.info(f"🎯 Phase {phase_num}: multi-layer mode (all layers)")
                # Phase 2: Unfreeze backbone (this will be handled by pipeline executor)
                # We don't unfreeze here because the checkpoint loading should happen first
                
            self.model.force_single_layer = False  # Use phase-based logic
    
    def _configure_backbone_freezing(self, freeze: bool, phase_num: int):
        """Configure backbone freezing for the current training phase."""
        try:
            if hasattr(self.model, 'freeze_backbone') and hasattr(self.model, 'unfreeze_backbone'):
                if freeze:
                    self.model.freeze_backbone()
                    logger.info(f"❄️ Backbone frozen for Phase {phase_num}")
                else:
                    self.model.unfreeze_backbone()
                    logger.info(f"🔥 Backbone unfrozen for Phase {phase_num}")
            else:
                # Manual backbone freezing/unfreezing
                backbone_params_modified = 0
                for name, param in self.model.named_parameters():
                    # Identify backbone parameters by common naming patterns
                    if any(backbone_part in name.lower() for backbone_part in ['backbone', 'model.0', 'model.1', 'model.2', 'model.3', 'model.4']):
                        param.requires_grad = not freeze
                        backbone_params_modified += 1
                
                action = "frozen" if freeze else "unfrozen"
                logger.info(f"{'❄️' if freeze else '🔥'} Manually {action} {backbone_params_modified} backbone parameters for Phase {phase_num}")
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to configure backbone freezing: {e}")
    
    def setup_early_stopping(self, phase_num: int, save_best_path: Optional[str] = None):
        """Set up phase-specific early stopping configuration."""
        # Get base early stopping configurationComprehensiveMetricsTracker
        es_config = self.config.get('training', {}).get('early_stopping', {})
        
        # Log the early stopping configuration for debugging
        logger.info(f"🔍 Early stopping config: {es_config}")
        
        # Check if phase-specific early stopping is enabled
        use_phase_specific = es_config.get('phase_specific', True)
        
        if use_phase_specific:
            logger.info(f"🎯 Using phase-specific early stopping for Phase {phase_num}")
            # Create phase-specific early stopping with custom criteria
            early_stopping = create_phase_specific_early_stopping(self.config)
            
            # Set the current phase
            early_stopping.set_phase(phase_num)
            
            logger.info(f"🔍 Phase-specific early stopping initialized:")
            if phase_num == 1:
                logger.info(f"   Phase 1 criteria: Loss plateau + {es_config.get('phase1', {}).get('metric_name', 'val_accuracy')} stability")
            else:
                logger.info(f"   Phase 2 criteria: F1/mAP improvement + overfitting detection")
            
            return early_stopping
        else:
            # Use legacy early stopping logic for backward compatibility
            phase_1_enabled = es_config.get('phase_1_enabled', True)  # Default disabled for Phase 1
            phase_2_enabled = es_config.get('phase_2_enabled', True)   # Default enabled for Phase 2
            
            # For two-phase mode, apply phase-specific early stopping logic
            if self.config.get('training_mode', 'two_phase') == 'two_phase':
                if phase_num == 1:
                    if phase_1_enabled:
                        logger.info(f"✅ Legacy early stopping enabled for Phase {phase_num}")
                    else:
                        logger.info(f"🚫 Early stopping disabled for Phase {phase_num}")
                        return None  # Return None to disable early stopping
                else:  # phase_num == 2
                    if phase_2_enabled:
                        logger.info(f"✅ Legacy early stopping enabled for Phase {phase_num}")
                    else:
                        logger.info(f"🚫 Early stopping disabled for Phase {phase_num}")
                        return None  # Return None to disable early stopping
            else:
                # For single-phase mode, use the general early stopping setting
                if es_config.get('enabled', True):
                    logger.info(f"✅ Legacy early stopping enabled for Phase {phase_num}")
                else:
                    logger.info(f"🚫 Early stopping disabled for Phase {phase_num}")
                    return None  # Return None to disable early stopping
            
            # Create standard early stopping instance with configuration
            early_stopping = create_early_stopping(self.config, save_best_path=save_best_path)
            
            logger.info(f"🔍 Legacy early stopping object type: {type(early_stopping).__name__} | Config enabled: {es_config.get('enabled', True)} | Patience: {es_config.get('patience', 'N/A')}")
            
            return early_stopping
    
    def _set_model_phase(self, phase_num: int):
        """Set current phase on model for layer mode control, handling nested YOLOv5 structure.
        
        Args:
            phase_num: The phase number to set (1 or 2)
            
        Logs:
            Only shows success message when all phase settings are successful.
            Individual phase settings are logged at debug level.
        """
        phase_set = False
        
        # Force create and set current_phase on the main model
        self.model.current_phase = phase_num
        phase_set = True
        logger.debug(f"Set model.current_phase to {phase_num}")
        
        # Propagate current_phase to all child modules (especially the head)
        self._propagate_phase_to_children(self.model, phase_num)
        
        # Try to set phase on YOLOv5 model if it exists
        if hasattr(self.model, 'yolov5_model'):
            try:
                yolo_model = self.model.yolov5_model
                yolo_model.current_phase = phase_num
                phase_set = True
                logger.debug(f"Set yolov5_model.current_phase to {phase_num}")
            except Exception as e:
                logger.debug(f"Could not set phase on yolov5_model: {e}")
                phase_set = False
        
        # Try to set phase on nested model if it exists  
        if hasattr(self.model, 'model'):
            try:
                nested_model = self.model.model
                nested_model.current_phase = phase_num
                phase_set = phase_set and True
                logger.debug(f"Set model.model.current_phase to {phase_num}")
                
                # Check for detection head in nested model
                if hasattr(nested_model, 'model') and hasattr(nested_model.model, '__iter__'):
                    try:
                        # Look for detection head (usually the last layer)
                        if len(nested_model.model) > 0:
                            last_layer = nested_model.model[-1]
                            last_layer.current_phase = phase_num
                            logger.debug(f"Set detection_head.current_phase to {phase_num}")
                    except Exception as e:
                        logger.debug(f"Could not set phase on detection head: {e}")
                        phase_set = False
            except Exception as e:
                logger.debug(f"Could not set phase on model.model: {e}")
                phase_set = False
        
        # Try to set phase on deeply nested YOLOv5 model
        if hasattr(self.model, 'yolov5_model') and hasattr(self.model.yolov5_model, 'model'):
            try:
                deep_model = self.model.yolov5_model.model
                deep_model.current_phase = phase_num
                logger.debug(f"Set yolov5_model.model.current_phase to {phase_num}")
            except Exception as e:
                logger.debug(f"Could not set phase on yolov5_model.model: {e}")
                phase_set = False
        
        # Also check for detection head specifically (where the phase might be needed)
        if hasattr(self.model, 'head'):
            try:
                head = self.model.head
                head.current_phase = phase_num
                phase_set = phase_set and True
                logger.debug(f"Set model.head.current_phase to {phase_num}")
            except Exception as e:
                logger.debug(f"Could not set phase on model.head: {e}")
                phase_set = False
        
        # Check nested head in YOLOv5 model
        if hasattr(self.model, 'yolov5_model') and hasattr(self.model.yolov5_model, 'head'):
            try:
                yolo_head = self.model.yolov5_model.head
                yolo_head.current_phase = phase_num
                logger.debug(f"Set yolov5_model.head.current_phase to {phase_num}")
            except Exception as e:
                logger.debug(f"Could not set phase on yolov5_model.head: {e}")
                phase_set = False
        
        if phase_set:
            logger.info(f"✅ Model phase successfully set to {phase_num} for layer mode control")
            
            # Verify phase was actually set by checking the model
            actual_phase = getattr(self.model, 'current_phase', None)
            logger.info(f"🔍 Phase verification: model.current_phase = {actual_phase}")
            
            # Also check nested models
            if hasattr(self.model, 'yolov5_model'):
                nested_phase = getattr(self.model.yolov5_model, 'current_phase', None)
                logger.info(f"🔍 Nested model phase: yolov5_model.current_phase = {nested_phase}")
        else:
            logger.warning(f"⚠️ Could not find current_phase attribute to set phase {phase_num}")
    
    def set_single_phase_mode(self, is_single_phase: bool):
        """Set single phase mode flag for proper logging."""
        self._is_single_phase = is_single_phase
    
    @property
    def is_single_phase(self) -> bool:
        """Get single phase mode flag."""
        return self._is_single_phase
    
    def _log_learning_rate_configuration(self, phase_num: int, base_lr: float, phase_config: Dict[str, Any]):
        """Log learning rate configuration for the current phase."""
        logger.info(f"📊 Phase {phase_num} Learning Rate Configuration:")
        
        # Get learning rates from the phase config (which may include overrides)
        learning_rates = phase_config.get('learning_rates', {})
        head_lr = learning_rates.get('head', base_lr)
        backbone_lr = learning_rates.get('backbone', base_lr * 0.1)  # Default scaling
        
        # Log the actual learning rates being used
        logger.info(f"   • Head LR-P{phase_num}: {head_lr}")
        logger.info(f"   • Backbone LR: {backbone_lr}")
        
        # Determine source of learning rates by checking if they differ from args defaults
        # These are the actual defaults from training_args_helper.py
        args_default_head_lr_p1 = 1e-3  # 0.001
        args_default_head_lr_p2 = 1e-4  # 0.0001  
        args_default_backbone_lr = 1e-5  # 0.00001 (same for both phases)
        
        is_custom = False
        if phase_num == 1:
            is_custom = (abs(head_lr - args_default_head_lr_p1) > 1e-9 or abs(backbone_lr - args_default_backbone_lr) > 1e-9)
        else:
            is_custom = (abs(head_lr - args_default_head_lr_p2) > 1e-9 or abs(backbone_lr - args_default_backbone_lr) > 1e-9)
        
        source = "Command-line overrides" if is_custom else "Default configuration"
        logger.info(f"   • Learning rates from: {source}")
    
    def _log_scheduler_configuration(self, phase_num: int, scheduler, optimizer_factory):
        """Log scheduler configuration for the current phase."""
        logger.info(f"⚙️ Phase {phase_num} Scheduler Configuration:")
        
        # Get scheduler configuration from optimizer factory (it has the actual config)
        scheduler_type = optimizer_factory.scheduler_type
        optimizer_type = optimizer_factory.optimizer_type
        weight_decay = optimizer_factory.weight_decay
        
        # Log scheduler information
        if scheduler is not None:
            scheduler_name = scheduler.__class__.__name__
            logger.info(f"   • Scheduler: {scheduler_type} ({scheduler_name})")
            
            # Log scheduler-specific parameters
            if scheduler_type == 'cosine':
                cosine_eta_min = optimizer_factory.cosine_eta_min
                logger.info(f"   • Cosine eta min: {cosine_eta_min}")
                if hasattr(scheduler, 'T_max'):
                    logger.info(f"   • T_max (epochs): {scheduler.T_max}")
            elif scheduler_type == 'step':
                if hasattr(scheduler, 'step_size'):
                    logger.info(f"   • Step size: {scheduler.step_size}")
                if hasattr(scheduler, 'gamma'):
                    logger.info(f"   • Gamma: {scheduler.gamma}")
            elif scheduler_type == 'plateau':
                if hasattr(scheduler, 'factor'):
                    logger.info(f"   • Factor: {scheduler.factor}")
                if hasattr(scheduler, 'patience'):
                    logger.info(f"   • Patience: {scheduler.patience}")
        else:
            logger.info(f"   • Scheduler: None (Phase {phase_num} uses no scheduler)")
        
        # Log optimizer information
        logger.info(f"   • Optimizer: {optimizer_type}")
        logger.info(f"   • Weight decay: {weight_decay}")
        
        # Mixed precision info
        mixed_precision = optimizer_factory.mixed_precision
        logger.info(f"   • Mixed precision: {'Enabled' if mixed_precision else 'Disabled'}")
    
    def _log_memory_optimization_info(self):
        """Log memory optimization and batch size information."""
        try:
            from smartcash.model.utils.memory_optimizer import MemoryOptimizer
            
            # Create memory optimizer to get recommendations
            memory_optimizer = MemoryOptimizer()
            optimal_config = memory_optimizer.get_optimal_training_config()
            
            # Get current config batch size
            current_batch_size = self.config.get('training', {}).get('batch_size', 16)
            
            logger.debug(f"🧠 Memory Optimization Analysis:")
            logger.debug(f"   • Current Batch Size: {current_batch_size}")
            logger.debug(f"   • Recommended Batch Size: {optimal_config.get('batch_size', 'N/A')}")
            logger.debug(f"   • Effective Batch Size: {optimal_config.get('effective_batch_size', 'N/A')}")
            logger.debug(f"   • Gradient Accumulation Steps: {optimal_config.get('gradient_accumulation_steps', 'N/A')}")
            logger.debug(f"   • Device: {memory_optimizer.device}")
            logger.debug(f"   • Platform: {'Apple Silicon' if memory_optimizer.platform_info.get('is_apple_silicon') else 'CUDA' if memory_optimizer.platform_info.get('is_cuda_workstation') else 'CPU'}")
            
            # Warning if batch sizes don't match
            if current_batch_size != optimal_config.get('batch_size'):
                logger.warning(f"⚠️ Current batch size ({current_batch_size}) differs from memory optimizer recommendation ({optimal_config.get('batch_size')})")
                logger.info(f"💡 Consider using the recommended batch size for optimal performance")
                
        except ImportError:
            logger.debug("Memory optimizer not available for batch size recommendations")
        except Exception as e:
            logger.debug(f"Could not get memory optimization info: {e}")
