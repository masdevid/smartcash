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
from smartcash.model.training.metrics_tracker import MetricsTracker
from smartcash.model.training.optimizer_factory import OptimizerFactory
from smartcash.model.training.loss_manager import LossManager
from smartcash.model.training.utils.early_stopping import create_early_stopping

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
    
    def setup_phase(self, phase_num: int, epochs: int) -> Dict[str, Any]:
        """
        Set up training phase with all required components.
        
        Args:
            phase_num: Training phase number (1 or 2)
            epochs: Total number of epochs to train for
            
        Returns:
            Dictionary containing all setup components
        """
        try:
            phase_config = self.config['training_phases'][f'phase_{phase_num}']
            
            # Set up data loaders
            data_factory = DataLoaderFactory(self.config)
            train_loader = data_factory.create_train_loader()
            val_loader = data_factory.create_val_loader()
            
            # Set up loss manager
            loss_manager = LossManager(self.config)
            
            # Configure model phase
            self._configure_model_phase(phase_num)
            
            # Set up optimizer and scheduler
            optimizer_factory = OptimizerFactory(self.config)
            base_lr = phase_config.get('learning_rate', 0.001)
            optimizer = optimizer_factory.create_optimizer(self.model, base_lr)
            scheduler = optimizer_factory.create_scheduler(optimizer, total_epochs=epochs)
            
            # Set up mixed precision scaler
            scaler = None
            if self.config['training']['mixed_precision']:
                import torch
                scaler = torch.amp.GradScaler('cuda')
            
            # Set up metrics tracker
            metrics_tracker = MetricsTracker(config=self.config)
            
            # Set up early stopping
            early_stopping = self._setup_early_stopping(phase_num)
            
            return {
                'train_loader': train_loader,
                'val_loader': val_loader,
                'loss_manager': loss_manager,
                'optimizer': optimizer,
                'scheduler': scheduler,
                'scaler': scaler,
                'metrics_tracker': metrics_tracker,
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
        elif training_mode == 'two_phase':
            # Two-phase mode: Phase 1 = single layer, Phase 2 = multi layer
            if phase_num == 1:
                logger.info(f"🎯 Phase {phase_num}: single layer mode (layer_1 only)")
            else:
                logger.info(f"🎯 Phase {phase_num}: multi-layer mode (all layers)")
            self.model.force_single_layer = False  # Use phase-based logic
    
    def _setup_early_stopping(self, phase_num: int):
        """Set up phase-specific early stopping configuration."""
        es_config = self.config['training']['early_stopping'].copy()
        
        # Check if phase-specific early stopping is enabled
        # For two-phase mode, Phase 1 should have early stopping disabled by default
        training_mode = self.config.get('training_mode', 'single_phase')
        if training_mode == 'two_phase':
            phase_1_enabled = es_config.get('phase_1_enabled', False)  # Disabled by default for Phase 1
            phase_2_enabled = es_config.get('phase_2_enabled', True)   # Enabled by default for Phase 2
        else:
            phase_1_enabled = es_config.get('phase_1_enabled', True)   # Single-phase respects config
            phase_2_enabled = es_config.get('phase_2_enabled', True)
        
        # For two-phase mode, apply phase-specific early stopping logic
        if self.config.get('training_mode') == 'two_phase':
            if phase_num == 1:
                es_config['enabled'] = phase_1_enabled
                if not phase_1_enabled:
                    logger.info(f"🚫 Early stopping disabled for Phase {phase_num}")
            elif phase_num == 2:
                es_config['enabled'] = phase_2_enabled
                if phase_2_enabled:
                    logger.info(f"✅ Early stopping enabled for Phase {phase_num}")
                else:
                    logger.info(f"🚫 Early stopping disabled for Phase {phase_num}")
        
        early_stopping = create_early_stopping({'training': {'early_stopping': es_config}})
        logger.info(f"🔍 Early stopping object type: {type(early_stopping).__name__} | Config enabled: {es_config.get('enabled', True)}")
        
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
        else:
            logger.warning(f"⚠️ Could not find current_phase attribute to set phase {phase_num}")
    
    def set_single_phase_mode(self, is_single_phase: bool):
        """Set single phase mode flag for proper logging."""
        self._is_single_phase = is_single_phase
    
    @property
    def is_single_phase(self) -> bool:
        """Get single phase mode flag."""
        return self._is_single_phase