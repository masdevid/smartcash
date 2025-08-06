"""
Component setup mixin for phase management.

Handles setup of training components like optimizers, schedulers, and callbacks.
"""

from typing import Dict, Any, Optional, Tuple
from smartcash.common.logger import get_logger

logger = get_logger(__name__)


class ComponentSetupMixin:
    """Mixin for setting up training components."""
    
    def setup_training_components(self, phase_num: int, epochs: int, 
                                save_best_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Set up all training components for a phase.
        
        Args:
            phase_num: Phase number
            epochs: Total number of epochs
            save_best_path: Path to save best model
            
        Returns:
            Dictionary containing all setup components
        """
        from smartcash.model.training.data_loader_factory import DataLoaderFactory
        from smartcash.model.training.utils.metrics_history import create_metrics_recorder
        from smartcash.model.training.optimizer_factory import OptimizerFactory
        from smartcash.model.training.loss_manager import LossManager
        
        logger = get_logger(self.__class__.__name__)
        
        try:
            phase_config = self.config['training_phases'][f'phase_{phase_num}']
            
            # Set up data loaders
            data_factory = DataLoaderFactory(self.config)
            train_loader = data_factory.create_train_loader()
            val_loader = data_factory.create_val_loader()
            
            # Set up loss manager with phase awareness
            loss_manager = LossManager(self.config)
            loss_manager.set_current_phase(phase_num)
            
            # Set up optimizer and scheduler
            optimizer_factory = OptimizerFactory(self.config)
            base_lr = phase_config.get('learning_rate', 0.001)
            
            # Log learning rate configuration
            self._log_learning_rate_configuration(phase_num, base_lr, phase_config)
            
            optimizer = optimizer_factory.create_optimizer(self.model, base_lr)
            scheduler = optimizer_factory.create_scheduler(optimizer, total_epochs=epochs)
            
            # Set up mixed precision scaler
            scaler = None
            if self.config['training']['mixed_precision']:
                import torch
                scaler = torch.amp.GradScaler('cuda')
            
            # Set up metrics recorder
            backbone = getattr(self.config, 'backbone', 'unknown') if hasattr(self.config, 'backbone') else self.config.get('backbone', 'unknown')
            data_name = getattr(self.config, 'data_name', 'data') if hasattr(self.config, 'data_name') else self.config.get('data_name', 'data')
            resume_mode = self._get_resume_mode()
            
            metrics_recorder = create_metrics_recorder(
                output_dir="outputs",
                backbone=backbone,
                data_name=data_name, 
                phase_num=phase_num,
                resume_mode=resume_mode
            )
            
            # Set up early stopping
            early_stopping = self.setup_early_stopping(phase_num, save_best_path)
            
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
    
    def setup_early_stopping(self, phase_num: int, save_best_path: Optional[str] = None):
        """Set up early stopping for the phase with best metrics preservation."""
        from smartcash.model.training.early_stopping import create_early_stopping, create_phase_specific_early_stopping
        
        logger = get_logger(self.__class__.__name__)
        es_config = self.config.get('training', {}).get('early_stopping', {})
        
        logger.info(f"🔍 Early stopping config: {es_config}")
        
        use_phase_specific = es_config.get('phase_specific', True)
        
        if use_phase_specific:
            logger.info(f"🎯 Using phase-specific early stopping for Phase {phase_num}")
            early_stopping = create_phase_specific_early_stopping(self.config)
            
            # Set save_best_path if provided
            if save_best_path and hasattr(early_stopping, 'save_best_path'):
                early_stopping.save_best_path = save_best_path
                logger.info(f"💾 Phase-specific early stopping will save best model to: {save_best_path}")
            
            early_stopping.set_phase(phase_num)
        else:
            # Use legacy early stopping logic
            phase_enabled = es_config.get(f'phase_{phase_num}_enabled', True)
            
            if phase_enabled:
                logger.info(f"✅ Legacy early stopping enabled for Phase {phase_num}")
                early_stopping = create_early_stopping(self.config, save_best_path)
            else:
                logger.info(f"🚫 Early stopping disabled for Phase {phase_num}")
                return None
        
        # Configure early stopping with previous best metrics if checkpoint manager is available
        if hasattr(self, 'checkpoint_manager') and self.checkpoint_manager:
            try:
                configured = self.checkpoint_manager.configure_early_stopping_with_best_metrics(
                    early_stopping, phase_num
                )
                if configured:
                    logger.info(f"✅ Configured early stopping with previous best metrics for Phase {phase_num}")
                else:
                    logger.info(f"ℹ️ No previous best metrics found - early stopping will start fresh")
            except Exception as e:
                logger.warning(f"⚠️ Failed to configure early stopping with best metrics: {e}")
        
        return early_stopping
    
    def _get_resume_mode(self) -> bool:
        """Determine if training is in resume mode."""
        if hasattr(self.config, '__getitem__'):  # dict-like
            resume_value = self.config.get('resume', False)
            resume_checkpoint_value = self.config.get('resume_checkpoint')
            return resume_value or bool(resume_checkpoint_value)
        else:  # object-like
            resume_value = getattr(self.config, 'resume', False)
            resume_checkpoint_value = getattr(self.config, 'resume_checkpoint', None)
            return resume_value or bool(resume_checkpoint_value)
    
    def _log_learning_rate_configuration(self, phase_num: int, base_lr: float, phase_config: Dict[str, Any]):
        """Log comprehensive learning rate and optimizer configuration."""
        logger = get_logger(self.__class__.__name__)
        
        # Learning Rate Configuration
        logger.info(f"📊 Phase {phase_num} Learning Rate Configuration:")
        
        # Get learning rates from phase config
        learning_rates = phase_config.get('learning_rates', {})
        head_lr_p1 = self.config.get('training_phases', {}).get('phase_1', {}).get('learning_rates', {}).get('head', 'default')
        head_lr_p2 = self.config.get('training_phases', {}).get('phase_2', {}).get('learning_rates', {}).get('head', 'default') 
        backbone_lr = learning_rates.get('backbone', 'default')
        
        logger.info(f"• Head LR (Phase 1): {head_lr_p1}")
        logger.info(f"• Head LR (Phase 2): {head_lr_p2}")
        logger.info(f"• Backbone LR: {backbone_lr}")
        
        # Determine if learning rates are from command line or default
        lr_source = "Command line arguments" if any(
            self.config.get('training_phases', {}).get(f'phase_{i}', {}).get('learning_rates', {}) 
            for i in [1, 2]
        ) else "Default configuration"
        logger.info(f"• Learning rates from: {lr_source}")
        
        # Optimizer and Scheduler Configuration
        logger.info(f"⚙️ Phase {phase_num} Scheduler Configuration:")
        
        # Get training config for optimizer/scheduler settings
        training_config = self.config.get('training', {})
        optimizer_type = training_config.get('optimizer', 'adamw')
        scheduler_type = training_config.get('scheduler', 'cosine')
        weight_decay = training_config.get('weight_decay', 0.01)
        mixed_precision = training_config.get('mixed_precision', False)
        
        # Scheduler-specific settings
        if scheduler_type == 'cosine':
            cosine_eta_min = training_config.get('cosine_eta_min', 1e-6)
            # Estimate T_max from phase epochs
            phase_epochs = phase_config.get('epochs', 30)
            logger.info(f"• Scheduler: {scheduler_type} (CosineAnnealingLR)")
            logger.info(f"• Cosine eta min: {cosine_eta_min}")
            logger.info(f"• T_max (epochs): {phase_epochs}")
        else:
            logger.info(f"• Scheduler: {scheduler_type}")
        
        logger.info(f"• Optimizer: {optimizer_type}")
        logger.info(f"• Weight decay: {weight_decay}")
        logger.info(f"• Mixed precision: {'Enabled' if mixed_precision else 'Disabled'}")