"""
Merged Training Pipeline - SmartCash YOLOv5 Integration

This module merges training_pipeline.py and unified_training_pipeline.py
into a single, optimized pipeline following SRP principles.

Features:
- YOLOv5 integration with SmartCash architecture
- Unified progress tracking with comprehensive callbacks
- Platform-aware configuration and optimization
- Two-phase and single-phase training modes
- Automatic checkpoint management and resume capability
- Memory optimization and cleanup
- Real-time visualization and metrics reporting
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable

from smartcash.common.logger import get_logger
# Removed unified_training_pipeline import - functionality merged here
from smartcash.model.training.utils.progress_tracker import TrainingProgressTracker
from smartcash.model.utils.memory_optimizer import emergency_cleanup
from smartcash.model.training.pipeline import SessionManager, ConfigurationBuilder, PipelineExecutor
from smartcash.model.training.utils.resume_utils import setup_training_session, validate_training_mode_and_params

logger = get_logger(__name__)


class TrainingPipeline:
    """
    Merged training pipeline combining YOLOv5 integration with unified functionality.
    
    This class consolidates the best features from both training_pipeline.py and
    unified_training_pipeline.py while eliminating duplication.
    """
    
    def __init__(self, 
                 use_yolov5_integration: bool = True,
                 progress_callback: Optional[Callable] = None, 
                 log_callback: Optional[Callable] = None,
                 metrics_callback: Optional[Callable] = None,
                 live_chart_callback: Optional[Callable] = None,
                 verbose: bool = True):
        """
        Initialize merged training pipeline.
        
        Args:
            use_yolov5_integration: Enable YOLOv5 integration
            progress_callback: Callback for progress updates
            log_callback: Callback for logging events
            metrics_callback: Callback for metrics updates
            live_chart_callback: Callback for live chart updates
            verbose: Enable verbose logging
        """
        self.use_yolov5_integration = use_yolov5_integration
        self.verbose = verbose
        
        # Core components
        self.config = None
        self.model_api = None
        self.model = None
        self.visualization_manager = None
        self.memory_optimizer = None
        self.session_manager = SessionManager()
        self.config_builder = None
        self.pipeline_executor = None
        
        # Callbacks
        self.log_callback = log_callback
        self.metrics_callback = metrics_callback
        self.live_chart_callback = live_chart_callback
        
        # Training state
        self.current_phase = None
        self.training_session_id = None
        self.phase_start_time = None
        self.training_start_time = None
        self._training_mode = None
        
        # Initialize progress tracker (will be updated based on training mode)
        self.progress_tracker = TrainingProgressTracker(
            progress_callback=progress_callback,
            verbose=verbose
        )
        
        logger.info(f"🚀 Training pipeline initialized")
        logger.info(f"🔧 YOLOv5 integration: {'enabled' if use_yolov5_integration else 'disabled'}")
    
    def run_full_training_pipeline(self,
                                  backbone: str = 'cspdarknet',
                                  pretrained: bool = True,
                                  phase_1_epochs: int = 1,
                                  phase_2_epochs: int = 1,
                                  checkpoint_dir: str = 'data/checkpoints',
                                  resume_from_checkpoint: bool = True,
                                  force_cpu: bool = False,
                                  training_mode: str = 'two_phase',
                                  single_phase_layer_mode: str = 'multi',
                                  single_phase_freeze_backbone: bool = False,
                                  # Learning rate configuration
                                  phase_1_lr: float = 0.001,
                                  phase_2_lr: float = 0.0001,
                                  # Early stopping configuration
                                  patience: int = 10,
                                  min_delta: float = 0.001,
                                  monitor: str = 'val_loss',
                                  # Training configuration
                                  batch_size: int = 8,
                                  num_workers: Optional[int] = None,
                                  model: Optional[Dict[str, Any]] = None,
                                  **kwargs) -> Dict[str, Any]:
        """
        SmartCash training pipeline with YOLOv5 architecture
        
        Args:
            backbone: Backbone type ('cspdarknet', 'efficientnet_b4')
            pretrained: Use pretrained weights
            phase_1_epochs: Epochs for phase 1 (frozen backbone)
            phase_2_epochs: Epochs for phase 2 (fine-tuning)
            checkpoint_dir: Directory for checkpoints
            resume_from_checkpoint: Enable checkpoint resuming
            force_cpu: Force CPU training
            training_mode: 'two_phase' or 'single_phase'
            single_phase_layer_mode: Layer mode for single phase
            single_phase_freeze_backbone: Freeze backbone in single phase
            phase_1_lr: Learning rate for phase 1
            phase_2_lr: Learning rate for phase 2
            patience: Early stopping patience (epochs)
            min_delta: Minimum change for early stopping
            monitor: Metric to monitor for early stopping
            batch_size: Training batch size
            num_workers: Number of data loading workers (auto-detected if None)
            model: Model configuration dictionary
            **kwargs: Additional arguments
            
        Returns:
            Training result dictionary
        """
        try:
            # Create training session
            self.training_session_id, resume_info = self.session_manager.create_session(
                backbone=backbone,
                training_mode=training_mode,
                resume_from_checkpoint=resume_from_checkpoint,
                checkpoint_dir=checkpoint_dir,
                resume_info=kwargs.get('resume_info')  # Support pre-loaded resume info
            )
            self.training_start_time = self.session_manager.training_start_time
            
            # Initialize specialized components with session
            self.config_builder = ConfigurationBuilder(self.training_session_id)
            self.pipeline_executor = PipelineExecutor(
                progress_tracker=self.progress_tracker,
                log_callback=self.log_callback,
                metrics_callback=self.metrics_callback,
                use_yolov5_integration=self.use_yolov5_integration
            )
            
            # Set up enhanced progress tracker with training mode
            if self._training_mode != training_mode:
                self._training_mode = training_mode
                current_callback = getattr(self.progress_tracker, 'progress_callback', None)
                self.progress_tracker = TrainingProgressTracker(
                    progress_callback=current_callback,
                    verbose=self.verbose,
                    training_mode=training_mode
                )
                # Update executor's progress tracker
                self.pipeline_executor.progress_tracker = self.progress_tracker
            
            # Log training start
            if self.log_callback:
                self.log_callback('info', f"🚀 Starting training pipeline", {
                    'session_id': self.training_session_id,
                    'backbone': backbone,
                    'training_mode': training_mode
                })
            
            # Validate training mode and parameters
            validate_training_mode_and_params(
                training_mode, single_phase_layer_mode, single_phase_freeze_backbone, phase_2_epochs
            )
            
            # Setup training session with resume capability
            # Note: resume_info might already be set from session manager (pre-loaded checkpoint)
            if resume_from_checkpoint and resume_info is None:
                # Only call setup_training_session if we don't already have resume_info
                _, resume_info = setup_training_session(
                    resume_from_checkpoint, checkpoint_dir, backbone
                )
            
            # Setup training configuration
            training_config = self._setup_training_config(
                backbone=backbone,
                pretrained=pretrained,
                phase_1_epochs=phase_1_epochs,
                phase_2_epochs=phase_2_epochs,
                checkpoint_dir=checkpoint_dir,
                training_mode=training_mode,
                single_phase_layer_mode=single_phase_layer_mode,
                single_phase_freeze_backbone=single_phase_freeze_backbone,
                phase_1_lr=phase_1_lr,
                phase_2_lr=phase_2_lr,
                patience=patience,
                min_delta=min_delta,
                monitor=monitor,
                batch_size=batch_size,
                num_workers=num_workers,
                model=model,
                force_cpu=force_cpu,
                **kwargs
            )
            
            # Execute pipeline phases
            if resume_info:
                # Resume from checkpoint
                result = self._execute_resume_pipeline(
                    resume_info, training_config
                )
            else:
                # Fresh training
                result = self._execute_fresh_pipeline(training_config)
            
            return result
            
            # This return is now handled in _execute_* methods
            
        except Exception as e:
            logger.error(f"❌ Training pipeline failed: {str(e)}")
            
            # Emergency cleanup
            try:
                emergency_cleanup()
            except:
                pass
            
            if self.log_callback:
                self.log_callback('error', f"Training pipeline failed: {str(e)}", {
                    'session_id': self.training_session_id,
                    'error_type': type(e).__name__
                })
            
            return {
                'success': False,
                'error': str(e),
                'session_id': self.training_session_id
            }
    
    def _setup_training_config(self, **kwargs) -> Dict[str, Any]:
        """Setup training configuration"""
        
        # Base configuration
        config = {
            'backbone': kwargs.get('backbone', 'cspdarknet'),
            'pretrained': kwargs.get('pretrained', True),
            'training_mode': kwargs.get('training_mode', 'two_phase'),
            'phase_1_epochs': kwargs.get('phase_1_epochs', 1),
            'phase_2_epochs': kwargs.get('phase_2_epochs', 1),
            'checkpoint_dir': Path(kwargs.get('checkpoint_dir', 'data/checkpoints')),
            'force_cpu': kwargs.get('force_cpu', False),
            'session_id': self.training_session_id
        }
        
        # Model configuration
        model_config = kwargs.get('model', {})
        config['model'] = {
            'model_name': model_config.get('model_name', 'smartcash_yolov5_integrated'),
            'backbone': config['backbone'],
            'pretrained': config['pretrained'],
            'layer_mode': model_config.get('layer_mode', 'multi'),
            'detection_layers': model_config.get('detection_layers', ['layer_1', 'layer_2', 'layer_3']),
            'num_classes': model_config.get('num_classes', 7),
            'img_size': model_config.get('img_size', 640),
            'feature_optimization': model_config.get('feature_optimization', {'enabled': True})
        }
        
        # Training phases configuration for TrainingPhaseManager
        config['training_phases'] = {
            'phase_1': {
                'learning_rate': kwargs.get('phase_1_lr', kwargs.get('head_lr_p1', 0.001)),
                'freeze_backbone': True,
                'layer_mode': 'single',
                'description': 'Frozen backbone training'
            },
            'phase_2': {
                'learning_rate': kwargs.get('phase_2_lr', kwargs.get('head_lr_p2', 0.0001)),
                'freeze_backbone': False,
                'layer_mode': 'multi',
                'description': 'Fine-tuning with multi-layer'
            }
        }
        
        # Auto-detect num_workers if not specified
        if kwargs.get('num_workers') is None:
            auto_workers = 0 if config['force_cpu'] else 4
        else:
            auto_workers = kwargs.get('num_workers')
        
        # Training configuration
        config['training'] = {
            'mixed_precision': False,  # Disable for CPU
            'batch_size': kwargs.get('batch_size', 8),
            'num_workers': auto_workers,
            'pin_memory': False,
            'training_mode': config['training_mode'],  # Add training mode to training config
            'early_stopping': {
                'enabled': kwargs.get('early_stopping_enabled', not kwargs.get('no_early_stopping', False)),
                'patience': kwargs.get('patience', kwargs.get('early_stopping_patience', 10)),
                'min_delta': kwargs.get('min_delta', kwargs.get('early_stopping_min_delta', 0.001)),
                'monitor': kwargs.get('monitor', kwargs.get('early_stopping_metric', 'val_loss'))
            }
        }
        
        # Paths configuration
        config['paths'] = {
            'checkpoints': str(config['checkpoint_dir']),
            'visualization': 'data/visualization',
            'logs': 'data/logs'
        }
        
        # Always use YOLOv5 architecture
        
        # Apply training overrides from kwargs
        from smartcash.model.training.utils.setup_utils import apply_training_overrides
        config = apply_training_overrides(config, **kwargs)
        
        logger.info(f"🔧 Training configuration setup: {config['backbone']} | yolov5")
        return config
    
    def _execute_fresh_pipeline(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute fresh training pipeline using pipeline executor."""
        return self.pipeline_executor.execute_fresh_pipeline(
            config, self.training_session_id, self.training_start_time
        )
    
    def _execute_resume_pipeline(self, resume_info: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute resumed training pipeline using pipeline executor."""
        return self.pipeline_executor.execute_resume_pipeline(
            resume_info, config, self.training_session_id, self.training_start_time
        )
    
    # Phase methods now handled by pipeline executor
    
    
    
    
    
    
    def _emit_log(self, level: str, message: str, data: dict = None):
        """Emit log event to UI via log callback."""
        if self.log_callback:
            try:
                log_data = {
                    'timestamp': time.time(),
                    'phase': self.current_phase,
                    'session_id': self.training_session_id,
                    'message': message,
                    'data': data or {}
                }
                self.log_callback(level, message, log_data)
            except Exception as e:
                logger.warning(f"Log callback error: {e}")


def run_training_pipeline(**kwargs) -> Dict[str, Any]:
    """
    Main entry point for merged training pipeline.
    
    Args:
        **kwargs: Training configuration arguments
        
    Returns:
        Training result dictionary
    """
    # Extract pipeline-specific arguments
    use_yolov5_integration = kwargs.pop('use_yolov5_integration', True)
    progress_callback = kwargs.pop('progress_callback', None)
    log_callback = kwargs.pop('log_callback', None)
    metrics_callback = kwargs.pop('metrics_callback', None)
    live_chart_callback = kwargs.pop('live_chart_callback', None)
    verbose = kwargs.pop('verbose', True)
    
    # Create and run pipeline
    pipeline = TrainingPipeline(
        use_yolov5_integration=use_yolov5_integration,
        progress_callback=progress_callback,
        log_callback=log_callback,
        metrics_callback=metrics_callback,
        live_chart_callback=live_chart_callback,
        verbose=verbose
    )
    
    return pipeline.run_full_training_pipeline(**kwargs)


# Maintain compatibility with existing API
def run_full_training_pipeline(**kwargs) -> Dict[str, Any]:
    """
    Compatibility wrapper for existing training pipeline API.
    
    Args:
        **kwargs: Training configuration arguments
        
    Returns:
        Training result dictionary
    """
    return run_training_pipeline(**kwargs)


# Export key functions
__all__ = [
    'TrainingPipeline',
    'run_training_pipeline',
    'run_full_training_pipeline'
]