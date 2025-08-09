"""
Simplified Training Pipeline - Direct PipelineExecutor Interface

This module provides a simplified interface to the training pipeline,
removing redundancy while maintaining backward compatibility.

Features:
- Direct interface to PipelineExecutor
- Simplified configuration and session management  
- Reduced layers of abstraction
"""

import time
import traceback
from typing import Dict, Any, Optional, Callable

from smartcash.common.logger import get_logger
from smartcash.model.training.utils.progress_tracker import TrainingProgressTracker
from smartcash.model.utils.memory_optimizer import emergency_cleanup
from smartcash.model.training.pipeline import SessionManager, ConfigurationBuilder, PipelineExecutor
from smartcash.model.training.utils.resume_utils import validate_training_mode_and_params

logger = get_logger(__name__)


class TrainingPipeline:
    """
    Simplified training pipeline facade - direct interface to PipelineExecutor.
    
    This class provides a clean API while delegating actual work to PipelineExecutor,
    eliminating redundancy and unnecessary abstraction layers.
    """
    
    def __init__(self, 
                 progress_callback: Optional[Callable] = None, 
                 log_callback: Optional[Callable] = None,
                 metrics_callback: Optional[Callable] = None,
                 live_chart_callback: Optional[Callable] = None,
                 verbose: bool = True):
        """Initialize the training pipeline.
        
        Args:
            progress_callback: Callback for training progress updates
            log_callback: Callback for log messages
            metrics_callback: Callback for training metrics
            live_chart_callback: Callback for live chart updates
            verbose: Whether to print verbose output
        """
        self.verbose = verbose
        self.log_callback = log_callback
        self.metrics_callback = metrics_callback
        self.live_chart_callback = live_chart_callback
        
        # Create progress tracker with only expected parameters
        self.progress_tracker = TrainingProgressTracker(
            progress_callback=progress_callback,
            metrics_callback=metrics_callback
        )
        
        # Create session manager for tracking
        self.session_manager = SessionManager()
        
        logger.info(f"🚀 Training pipeline initialized (simplified)")
    
    def run_training(self, 
                    epochs: int = 10,
                    config: Optional[Dict[str, Any]] = None,
                    **kwargs) -> Dict[str, Any]:
        """
        Run the training pipeline with the given configuration.
        
        Args:
            epochs: Total number of training epochs
            config: Training configuration dictionary
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary containing training results
        """
        try:
            # Create session and extract basic parameters
            session_id, resume_info = self.session_manager.create_session(
                backbone=config.get('backbone', 'cspdarknet') if config else 'cspdarknet',
                training_mode=config.get('training_mode', 'two_phase') if config else 'two_phase',
                resume_from_checkpoint=kwargs.get('resume_from_checkpoint', False),
                checkpoint_dir=config.get('checkpoint_dir', 'data/checkpoints') if config else 'data/checkpoints',
                resume_info=kwargs.get('resume_info')
            )
            
            # Validate parameters
            validate_training_mode_and_params(
                config.get('training_mode', 'two_phase') if config else 'two_phase',
                config.get('single_phase_layer_mode', 'multi') if config else 'multi',
                config.get('single_phase_freeze_backbone', False) if config else False,
                config.get('phase_2_epochs', 1) if config else 1
            )
            
            # Create PipelineExecutor
            executor = PipelineExecutor(
                progress_tracker=self.progress_tracker,
                log_callback=self.log_callback,
                metrics_callback=self.metrics_callback,
                is_resuming=bool(resume_info)
            )
            
            # Merge config with any additional kwargs
            training_config = config.copy() if config else {}
            training_config.update({
                'epochs': epochs,
                'session_id': session_id,
                'start_time': self.session_manager.training_start_time,
                **kwargs
            })
            
            # Execute pipeline
            if resume_info:
                result = executor.execute_resume_pipeline(
                    resume_info=resume_info,
                    config=training_config,
                    session_id=session_id,
                    start_time=self.session_manager.training_start_time
                )
            else:
                result = executor.execute_fresh_pipeline(
                    config=training_config,
                    session_id=session_id,
                    start_time=self.session_manager.training_start_time
                )
            
            return result
            
        except Exception as e:
            error_msg = f"❌ Training pipeline failed: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            
            # Emergency cleanup to free resources
            emergency_cleanup()
            
            # Log error through callback if available
            if self.log_callback:
                self.log_callback('error', error_msg, {
                    'error_type': type(e).__name__,
                    'traceback': traceback.format_exc()
                })
            
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'traceback': traceback.format_exc(),
                'session_id': session_id if 'session_id' in locals() else 'unknown'
            }