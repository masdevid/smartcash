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
                 use_yolov5_integration: bool = True,
                 progress_callback: Optional[Callable] = None, 
                 log_callback: Optional[Callable] = None,
                 metrics_callback: Optional[Callable] = None,
                 live_chart_callback: Optional[Callable] = None,
                 verbose: bool = True):
        """Initialize simplified training pipeline facade."""
        # Store parameters for PipelineExecutor
        self.use_yolov5_integration = use_yolov5_integration
        self.verbose = verbose
        self.log_callback = log_callback
        self.metrics_callback = metrics_callback
        self.live_chart_callback = live_chart_callback
        
        # Create progress tracker
        self.progress_tracker = TrainingProgressTracker(
            progress_callback=progress_callback,
            verbose=verbose
        )
        
        # Create session manager for tracking
        self.session_manager = SessionManager()
        
        logger.info(f"🚀 Training pipeline initialized (simplified)")
    
    def run_full_training_pipeline(self, **kwargs) -> Dict[str, Any]:
        """
        Simplified training pipeline - direct delegation to PipelineExecutor.
        
        This method has been streamlined to remove redundancy while maintaining
        the same interface for backward compatibility.
        """
        try:
            # Create session and extract basic parameters
            session_id, resume_info = self.session_manager.create_session(
                backbone=kwargs.get('backbone', 'cspdarknet'),
                training_mode=kwargs.get('training_mode', 'two_phase'),
                resume_from_checkpoint=kwargs.get('resume_from_checkpoint', True),
                checkpoint_dir=kwargs.get('checkpoint_dir', 'data/checkpoints'),
                resume_info=kwargs.get('resume_info')
            )
            
            # Validate parameters
            validate_training_mode_and_params(
                kwargs.get('training_mode', 'two_phase'), 
                kwargs.get('single_phase_layer_mode', 'multi'),
                kwargs.get('single_phase_freeze_backbone', False), 
                kwargs.get('phase_2_epochs', 1)
            )
            
            # Create PipelineExecutor
            executor = PipelineExecutor(
                progress_tracker=self.progress_tracker,
                log_callback=self.log_callback,
                metrics_callback=self.metrics_callback,
                use_yolov5_integration=self.use_yolov5_integration
            )
            
            # Build configuration
            config_builder = ConfigurationBuilder(session_id)
            config = config_builder.build_training_config(**kwargs)
            
            # Execute pipeline directly
            if resume_info:
                result = executor.execute_resume_pipeline(
                    resume_info, config, session_id, self.session_manager.training_start_time
                )
            else:
                result = executor.execute_fresh_pipeline(
                    config, session_id, self.session_manager.training_start_time
                )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Training pipeline failed: {str(e)}")
            emergency_cleanup()
            
            if self.log_callback:
                self.log_callback('error', f"Training pipeline failed: {traceback.format_exc()}", {
                    'error_type': type(e).__name__
                })
            
            return {
                'success': False,
                'error': traceback.format_exc(),
                'session_id': getattr(self, 'session_id', 'unknown')
            }