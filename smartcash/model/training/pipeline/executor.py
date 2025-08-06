"""
Simplified Pipeline Executor

This module provides a lean pipeline executor that coordinates between the
PipelineOrchestrator and ModelManager, following SRP principles.
"""

import os
import glob
from typing import Dict, Any, Optional, Callable

from smartcash.common.logger import get_logger
from smartcash.model.training.phases.mixins.callbacks import CallbacksMixin
from smartcash.model.training.pipeline.configuration_builder import ConfigurationBuilder
from .orchestrator import PipelineOrchestrator
from .model_manager import ModelManager

logger = get_logger(__name__)


class PipelineExecutor(CallbacksMixin):
    """
    Simplified pipeline executor that coordinates model setup and training orchestration.
    
    Responsibilities:
    - Coordinate between ModelManager and PipelineOrchestrator
    - Build training configuration
    - Provide unified interface for training pipeline execution
    
    Does NOT handle:
    - Detailed orchestration logic (delegated to PipelineOrchestrator)
    - Model building/validation (delegated to ModelManager)
    - Individual training loops (delegated to TrainingPhaseManager)
    """
    
    def __init__(self, progress_tracker, log_callback: Optional[Callable] = None,
                 metrics_callback: Optional[Callable] = None,
                 live_chart_callback: Optional[Callable] = None,
                 progress_callback: Optional[Callable] = None,
                 use_yolov5_integration: bool = True,
                 is_resuming: bool = False):
        """
        Initialize simplified pipeline executor.
        
        Args:
            progress_tracker: Progress tracking instance
            log_callback: Callback for logging events
            metrics_callback: Callback for metrics updates
            live_chart_callback: Callback for live chart updates
            progress_callback: Callback for progress updates
            use_yolov5_integration: Enable YOLOv5 integration
        """
        super().__init__()
        
        self.progress_tracker = progress_tracker
        self.use_yolov5_integration = use_yolov5_integration
        
        # Set callbacks using mixin
        self.set_callbacks(
            log_callback=log_callback,
            metrics_callback=metrics_callback,
            live_chart_callback=live_chart_callback,
            progress_callback=progress_callback
        )
        
        # Initialize component managers
        self.model_manager = ModelManager(progress_tracker, log_callback, is_resuming=is_resuming)
        self.orchestrator = PipelineOrchestrator(
            progress_tracker, log_callback, metrics_callback,
            live_chart_callback, progress_callback
        )
    
    def run_training_pipeline(self, session_id: str, **training_kwargs) -> Dict[str, Any]:
        """
        Execute the complete training pipeline.
        
        Args:
            session_id: Training session identifier
            **training_kwargs: Training configuration parameters
            
        Returns:
            Dictionary containing training results
        """
        # Define total overall steps for the progress bar
        TOTAL_OVERALL_STEPS = 3 # Configuration, Model Setup, Pipeline Orchestration
        current_overall_step = 0

        try:
            self.emit_log('info', f'🚀 Starting training pipeline execution for session {session_id}')

            # Step 1: Build training configuration
            self.progress_tracker.update_overall_progress("Building Configuration", current_overall_step, TOTAL_OVERALL_STEPS)
            config = self._build_training_configuration(session_id, **training_kwargs)
            current_overall_step += 1

            # Step 2: Set up model (includes Build Model and initial Validate Model)
            self.progress_tracker.update_overall_progress("Setting up Model", current_overall_step, TOTAL_OVERALL_STEPS)
            model_api, model = self.model_manager.setup_model(config, self.use_yolov5_integration, is_resuming=self.is_resuming)
            current_overall_step += 1

            # Step 3: Execute training pipeline orchestration
            self.progress_tracker.update_overall_progress("Executing Training Pipeline", current_overall_step, TOTAL_OVERALL_STEPS)
            results = self.orchestrator.execute_pipeline(config, model_api, model)
            current_overall_step += 1

            # Add model info to results
            results['model_info'] = self.model_manager.get_model_info()

            self.emit_log('info', '✅ Training pipeline execution completed successfully')

            # Final overall progress update
            self.progress_tracker.update_overall_progress("Completed", TOTAL_OVERALL_STEPS, TOTAL_OVERALL_STEPS)

            return results

        except Exception as e:
            self.emit_log('error', f'❌ Training pipeline execution failed: {str(e)}')
            raise
        finally:
            self._cleanup_resources()
            self.progress_tracker.close() # Ensure all bars are closed on exit
    
    def _build_training_configuration(self, session_id: str, **training_kwargs) -> Dict[str, Any]:
        """Build comprehensive training configuration."""
        try:
            self.emit_log('info', '📋 Building training configuration')
            
            config_builder = ConfigurationBuilder(session_id)
            config = config_builder.build_training_config(**training_kwargs)
            
            self.emit_log('info', '✅ Training configuration built successfully', {
                'session_id': session_id,
                'training_mode': config.get('training', {}).get('training_mode'),
                'backbone': config.get('model', {}).get('backbone'),
                'total_phases': len([k for k in config.get('training_phases', {}) if 'phase_' in k])
            })
            
            return config
            
        except Exception as e:
            self.emit_log('error', f'❌ Failed to build training configuration: {str(e)}')
            raise
    
    def _cleanup_resources(self):
        """Clean up all pipeline resources."""
        try:
            self.model_manager.cleanup()
            self.orchestrator.cleanup_callbacks()  
            self.cleanup_callbacks()
            
            self.emit_log('info', '🧹 Pipeline executor resources cleaned up')
            
        except Exception as e:
            logger.warning(f'⚠️ Error during pipeline executor cleanup: {e}')
    
    # Legacy compatibility methods
    def execute_fresh_pipeline(self, config: Dict[str, Any], session_id: str, start_time: float = None) -> Dict[str, Any]:
        """Legacy method for executing fresh training pipeline."""
        try:
            self.emit_log('info', f'🚀 Executing fresh training pipeline for session {session_id}')

            # Set up model
            model_api, model = self.model_manager.setup_model(config, self.use_yolov5_integration)

            # Delete last checkpoint if it exists (now that checkpoint_manager is initialized)
            last_checkpoint_path = self.model_manager.checkpoint_manager.get_last_checkpoint_path()
            if last_checkpoint_path:
                self.emit_log('info', f'🗑️ Deleting last checkpoint: {last_checkpoint_path}')
                self.model_manager.checkpoint_manager.delete_checkpoint(last_checkpoint_path)

            # Execute training pipeline orchestration
            results = self.orchestrator.execute_pipeline(config, model_api, model)

            # Add model info to results
            results['model_info'] = self.model_manager.get_model_info()
            results['session_id'] = session_id

            self.emit_log('info', '✅ Fresh training pipeline execution completed successfully')

            return results

        except Exception as e:
            self.emit_log('error', f'❌ Fresh training pipeline execution failed: {str(e)}')
            raise
        finally:
            self._cleanup_resources()
    
    def execute_resume_pipeline(self, resume_info: Dict[str, Any], config: Dict[str, Any], 
                               session_id: str, start_time: float = None) -> Dict[str, Any]:
        """Legacy method for executing resumed training pipeline."""
        try:
            self.emit_log('info', f'🔄 Executing resumed training pipeline for session {session_id}')
            
            # Update config with resume information
            config['resume_checkpoint'] = resume_info.get('checkpoint_path')
            config['resume_epoch'] = resume_info.get('epoch', 0)
            config['resume_phase'] = resume_info.get('phase', 1)
            
            # Set up model (may load from checkpoint)
            model_api, model = self.model_manager.setup_model(config, self.use_yolov5_integration)
            
            # Execute training pipeline orchestration
            results = self.orchestrator.execute_pipeline(config, model_api, model)
            
            # Add model info and resume info to results
            results['model_info'] = self.model_manager.get_model_info()
            results['session_id'] = session_id
            results['resume_info'] = resume_info
            
            self.emit_log('info', '✅ Resumed training pipeline execution completed successfully')
            
            return results
            
        except Exception as e:
            self.emit_log('error', f'❌ Resumed training pipeline execution failed: {str(e)}')
            raise
        finally:
            self._cleanup_resources()
    
    def build_model_with_validation(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy method for backward compatibility."""
        try:
            model_api, model = self.model_manager.setup_model(config, self.use_yolov5_integration)
            return {
                'success': True,
                'model_api': model_api,
                'model': model,
                'model_info': self.model_manager.get_model_info()
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def create_training_config(self, session_id: str, **kwargs) -> Dict[str, Any]:
        """Legacy method for backward compatibility."""
        return self._build_training_configuration(session_id, **kwargs)
    
    # Property accessors for backward compatibility
    @property
    def config(self) -> Optional[Dict[str, Any]]:
        """Get current configuration."""
        return self.orchestrator.config if self.orchestrator else None
    
    @property
    def model_api(self):
        """Get model API instance."""
        return self.model_manager.model_api if self.model_manager else None
    
    @property
    def model(self):
        """Get model instance."""
        return self.model_manager.model if self.model_manager else None