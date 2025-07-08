"""
File: smartcash/ui/model/train/operations/resume_training_operation.py
Resume training operation handler.
"""

import asyncio
from typing import Dict, Any, Callable

from smartcash.ui.core.handlers.operation_handler import OperationHandler
from ..constants import TrainingOperation, OPERATION_PROGRESS_STEPS
from ..services.training_service import TrainingService


class ResumeTrainingOperation(OperationHandler):
    """Operation handler for resuming training."""
    
    def __init__(self, training_service: TrainingService = None):
        """Initialize the resume training operation."""
        super().__init__(module_name="train", parent_module="model")
        self.operation_type = TrainingOperation.RESUME.value
        self.progress_steps = OPERATION_PROGRESS_STEPS[self.operation_type]
        self.training_service = training_service or TrainingService()
    
    def initialize(self):
        """Initialize the operation handler."""
        # Initialize any required resources
        pass
    
    def get_operations(self) -> Dict[str, Callable]:
        """
        Get available operations.
        
        Returns:
            Dictionary mapping operation names to callable methods
        """
        return {
            'resume_training': self.execute_operation
        }
    
    async def execute_operation(self, checkpoint_path: str,
                              additional_epochs: int = None,
                              progress_callback: Callable = None,
                              log_callback: Callable = None) -> Dict[str, Any]:
        """
        Execute resume training operation.
        
        Args:
            checkpoint_path: Path to checkpoint file
            additional_epochs: Number of additional epochs to train
            progress_callback: Progress update callback
            log_callback: Logging callback
            
        Returns:
            Operation result dictionary
        """
        try:
            operation_id = f"resume_training_{asyncio.get_event_loop().time()}"
            
            if log_callback:
                log_callback(f"🔄 Resuming training from checkpoint: {checkpoint_path}")
            
            # Step 1: Load checkpoint
            if progress_callback:
                progress_callback(10, self.progress_steps[0])
            
            # Step 2: Restore training state
            if progress_callback:
                progress_callback(30, self.progress_steps[1])
            
            # Step 3: Validate model
            if progress_callback:
                progress_callback(50, self.progress_steps[2])
            
            # Step 4: Resume training
            if progress_callback:
                progress_callback(60, self.progress_steps[3])
            
            resume_result = await self.training_service.resume_training(
                checkpoint_path=checkpoint_path,
                additional_epochs=additional_epochs,
                progress_callback=self._create_substep_callback(progress_callback, 60, 95),
                log_callback=log_callback
            )
            
            if progress_callback:
                progress_callback(100, "Resume operation completed")
            
            if log_callback:
                if resume_result.get("success", False):
                    log_callback("✅ Training resumed successfully")
                else:
                    log_callback(f"❌ Resume training failed: {resume_result.get('message', 'Unknown error')}")
            
            return {
                "success": resume_result.get("success", False),
                "message": resume_result.get("message", "Resume training operation completed"),
                "operation_id": operation_id,
                "resume_result": resume_result,
                "phase": resume_result.get("phase", "unknown")
            }
            
        except Exception as e:
            error_msg = f"Resume training operation failed: {str(e)}"
            if log_callback:
                log_callback(f"❌ {error_msg}")
            
            return {
                "success": False,
                "message": error_msg,
                "operation_id": operation_id,
                "error": str(e)
            }
    
    def _create_substep_callback(self, main_callback: Callable, start_percent: int, end_percent: int) -> Callable:
        """Create a substep progress callback that maps to a portion of the main progress."""
        def substep_callback(percent: int, message: str = ""):
            if main_callback:
                # Map substep progress to main progress range
                mapped_percent = start_percent + (percent * (end_percent - start_percent) // 100)
                main_callback(mapped_percent, message)
        
        return substep_callback