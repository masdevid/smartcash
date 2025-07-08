"""
File: smartcash/ui/model/train/operations/start_training_operation.py
Start training operation handler.
"""

import asyncio
from typing import Dict, Any, Callable

from smartcash.ui.core.handlers.operation_handler import OperationHandler
from ..constants import TrainingOperation, OPERATION_PROGRESS_STEPS
from ..services.training_service import TrainingService


class StartTrainingOperation(OperationHandler):
    """Operation handler for starting training."""
    
    def __init__(self):
        """Initialize the start training operation."""
        super().__init__(module_name="train", parent_module="model")
        self.operation_type = TrainingOperation.START.value
        self.progress_steps = OPERATION_PROGRESS_STEPS[self.operation_type]
        self.training_service = TrainingService()
    
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
            'start_training': self.execute_operation
        }
    
    async def execute_operation(self, config: Dict[str, Any], 
                              progress_callback: Callable = None,
                              log_callback: Callable = None,
                              metrics_callback: Callable = None) -> Dict[str, Any]:
        """
        Execute start training operation.
        
        Args:
            config: Training configuration
            progress_callback: Progress update callback
            log_callback: Logging callback
            metrics_callback: Metrics update callback
            
        Returns:
            Operation result dictionary
        """
        try:
            operation_id = f"start_training_{asyncio.get_event_loop().time()}"
            
            if log_callback:
                log_callback("🚀 Starting training operation...")
            
            # Step 1: Initialize training
            if progress_callback:
                progress_callback(10, self.progress_steps[0])
            
            init_result = await self.training_service.initialize_training(
                config=config,
                progress_callback=self._create_substep_callback(progress_callback, 10, 30),
                log_callback=log_callback
            )
            
            if not init_result.get("success", False):
                return {
                    "success": False,
                    "message": f"Training initialization failed: {init_result.get('message', 'Unknown error')}",
                    "operation_id": operation_id
                }
            
            # Step 2: Start training process
            if progress_callback:
                progress_callback(30, self.progress_steps[4])
            
            training_result = await self.training_service.start_training(
                epochs=config.get("training", {}).get("epochs"),
                progress_callback=self._create_substep_callback(progress_callback, 30, 95),
                log_callback=log_callback
            )
            
            if progress_callback:
                progress_callback(100, self.progress_steps[5])
            
            if log_callback:
                if training_result.get("success", False):
                    log_callback("✅ Training operation completed successfully")
                else:
                    log_callback(f"❌ Training operation failed: {training_result.get('message', 'Unknown error')}")
            
            return {
                "success": training_result.get("success", False),
                "message": training_result.get("message", "Training operation completed"),
                "operation_id": operation_id,
                "training_result": training_result,
                "metrics": training_result.get("final_metrics", {}),
                "phase": training_result.get("phase", "unknown")
            }
            
        except Exception as e:
            error_msg = f"Start training operation failed: {str(e)}"
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