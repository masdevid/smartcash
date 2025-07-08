"""
File: smartcash/ui/model/evaluate/operations/scenario_evaluation_operation.py
Description: Operation for evaluating individual scenarios
"""

import asyncio
from typing import Dict, Any, Optional, Callable

from smartcash.ui.core.handlers.operation_handler import OperationHandler
from ..constants import EvaluationOperation
from ..services.evaluation_service import EvaluationService


class ScenarioEvaluationOperation(OperationHandler):
    """Operation handler for evaluating individual scenarios."""
    
    def __init__(self, evaluation_service: Optional[EvaluationService] = None):
        """Initialize scenario evaluation operation.
        
        Args:
            evaluation_service: Optional evaluation service instance
        """
        super().__init__("scenario_evaluation")
        self.evaluation_service = evaluation_service or EvaluationService()
        self.operation_type = EvaluationOperation.TEST_SCENARIO
    
    async def execute(self, 
                     config: Dict[str, Any],
                     progress_callback: Optional[Callable] = None,
                     log_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Execute scenario evaluation operation.
        
        Args:
            config: Operation configuration containing scenario and model info
            progress_callback: Progress update callback
            log_callback: Log message callback
            
        Returns:
            Dictionary containing operation results
        """
        try:
            # Extract configuration
            scenario = config.get("scenario")
            model = config.get("model")
            checkpoint_path = config.get("checkpoint_path")
            selected_metrics = config.get("selected_metrics")
            
            if not scenario or not model:
                raise ValueError("Scenario and model are required")
            
            # Set up callbacks
            self.evaluation_service.set_callbacks(
                progress_callback=progress_callback,
                log_callback=log_callback
            )
            
            # Log operation start
            self.logger.info(f"🧪 Starting scenario evaluation: {scenario} with {model}")
            
            # Execute evaluation
            result = await self.evaluation_service.run_scenario_evaluation(
                scenario=scenario,
                model=model,
                checkpoint_path=checkpoint_path,
                selected_metrics=selected_metrics
            )
            
            if result["success"]:
                self.logger.info(f"✅ Scenario evaluation completed successfully")
                return {
                    "success": True,
                    "operation": self.operation_type.value,
                    "result": result,
                    "message": f"Successfully evaluated {scenario} scenario with {model} model"
                }
            else:
                self.logger.error(f"❌ Scenario evaluation failed: {result.get('error', 'Unknown error')}")
                return {
                    "success": False,
                    "operation": self.operation_type.value,
                    "error": result.get("error", "Unknown error"),
                    "message": f"Failed to evaluate {scenario} scenario"
                }
        
        except Exception as e:
            self.logger.error(f"❌ Scenario evaluation operation failed: {str(e)}")
            return {
                "success": False,
                "operation": self.operation_type.value,
                "error": str(e),
                "message": "Scenario evaluation operation failed"
            }
    
    def get_operation_info(self) -> Dict[str, Any]:
        """Get information about this operation.
        
        Returns:
            Dictionary with operation information
        """
        return {
            "name": "Scenario Evaluation",
            "description": "Evaluate model performance on a specific scenario",
            "operation_type": self.operation_type.value,
            "required_params": ["scenario", "model"],
            "optional_params": ["checkpoint_path"],
            "estimated_duration": "2-5 minutes"
        }
    
    def get_operations(self) -> Dict[str, Callable]:
        """Get available operations.
        
        Returns:
            Dictionary of available operations
        """
        return {
            "execute": self.execute,
            "validate_config": self.validate_config,
            "get_operation_info": self.get_operation_info
        }
    
    def initialize(self) -> Dict[str, Any]:
        """Initialize the operation handler.
        
        Returns:
            Dictionary containing initialization results
        """
        return {
            "success": True,
            "operation_type": self.operation_type.value,
            "service_status": self.evaluation_service.get_current_status()
        }
    
    def validate_config(self, config: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate operation configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(config, dict):
            return False, "Configuration must be a dictionary"
        
        if "scenario" not in config:
            return False, "Scenario is required"
        
        if "model" not in config:
            return False, "Model is required"
        
        valid_scenarios = ["position_variation", "lighting_variation"]
        if config["scenario"] not in valid_scenarios:
            return False, f"Scenario must be one of: {valid_scenarios}"
        
        valid_models = ["cspdarknet", "efficientnet_b4"]
        if config["model"] not in valid_models:
            return False, f"Model must be one of: {valid_models}"
        
        return True, None