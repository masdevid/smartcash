"""
File: smartcash/ui/model/evaluate/operations/comprehensive_evaluation_operation.py
Description: Operation for comprehensive evaluation across multiple scenarios
"""

import asyncio
from typing import Dict, Any, Optional, Callable, List

from smartcash.ui.core.handlers.operation_handler import OperationHandler
from ..constants import EvaluationOperation, DEFAULT_TEST_MATRIX
from ..services.evaluation_service import EvaluationService


class ComprehensiveEvaluationOperation(OperationHandler):
    """Operation handler for comprehensive scenario evaluation."""
    
    def __init__(self, evaluation_service: Optional[EvaluationService] = None):
        """Initialize comprehensive evaluation operation.
        
        Args:
            evaluation_service: Optional evaluation service instance
        """
        super().__init__("comprehensive_evaluation")
        self.evaluation_service = evaluation_service or EvaluationService()
        self.operation_type = EvaluationOperation.TEST_ALL_SCENARIOS
    
    async def execute(self, 
                     config: Dict[str, Any],
                     progress_callback: Optional[Callable] = None,
                     log_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Execute comprehensive evaluation operation.
        
        Args:
            config: Operation configuration 
            progress_callback: Progress update callback
            log_callback: Log message callback
            
        Returns:
            Dictionary containing operation results
        """
        try:
            # Extract configuration
            scenarios = config.get("scenarios", ["position_variation", "lighting_variation"])
            models = config.get("models", ["cspdarknet", "efficientnet_b4"])
            test_matrix = config.get("test_matrix", DEFAULT_TEST_MATRIX)
            selected_metrics = config.get("selected_metrics")
            
            # Set up callbacks
            self.evaluation_service.set_callbacks(
                progress_callback=progress_callback,
                log_callback=log_callback
            )
            
            # Log operation start
            total_tests = len(scenarios) * len(models)
            self.logger.info(f"🚀 Starting comprehensive evaluation: {total_tests} tests")
            
            if log_callback:
                log_callback(f"Evaluating scenarios: {', '.join(scenarios)}", "info")
                log_callback(f"Evaluating models: {', '.join(models)}", "info")
            
            # Execute comprehensive evaluation
            result = await self.evaluation_service.run_comprehensive_evaluation(
                scenarios=scenarios,
                models=models,
                selected_metrics=selected_metrics
            )
            
            if result["success"]:
                self.logger.info(f"🎉 Comprehensive evaluation completed successfully")
                
                # Log summary
                if log_callback and "summary" in result:
                    summary = result["summary"]
                    if summary["success"]:
                        log_callback(f"✅ Best scenario: {summary.get('best_scenario', 'N/A')}", "info")
                        log_callback(f"✅ Best model: {summary.get('best_model', 'N/A')}", "info")
                        
                        overall_metrics = summary.get("overall_metrics", {})
                        if "map" in overall_metrics:
                            log_callback(f"📊 Overall mAP: {overall_metrics['map']:.3f}", "info")
                
                return {
                    "success": True,
                    "operation": self.operation_type.value,
                    "result": result,
                    "message": f"Successfully completed {result['completed_tests']}/{result['total_tests']} evaluations"
                }
            else:
                errors = result.get("errors", [])
                error_msg = f"Comprehensive evaluation completed with {len(errors)} errors"
                self.logger.warning(f"⚠️ {error_msg}")
                
                return {
                    "success": False,
                    "operation": self.operation_type.value,
                    "result": result,
                    "error": error_msg,
                    "message": error_msg
                }
        
        except Exception as e:
            self.logger.error(f"❌ Comprehensive evaluation operation failed: {str(e)}")
            return {
                "success": False,
                "operation": self.operation_type.value,
                "error": str(e),
                "message": "Comprehensive evaluation operation failed"
            }
    
    def get_operation_info(self) -> Dict[str, Any]:
        """Get information about this operation.
        
        Returns:
            Dictionary with operation information
        """
        return {
            "name": "Comprehensive Evaluation",
            "description": "Run comprehensive evaluation across all scenarios and models",
            "operation_type": self.operation_type.value,
            "required_params": [],
            "optional_params": ["scenarios", "models", "test_matrix"],
            "estimated_duration": "10-20 minutes",
            "default_test_matrix": DEFAULT_TEST_MATRIX
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
            "service_status": self.evaluation_service.get_current_status(),
            "default_test_matrix": DEFAULT_TEST_MATRIX
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
        
        # Validate scenarios if provided
        if "scenarios" in config:
            scenarios = config["scenarios"]
            if not isinstance(scenarios, list):
                return False, "Scenarios must be a list"
            
            valid_scenarios = ["position_variation", "lighting_variation"]
            for scenario in scenarios:
                if scenario not in valid_scenarios:
                    return False, f"Invalid scenario: {scenario}. Must be one of: {valid_scenarios}"
        
        # Validate models if provided
        if "models" in config:
            models = config["models"]
            if not isinstance(models, list):
                return False, "Models must be a list"
            
            valid_models = ["cspdarknet", "efficientnet_b4"]
            for model in models:
                if model not in valid_models:
                    return False, f"Invalid model: {model}. Must be one of: {valid_models}"
        
        # Validate test matrix if provided
        if "test_matrix" in config:
            test_matrix = config["test_matrix"]
            if not isinstance(test_matrix, list):
                return False, "Test matrix must be a list"
            
            for i, test in enumerate(test_matrix):
                if not isinstance(test, dict):
                    return False, f"Test matrix item {i} must be a dictionary"
                
                if "scenario" not in test or "model" not in test:
                    return False, f"Test matrix item {i} must have 'scenario' and 'model' keys"
        
        return True, None