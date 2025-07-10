"""
File: smartcash/ui/model/evaluate/operations/manager.py
Description: Manager for evaluation operations
"""

from typing import Dict, Any, Optional, List, Tuple
import logging
from dataclasses import dataclass, field
from enum import Enum, auto

class EvaluationOperationType(Enum):
    """Types of evaluation operations"""
    CHECKPOINT_EVALUATION = auto()
    SCENARIO_EVALUATION = auto()
    COMPREHENSIVE_EVALUATION = auto()

@dataclass
class EvaluationOperationConfig:
    """Configuration for an evaluation operation"""
    operation_type: EvaluationOperationType
    config: Dict[str, Any] = field(default_factory=dict)
    ui_components: Dict[str, Any] = field(default_factory=dict)
    callback: Optional[callable] = None

class EvaluationOperationManager:
    """Manager for evaluation operations"""
    
    def __init__(self, config: Dict[str, Any], ui_components: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """Initialize evaluation operation manager.
        
        Args:
            config: Configuration dictionary
            ui_components: Dictionary of UI components
            logger: Optional logger instance
        """
        self.config = config
        self.ui_components = ui_components
        self.logger = logger or logging.getLogger(__name__)
        self._operations: Dict[str, EvaluationOperationConfig] = {}
        self._active_operation: Optional[str] = None
    
    def register_operation(self, operation_id: str, operation_type: EvaluationOperationType, 
                          config: Optional[Dict[str, Any]] = None, 
                          ui_components: Optional[Dict[str, Any]] = None,
                          callback: Optional[callable] = None) -> None:
        """Register a new evaluation operation.
        
        Args:
            operation_id: Unique identifier for the operation
            operation_type: Type of evaluation operation
            config: Operation-specific configuration
            ui_components: UI components for the operation
            callback: Optional callback function to be called on operation completion
        """
        self._operations[operation_id] = EvaluationOperationConfig(
            operation_type=operation_type,
            config=config or {},
            ui_components=ui_components or {},
            callback=callback
        )
        self.logger.info(f"✅ Registered evaluation operation: {operation_id}")
    
    def execute_operation(self, operation_id: str, **kwargs) -> Any:
        """Execute a registered evaluation operation.
        
        Args:
            operation_id: ID of the operation to execute
            **kwargs: Additional arguments for the operation
            
        Returns:
            Result of the operation
            
        Raises:
            KeyError: If operation_id is not registered
        """
        if operation_id not in self._operations:
            raise KeyError(f"No operation registered with ID: {operation_id}")
        
        self._active_operation = operation_id
        operation = self._operations[operation_id]
        
        try:
            self.logger.info(f"🚀 Starting evaluation operation: {operation_id}")
            
            # Execute the appropriate operation based on type
            if operation.operation_type == EvaluationOperationType.CHECKPOINT_EVALUATION:
                result = self._execute_checkpoint_evaluation(operation, **kwargs)
            elif operation.operation_type == EvaluationOperationType.SCENARIO_EVALUATION:
                result = self._execute_scenario_evaluation(operation, **kwargs)
            elif operation.operation_type == EvaluationOperationType.COMPREHENSIVE_EVALUATION:
                result = self._execute_comprehensive_evaluation(operation, **kwargs)
            else:
                raise ValueError(f"Unsupported operation type: {operation.operation_type}")
            
            # Call the callback if provided
            if operation.callback:
                operation.callback(result)
                
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error executing operation {operation_id}: {e}", exc_info=True)
            raise
        finally:
            self._active_operation = None
    
    def _execute_checkpoint_evaluation(self, operation: EvaluationOperationConfig, **kwargs) -> Dict[str, Any]:
        """Execute a checkpoint evaluation operation.
        
        Args:
            operation: Operation configuration
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing evaluation results
        """
        # Import here to avoid circular imports
        from .checkpoint_operation import CheckpointEvaluationOperation
        
        op = CheckpointEvaluationOperation(
            config=operation.config,
            ui_components=operation.ui_components,
            logger=self.logger
        )
        return op.execute(**kwargs)
    
    def _execute_scenario_evaluation(self, operation: EvaluationOperationConfig, **kwargs) -> Dict[str, Any]:
        """Execute a scenario evaluation operation.
        
        Args:
            operation: Operation configuration
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing evaluation results
        """
        # Import here to avoid circular imports
        from .scenario_evaluation_operation import ScenarioEvaluationOperation
        
        op = ScenarioEvaluationOperation(
            config=operation.config,
            ui_components=operation.ui_components,
            logger=self.logger
        )
        return op.execute(**kwargs)
    
    def _execute_comprehensive_evaluation(self, operation: EvaluationOperationConfig, **kwargs) -> Dict[str, Any]:
        """Execute a comprehensive evaluation operation.
        
        Args:
            operation: Operation configuration
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing evaluation results
        """
        # Import here to avoid circular imports
        from .comprehensive_evaluation_operation import ComprehensiveEvaluationOperation
        
        op = ComprehensiveEvaluationOperation(
            config=operation.config,
            ui_components=operation.ui_components,
            logger=self.logger
        )
        return op.execute(**kwargs)
    
    def get_active_operation(self) -> Optional[str]:
        """Get the ID of the currently active operation.
        
        Returns:
            ID of the active operation, or None if no operation is active
        """
        return self._active_operation
    
    def get_operation_config(self, operation_id: str) -> Dict[str, Any]:
        """Get the configuration for a registered operation.
        
        Args:
            operation_id: ID of the operation
            
        Returns:
            Operation configuration dictionary
            
        Raises:
            KeyError: If operation_id is not registered
        """
        if operation_id not in self._operations:
            raise KeyError(f"No operation registered with ID: {operation_id}")
        return self._operations[operation_id].config
    
    def update_operation_config(self, operation_id: str, config_updates: Dict[str, Any]) -> None:
        """Update the configuration for a registered operation.
        
        Args:
            operation_id: ID of the operation
            config_updates: Dictionary of configuration updates
            
        Raises:
            KeyError: If operation_id is not registered
        """
        if operation_id not in self._operations:
            raise KeyError(f"No operation registered with ID: {operation_id}")
        self._operations[operation_id].config.update(config_updates)
        self.logger.info(f"✅ Updated configuration for operation: {operation_id}")
    
    def get_registered_operations(self) -> List[str]:
        """Get a list of all registered operation IDs.
        
        Returns:
            List of operation IDs
        """
        return list(self._operations.keys())
