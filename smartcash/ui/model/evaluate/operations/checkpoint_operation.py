"""
File: smartcash/ui/model/evaluate/operations/checkpoint_operation.py
Description: Operation for checkpoint management and loading
"""

import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List

from smartcash.ui.core.handlers.operation_handler import OperationHandler
from ..constants import EvaluationOperation
from ..services.evaluation_service import EvaluationService


class CheckpointOperation(OperationHandler):
    """Operation handler for checkpoint management."""
    
    def __init__(self, evaluation_service: Optional[EvaluationService] = None):
        """Initialize checkpoint operation.
        
        Args:
            evaluation_service: Optional evaluation service instance
        """
        super().__init__("checkpoint_management")
        self.evaluation_service = evaluation_service or EvaluationService()
        self.operation_type = EvaluationOperation.LOAD_CHECKPOINT
    
    async def execute(self, 
                     config: Dict[str, Any],
                     progress_callback: Optional[Callable] = None,
                     log_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Execute checkpoint operation.
        
        Args:
            config: Operation configuration containing action and checkpoint info
            progress_callback: Progress update callback
            log_callback: Log message callback
            
        Returns:
            Dictionary containing operation results
        """
        try:
            action = config.get("action", "load")
            
            if action == "load":
                return await self._load_checkpoint(config, progress_callback, log_callback)
            elif action == "list":
                return await self._list_checkpoints(config, progress_callback, log_callback)
            elif action == "analyze":
                return await self._analyze_checkpoint(config, progress_callback, log_callback)
            elif action == "select_best":
                return await self._select_best_checkpoint(config, progress_callback, log_callback)
            else:
                raise ValueError(f"Unknown checkpoint action: {action}")
        
        except Exception as e:
            self.logger.error(f"❌ Checkpoint operation failed: {str(e)}")
            return {
                "success": False,
                "operation": self.operation_type.value,
                "error": str(e),
                "message": "Checkpoint operation failed"
            }
    
    async def _load_checkpoint(self, 
                              config: Dict[str, Any],
                              progress_callback: Optional[Callable] = None,
                              log_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Load a specific checkpoint.
        
        Args:
            config: Configuration containing checkpoint path and model info
            progress_callback: Progress callback
            log_callback: Log callback
            
        Returns:
            Dictionary with load results
        """
        model = config.get("model")
        checkpoint_path = config.get("checkpoint_path")
        
        if not model:
            raise ValueError("Model is required for checkpoint loading")
        
        if log_callback:
            log_callback(f"📂 Loading checkpoint for {model} model", "info")
        
        if progress_callback:
            progress_callback(20, "Initializing checkpoint loader...")
        
        # Set up callbacks
        self.evaluation_service.set_callbacks(
            progress_callback=progress_callback,
            log_callback=log_callback
        )
        
        if progress_callback:
            progress_callback(50, "Loading checkpoint...")
        
        # Use evaluation service to load checkpoint
        result = await self.evaluation_service._load_checkpoint(model, checkpoint_path)
        
        if progress_callback:
            progress_callback(100, "Checkpoint loaded successfully" if result["success"] else "Checkpoint loading failed")
        
        if result["success"]:
            checkpoint_info = result["checkpoint_info"]
            if log_callback:
                log_callback(f"✅ Checkpoint loaded: {checkpoint_info.get('path', 'Unknown')}", "info")
                if "map" in checkpoint_info:
                    log_callback(f"📊 Model mAP: {checkpoint_info['map']:.3f}", "info")
            
            return {
                "success": True,
                "operation": self.operation_type.value,
                "result": result,
                "message": f"Successfully loaded checkpoint for {model}"
            }
        else:
            if log_callback:
                log_callback(f"❌ Failed to load checkpoint: {result.get('error', 'Unknown error')}", "error")
            
            return {
                "success": False,
                "operation": self.operation_type.value,
                "error": result.get("error", "Unknown error"),
                "message": f"Failed to load checkpoint for {model}"
            }
    
    async def _list_checkpoints(self, 
                               config: Dict[str, Any],
                               progress_callback: Optional[Callable] = None,
                               log_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """List available checkpoints.
        
        Args:
            config: Configuration containing search parameters
            progress_callback: Progress callback
            log_callback: Log callback
            
        Returns:
            Dictionary with checkpoint list
        """
        model = config.get("model")
        checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
        
        if log_callback:
            log_callback(f"🔍 Searching for checkpoints in {checkpoint_dir}", "info")
        
        if progress_callback:
            progress_callback(30, "Scanning checkpoint directory...")
        
        # Simulate checkpoint discovery
        await asyncio.sleep(0.5)
        
        # Mock checkpoint data (in real implementation, would scan filesystem)
        mock_checkpoints = []
        
        if not model or model == "cspdarknet":
            mock_checkpoints.extend([
                {
                    "path": f"{checkpoint_dir}/cspdarknet/best.pt",
                    "model": "cspdarknet",
                    "epoch": 100,
                    "map": 0.847,
                    "size_mb": 125.3,
                    "modified": "2024-12-10T15:30:00"
                },
                {
                    "path": f"{checkpoint_dir}/cspdarknet/latest.pt",
                    "model": "cspdarknet", 
                    "epoch": 120,
                    "map": 0.852,
                    "size_mb": 125.3,
                    "modified": "2024-12-10T18:45:00"
                }
            ])
        
        if not model or model == "efficientnet_b4":
            mock_checkpoints.extend([
                {
                    "path": f"{checkpoint_dir}/efficientnet_b4/best.pt",
                    "model": "efficientnet_b4",
                    "epoch": 95,
                    "map": 0.891,
                    "size_mb": 87.6,
                    "modified": "2024-12-10T14:20:00"
                },
                {
                    "path": f"{checkpoint_dir}/efficientnet_b4/latest.pt",
                    "model": "efficientnet_b4",
                    "epoch": 110,
                    "map": 0.885,
                    "size_mb": 87.6,
                    "modified": "2024-12-10T17:30:00"
                }
            ])
        
        if progress_callback:
            progress_callback(100, f"Found {len(mock_checkpoints)} checkpoints")
        
        if log_callback:
            log_callback(f"📁 Found {len(mock_checkpoints)} checkpoint(s)", "info")
            for checkpoint in mock_checkpoints:
                log_callback(f"  • {checkpoint['path']} (mAP: {checkpoint['map']:.3f})", "info")
        
        return {
            "success": True,
            "operation": "list_checkpoints",
            "checkpoints": mock_checkpoints,
            "count": len(mock_checkpoints),
            "message": f"Found {len(mock_checkpoints)} checkpoints"
        }
    
    async def _analyze_checkpoint(self, 
                                 config: Dict[str, Any],
                                 progress_callback: Optional[Callable] = None,
                                 log_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Analyze a specific checkpoint.
        
        Args:
            config: Configuration containing checkpoint path
            progress_callback: Progress callback
            log_callback: Log callback
            
        Returns:
            Dictionary with analysis results
        """
        checkpoint_path = config.get("checkpoint_path")
        
        if not checkpoint_path:
            raise ValueError("Checkpoint path is required for analysis")
        
        if log_callback:
            log_callback(f"🔍 Analyzing checkpoint: {checkpoint_path}", "info")
        
        if progress_callback:
            progress_callback(20, "Loading checkpoint for analysis...")
        
        # Simulate checkpoint analysis
        await asyncio.sleep(1.0)
        
        if progress_callback:
            progress_callback(60, "Extracting checkpoint metadata...")
        
        # Mock analysis results
        analysis = {
            "path": checkpoint_path,
            "model_architecture": "YOLO with backbone",
            "backbone": "efficientnet_b4" if "efficientnet" in checkpoint_path else "cspdarknet",
            "total_parameters": 87234567 if "efficientnet" in checkpoint_path else 125678901,
            "model_size_mb": 87.6 if "efficientnet" in checkpoint_path else 125.3,
            "training_metrics": {
                "final_epoch": 95 if "efficientnet" in checkpoint_path else 100,
                "best_map": 0.891 if "efficientnet" in checkpoint_path else 0.847,
                "best_precision": 0.903 if "efficientnet" in checkpoint_path else 0.862,
                "best_recall": 0.878 if "efficientnet" in checkpoint_path else 0.831
            },
            "training_config": {
                "optimizer": "AdamW",
                "learning_rate": 0.001,
                "batch_size": 16,
                "image_size": 640
            },
            "compatibility": {
                "inference_ready": True,
                "format_version": "v2.0",
                "torch_version": "2.0.1"
            }
        }
        
        if progress_callback:
            progress_callback(100, "Analysis completed")
        
        if log_callback:
            log_callback(f"✅ Checkpoint analysis completed", "info")
            log_callback(f"  📊 Best mAP: {analysis['training_metrics']['best_map']:.3f}", "info")
            log_callback(f"  🔧 Parameters: {analysis['total_parameters']:,}", "info")
            log_callback(f"  💾 Size: {analysis['model_size_mb']:.1f} MB", "info")
        
        return {
            "success": True,
            "operation": "analyze_checkpoint",
            "analysis": analysis,
            "message": f"Successfully analyzed checkpoint: {checkpoint_path}"
        }
    
    async def _select_best_checkpoint(self, 
                                     config: Dict[str, Any],
                                     progress_callback: Optional[Callable] = None,
                                     log_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Select the best checkpoint based on criteria.
        
        Args:
            config: Configuration containing selection criteria
            progress_callback: Progress callback
            log_callback: Log callback
            
        Returns:
            Dictionary with best checkpoint selection
        """
        model = config.get("model")
        metric = config.get("metric", "map")
        mode = config.get("mode", "max")
        
        if log_callback:
            log_callback(f"🎯 Selecting best checkpoint for {model} based on {metric}", "info")
        
        if progress_callback:
            progress_callback(20, "Listing available checkpoints...")
        
        # Get available checkpoints
        list_config = {"model": model}
        checkpoint_list = await self._list_checkpoints(list_config)
        
        if not checkpoint_list["success"] or not checkpoint_list["checkpoints"]:
            raise Exception("No checkpoints found for selection")
        
        checkpoints = checkpoint_list["checkpoints"]
        
        if progress_callback:
            progress_callback(60, f"Evaluating {len(checkpoints)} checkpoints...")
        
        # Select best based on metric
        if metric == "map":
            key_func = lambda x: x.get("map", 0)
        elif metric == "epoch":
            key_func = lambda x: x.get("epoch", 0)
        else:
            key_func = lambda x: x.get(metric, 0)
        
        if mode == "max":
            best_checkpoint = max(checkpoints, key=key_func)
        else:
            best_checkpoint = min(checkpoints, key=key_func)
        
        if progress_callback:
            progress_callback(100, "Best checkpoint selected")
        
        if log_callback:
            log_callback(f"✅ Selected best checkpoint: {best_checkpoint['path']}", "info")
            log_callback(f"  📊 {metric}: {best_checkpoint.get(metric, 'N/A')}", "info")
        
        return {
            "success": True,
            "operation": "select_best_checkpoint",
            "best_checkpoint": best_checkpoint,
            "selection_criteria": {"metric": metric, "mode": mode},
            "total_evaluated": len(checkpoints),
            "message": f"Selected best checkpoint based on {metric}"
        }
    
    def get_operation_info(self) -> Dict[str, Any]:
        """Get information about this operation.
        
        Returns:
            Dictionary with operation information
        """
        return {
            "name": "Checkpoint Management",
            "description": "Load, analyze, and manage model checkpoints",
            "operation_type": self.operation_type.value,
            "actions": ["load", "list", "analyze", "select_best"],
            "required_params": ["action"],
            "optional_params": ["model", "checkpoint_path", "checkpoint_dir", "metric", "mode"],
            "estimated_duration": "1-3 minutes"
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
        
        action = config.get("action", "load")
        valid_actions = ["load", "list", "analyze", "select_best"]
        if action not in valid_actions:
            return False, f"Action must be one of: {valid_actions}"
        
        if action == "load":
            if "model" not in config:
                return False, "Model is required for load action"
        
        if action == "analyze":
            if "checkpoint_path" not in config:
                return False, "Checkpoint path is required for analyze action"
        
        if action == "select_best":
            if "model" not in config:
                return False, "Model is required for select_best action"
        
        # Validate model if provided
        if "model" in config:
            valid_models = ["cspdarknet", "efficientnet_b4"]
            if config["model"] not in valid_models:
                return False, f"Model must be one of: {valid_models}"
        
        return True, None
    
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
            "operation_type": "checkpoint_management",
            "supported_actions": ["load", "list", "analyze", "select_best"],
            "supported_models": ["cspdarknet", "efficientnet_b4"]
        }