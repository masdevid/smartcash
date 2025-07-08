"""
File: smartcash/ui/model/evaluate/services/evaluation_service.py
Description: Service for managing model evaluation operations with backend integration
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple
from datetime import datetime

from smartcash.ui.logger import get_module_logger
from ..constants import (
    DEFAULT_CONFIG, EvaluationOperation, EvaluationPhase, 
    TestScenario, BackboneModel, OPERATION_MESSAGES
)


class EvaluationService:
    """Service for managing model evaluation operations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize evaluation service.
        
        Args:
            config: Optional configuration override
        """
        self.logger = get_module_logger(__name__)
        self.config = config or DEFAULT_CONFIG.copy()
        
        # State management
        self.current_phase = EvaluationPhase.IDLE
        self.current_operation = None
        self.evaluation_results = {}
        self.active_scenarios = set()
        self.loaded_checkpoints = {}
        
        # Callbacks
        self.progress_callback: Optional[Callable] = None
        self.log_callback: Optional[Callable] = None
        self.metrics_callback: Optional[Callable] = None
        
        # Backend integration flag
        self._backend_available = None
        
        self.logger.info("🎯 Evaluation service initialized")
    
    def set_callbacks(self, 
                     progress_callback: Optional[Callable] = None,
                     log_callback: Optional[Callable] = None,
                     metrics_callback: Optional[Callable] = None) -> None:
        """Set callbacks for UI integration.
        
        Args:
            progress_callback: Progress update callback
            log_callback: Log message callback
            metrics_callback: Metrics update callback
        """
        self.progress_callback = progress_callback
        self.log_callback = log_callback
        self.metrics_callback = metrics_callback
        self.logger.debug("📡 Callbacks configured for evaluation service")
    
    def _log_message(self, message: str, level: str = "info") -> None:
        """Log message with callback support."""
        getattr(self.logger, level)(message)
        if self.log_callback:
            try:
                self.log_callback(message, level)
            except Exception as e:
                self.logger.error(f"Error in log callback: {e}")
    
    def _update_progress(self, progress: int, message: str = "") -> None:
        """Update progress with callback support."""
        if self.progress_callback:
            try:
                self.progress_callback(progress, message)
            except Exception as e:
                self.logger.error(f"Error in progress callback: {e}")
    
    def _update_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update metrics with callback support."""
        if self.metrics_callback:
            try:
                self.metrics_callback(metrics)
            except Exception as e:
                self.logger.error(f"Error in metrics callback: {e}")
    
    def _check_backend_availability(self) -> bool:
        """Check if backend evaluation services are available."""
        if self._backend_available is not None:
            return self._backend_available
        
        try:
            # Try to import backend evaluation components
            from smartcash.model.evaluation import EvaluationService as BackendEvaluationService
            from smartcash.model.evaluation import ScenarioManager, CheckpointSelector
            from smartcash.dataset.augmentor import service as augmentor_service
            
            self._backend_available = True
            self.logger.info("✅ Backend evaluation services available")
            return True
            
        except ImportError as e:
            self._backend_available = False
            self.logger.warning(f"⚠️ Backend evaluation services not available: {e}")
            return False
    
    async def run_scenario_evaluation(self, 
                                    scenario: str, 
                                    model: str,
                                    checkpoint_path: Optional[str] = None,
                                    selected_metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run evaluation for a specific scenario and model.
        
        Args:
            scenario: Scenario name (position_variation or lighting_variation)
            model: Model name (cspdarknet or efficientnet_b4)
            checkpoint_path: Optional specific checkpoint path
            selected_metrics: List of metrics to calculate
            
        Returns:
            Dictionary containing evaluation results
        """
        # Use selected metrics or default to all metrics
        if selected_metrics is None:
            from ..constants import DEFAULT_ENABLED_METRICS
            selected_metrics = DEFAULT_ENABLED_METRICS
        self.current_phase = EvaluationPhase.AUGMENTING
        self.current_operation = EvaluationOperation.TEST_SCENARIO
        
        operation_msg = OPERATION_MESSAGES[EvaluationOperation.TEST_SCENARIO]
        self._log_message(operation_msg["start"])
        
        try:
            # Phase 1: Dataset Augmentation
            self._log_message(f"🎨 Augmenting test dataset for {scenario} scenario")
            self._update_progress(10, "Preparing test data...")
            
            augmentation_result = await self._augment_test_dataset(scenario)
            if not augmentation_result["success"]:
                raise Exception(f"Dataset augmentation failed: {augmentation_result.get('error', 'Unknown error')}")
            
            # Phase 2: Load checkpoint
            self.current_phase = EvaluationPhase.EVALUATING
            self._update_progress(30, "Loading model checkpoint...")
            
            checkpoint_result = await self._load_checkpoint(model, checkpoint_path)
            if not checkpoint_result["success"]:
                raise Exception(f"Checkpoint loading failed: {checkpoint_result.get('error', 'Unknown error')}")
            
            # Phase 3: Run evaluation
            self._update_progress(50, f"Running {scenario} evaluation...")
            
            evaluation_result = await self._run_backend_evaluation(
                scenario, model, augmentation_result["dataset_path"], checkpoint_result["checkpoint_info"], selected_metrics
            )
            
            # Phase 4: Process results
            self.current_phase = EvaluationPhase.GENERATING_REPORT
            self._update_progress(80, "Processing evaluation results...")
            
            processed_results = await self._process_evaluation_results(evaluation_result, scenario, model)
            
            # Phase 5: Complete
            self.current_phase = EvaluationPhase.COMPLETED
            self._update_progress(100, "Evaluation completed successfully")
            
            # Store results
            result_key = f"{scenario}_{model}"
            self.evaluation_results[result_key] = processed_results
            
            # Update metrics callback
            self._update_metrics(processed_results["metrics"])
            
            self._log_message(operation_msg["success"], "info")
            
            return {
                "success": True,
                "scenario": scenario,
                "model": model,
                "results": processed_results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.current_phase = EvaluationPhase.ERROR
            error_msg = f"{operation_msg['error']}: {str(e)}"
            self._log_message(error_msg, "error")
            
            return {
                "success": False,
                "scenario": scenario,
                "model": model,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def run_comprehensive_evaluation(self, 
                                         scenarios: Optional[List[str]] = None,
                                         models: Optional[List[str]] = None,
                                         selected_metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run comprehensive evaluation across multiple scenarios and models.
        
        Args:
            scenarios: List of scenarios to test (default: all)
            models: List of models to test (default: all)
            selected_metrics: List of metrics to calculate
            
        Returns:
            Dictionary containing comprehensive evaluation results
        """
        scenarios = scenarios or ["position_variation", "lighting_variation"]
        models = models or ["cspdarknet", "efficientnet_b4"]
        
        self.current_operation = EvaluationOperation.TEST_ALL_SCENARIOS
        operation_msg = OPERATION_MESSAGES[EvaluationOperation.TEST_ALL_SCENARIOS]
        
        self._log_message(operation_msg["start"])
        
        total_tests = len(scenarios) * len(models)
        completed_tests = 0
        results = {}
        errors = []
        
        try:
            for scenario in scenarios:
                for model in models:
                    completed_tests += 1
                    progress_msg = operation_msg["progress"].format(
                        current=completed_tests, total=total_tests
                    )
                    self._log_message(progress_msg)
                    
                    # Run individual scenario evaluation
                    result = await self.run_scenario_evaluation(scenario, model, selected_metrics=selected_metrics)
                    
                    result_key = f"{scenario}_{model}"
                    results[result_key] = result
                    
                    if not result["success"]:
                        errors.append(f"{scenario}/{model}: {result.get('error', 'Unknown error')}")
                    
                    # Update overall progress
                    overall_progress = int((completed_tests / total_tests) * 100)
                    self._update_progress(overall_progress, f"Completed {completed_tests}/{total_tests} evaluations")
            
            # Generate summary
            summary = self._generate_evaluation_summary(results)
            
            if errors:
                self._log_message(f"⚠️ Completed with {len(errors)} errors", "warning")
                for error in errors:
                    self._log_message(f"  ❌ {error}", "warning")
            else:
                self._log_message(operation_msg["success"], "info")
            
            return {
                "success": len(errors) == 0,
                "total_tests": total_tests,
                "completed_tests": completed_tests,
                "errors": errors,
                "results": results,
                "summary": summary,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            error_msg = f"{operation_msg['error']}: {str(e)}"
            self._log_message(error_msg, "error")
            
            return {
                "success": False,
                "error": str(e),
                "results": results,
                "timestamp": datetime.now().isoformat()
            }
    
    async def _augment_test_dataset(self, scenario: str) -> Dict[str, Any]:
        """Augment test dataset for specific scenario.
        
        Args:
            scenario: Scenario name
            
        Returns:
            Dictionary with augmentation results
        """
        if not self._check_backend_availability():
            # Simulation mode
            await asyncio.sleep(1.0)  # Simulate augmentation time
            return {
                "success": True,
                "dataset_path": f"data/evaluation/{scenario}/augmented",
                "num_variations": 5,
                "simulation": True
            }
        
        try:
            # Use real backend augmentation service
            from smartcash.dataset.augmentor import service as augmentor_service
            
            config = self.config["evaluation"]["scenarios"][scenario]["augmentation_config"]
            
            # Configure augmentation based on scenario
            if scenario == "position_variation":
                augment_config = {
                    "rotation": {"min": -30, "max": 30},
                    "translation": {"min": -0.2, "max": 0.2},
                    "scale": {"min": 0.8, "max": 1.2},
                    "num_variations": config["num_variations"]
                }
            elif scenario == "lighting_variation":
                augment_config = {
                    "brightness": {"min": -0.3, "max": 0.3},
                    "contrast": {"min": 0.7, "max": 1.3},
                    "gamma": {"min": 0.7, "max": 1.3},
                    "num_variations": config["num_variations"]
                }
            else:
                raise ValueError(f"Unknown scenario: {scenario}")
            
            # Run augmentation
            test_dir = self.config["evaluation"]["data"]["test_dir"]
            output_dir = f"{self.config['evaluation']['data']['evaluation_dir']}/{scenario}/augmented"
            
            result = await augmentor_service.augment_dataset(
                input_dir=test_dir,
                output_dir=output_dir,
                config=augment_config
            )
            
            return {
                "success": True,
                "dataset_path": output_dir,
                "num_variations": result.get("num_generated", config["num_variations"]),
                "simulation": False
            }
            
        except Exception as e:
            self.logger.error(f"Dataset augmentation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _load_checkpoint(self, model: str, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """Load model checkpoint.
        
        Args:
            model: Model name
            checkpoint_path: Optional specific checkpoint path
            
        Returns:
            Dictionary with checkpoint loading results
        """
        if not self._check_backend_availability():
            # Simulation mode
            await asyncio.sleep(0.5)  # Simulate loading time
            return {
                "success": True,
                "checkpoint_info": {
                    "path": checkpoint_path or f"checkpoints/best_{model}.pt",
                    "model": model,
                    "epoch": 100,
                    "map": 0.85 + (hash(model) % 100) / 1000  # Simulated mAP
                },
                "simulation": True
            }
        
        try:
            from smartcash.model.evaluation import CheckpointSelector
            
            checkpoint_selector = CheckpointSelector()
            
            if checkpoint_path:
                # Use specific checkpoint
                checkpoint_info = checkpoint_selector.analyze_checkpoint(checkpoint_path)
            else:
                # Auto-select best checkpoint for model
                checkpoints = checkpoint_selector.find_checkpoints(
                    model_dir=f"checkpoints/{model}",
                    pattern="*.pt"
                )
                
                if not checkpoints:
                    raise Exception(f"No checkpoints found for model: {model}")
                
                checkpoint_info = checkpoint_selector.select_best_checkpoint(
                    checkpoints, 
                    metric="map",
                    mode="max"
                )
            
            return {
                "success": True,
                "checkpoint_info": checkpoint_info,
                "simulation": False
            }
            
        except Exception as e:
            self.logger.error(f"Checkpoint loading failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _run_backend_evaluation(self, 
                                    scenario: str, 
                                    model: str, 
                                    dataset_path: str,
                                    checkpoint_info: Dict[str, Any],
                                    selected_metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run backend evaluation.
        
        Args:
            scenario: Scenario name
            model: Model name
            dataset_path: Path to augmented dataset
            checkpoint_info: Checkpoint information
            selected_metrics: List of metrics to calculate
            
        Returns:
            Dictionary with evaluation results
        """
        if not self._check_backend_availability():
            # Simulation mode - generate realistic fake results
            await asyncio.sleep(2.0)  # Simulate evaluation time
            
            # Generate scenario-specific simulated results
            base_metrics = {
                "map": 0.78,
                "precision": 0.82,
                "recall": 0.76,
                "f1_score": 0.79,
                "inference_time": 45.2
            }
            
            # Adjust metrics based on scenario and model BEFORE filtering
            if scenario == "position_variation":
                base_metrics["map"] *= 0.92  # Position variation is more challenging
                base_metrics["precision"] *= 0.94
                base_metrics["recall"] *= 0.90
            
            if model == "efficientnet_b4":
                base_metrics["map"] *= 1.05  # EfficientNet performs better
                base_metrics["precision"] *= 1.03
                base_metrics["inference_time"] *= 1.2  # But slower
            
            # Recalculate F1 score if precision and recall are available
            if "precision" in base_metrics and "recall" in base_metrics:
                p, r = base_metrics["precision"], base_metrics["recall"]
                base_metrics["f1_score"] = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
            
            # Filter metrics based on selection AFTER adjustments
            if selected_metrics:
                filtered_metrics = {k: v for k, v in base_metrics.items() if k in selected_metrics}
                base_metrics = filtered_metrics
            
            return {
                "success": True,
                "metrics": base_metrics,
                "simulation": True,
                "num_images": 150,
                "total_detections": 287
            }
        
        try:
            from smartcash.model.evaluation import EvaluationService as BackendEvaluationService
            
            backend_evaluator = EvaluationService(self.config)
            
            # Run evaluation
            result = await backend_evaluator.run_scenario(
                scenario_name=scenario,
                checkpoint_path=checkpoint_info["path"],
                dataset_path=dataset_path,
                selected_metrics=selected_metrics
            )
            
            return {
                "success": True,
                "metrics": result["metrics"],
                "simulation": False,
                "num_images": result.get("num_images", 0),
                "total_detections": result.get("total_detections", 0)
            }
            
        except Exception as e:
            self.logger.error(f"Backend evaluation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _process_evaluation_results(self, 
                                        evaluation_result: Dict[str, Any],
                                        scenario: str,
                                        model: str) -> Dict[str, Any]:
        """Process and format evaluation results.
        
        Args:
            evaluation_result: Raw evaluation results
            scenario: Scenario name
            model: Model name
            
        Returns:
            Processed results dictionary
        """
        if not evaluation_result["success"]:
            return evaluation_result
        
        metrics = evaluation_result["metrics"]
        
        # Format metrics for display
        formatted_metrics = {}
        for metric_name, value in metrics.items():
            if metric_name == "inference_time":
                formatted_metrics[metric_name] = f"{value:.2f}ms"
            else:
                formatted_metrics[metric_name] = f"{value:.3f}"
        
        # Calculate performance grade
        map_score = metrics.get("map", 0)
        if map_score >= 0.9:
            grade = "A+"
        elif map_score >= 0.8:
            grade = "A"
        elif map_score >= 0.7:
            grade = "B"
        elif map_score >= 0.6:
            grade = "C"
        else:
            grade = "D"
        
        return {
            "scenario": scenario,
            "model": model,
            "metrics": metrics,
            "formatted_metrics": formatted_metrics,
            "grade": grade,
            "num_images": evaluation_result.get("num_images", 0),
            "total_detections": evaluation_result.get("total_detections", 0),
            "simulation": evaluation_result.get("simulation", False),
            "timestamp": datetime.now().isoformat()
        }
    
    def _generate_evaluation_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of comprehensive evaluation results.
        
        Args:
            results: Dictionary of all evaluation results
            
        Returns:
            Summary dictionary
        """
        successful_results = [r for r in results.values() if r["success"]]
        
        if not successful_results:
            return {"success": False, "message": "No successful evaluations"}
        
        # Extract metrics from successful results
        all_metrics = []
        scenario_metrics = {}
        model_metrics = {}
        
        for result in successful_results:
            if "results" in result and "metrics" in result["results"]:
                metrics = result["results"]["metrics"]
                all_metrics.append(metrics)
                
                scenario = result["scenario"]
                model = result["model"]
                
                if scenario not in scenario_metrics:
                    scenario_metrics[scenario] = []
                if model not in model_metrics:
                    model_metrics[model] = []
                
                scenario_metrics[scenario].append(metrics)
                model_metrics[model].append(metrics)
        
        if not all_metrics:
            return {"success": False, "message": "No metrics available"}
        
        # Calculate overall averages
        avg_metrics = {}
        for metric_name in ["map", "precision", "recall", "f1_score", "inference_time"]:
            values = [m.get(metric_name, 0) for m in all_metrics if metric_name in m]
            if values:
                avg_metrics[metric_name] = sum(values) / len(values)
        
        # Calculate scenario averages
        scenario_averages = {}
        for scenario, metrics_list in scenario_metrics.items():
            scenario_avg = {}
            for metric_name in ["map", "precision", "recall", "f1_score"]:
                values = [m.get(metric_name, 0) for m in metrics_list if metric_name in m]
                if values:
                    scenario_avg[metric_name] = sum(values) / len(values)
            scenario_averages[scenario] = scenario_avg
        
        # Calculate model averages
        model_averages = {}
        for model, metrics_list in model_metrics.items():
            model_avg = {}
            for metric_name in ["map", "precision", "recall", "f1_score"]:
                values = [m.get(metric_name, 0) for m in metrics_list if metric_name in m]
                if values:
                    model_avg[metric_name] = sum(values) / len(values)
            model_averages[model] = model_avg
        
        # Determine best performing combinations
        best_scenario = max(scenario_averages.items(), key=lambda x: x[1].get("map", 0))[0] if scenario_averages else None
        best_model = max(model_averages.items(), key=lambda x: x[1].get("map", 0))[0] if model_averages else None
        
        return {
            "success": True,
            "overall_metrics": avg_metrics,
            "scenario_performance": scenario_averages,
            "model_performance": model_averages,
            "best_scenario": best_scenario,
            "best_model": best_model,
            "total_evaluations": len(successful_results),
            "timestamp": datetime.now().isoformat()
        }
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current evaluation service status.
        
        Returns:
            Status dictionary
        """
        return {
            "phase": self.current_phase.value if self.current_phase else "idle",
            "operation": self.current_operation.value if self.current_operation else None,
            "active_scenarios": list(self.active_scenarios),
            "loaded_checkpoints": list(self.loaded_checkpoints.keys()),
            "num_results": len(self.evaluation_results),
            "backend_available": self._check_backend_availability()
        }
    
    def get_evaluation_results(self) -> Dict[str, Any]:
        """Get all evaluation results.
        
        Returns:
            Dictionary of all stored evaluation results
        """
        return self.evaluation_results.copy()
    
    def clear_results(self) -> None:
        """Clear all stored evaluation results."""
        self.evaluation_results.clear()
        self.active_scenarios.clear()
        self.loaded_checkpoints.clear()
        self.current_phase = EvaluationPhase.IDLE
        self.current_operation = None
        self.logger.info("🧹 Evaluation results cleared")