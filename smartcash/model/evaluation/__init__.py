"""
File: smartcash/model/evaluation/__init__.py
Deskripsi: Evaluation pipeline exports untuk research scenarios
"""

# Core evaluation components - refactored SRP version
from .evaluation_service import EvaluationService, create_evaluation_service, run_evaluation_pipeline
from .scenario_manager import ScenarioManager, create_scenario_manager, setup_research_scenarios
from .checkpoint_selector import CheckpointSelector, create_checkpoint_selector, get_available_checkpoints

# New modular components
from .evaluators.scenario_evaluator import ScenarioEvaluator, create_scenario_evaluator
from .processors.inference_processor import InferenceProcessor, create_inference_processor
from .processors.data_loader import EvaluationDataLoader, create_evaluation_data_loader

# Evaluation utils
from .utils.evaluation_progress_bridge import EvaluationProgressBridge, create_evaluation_progress_bridge
from .utils.inference_timer import InferenceTimer, create_inference_timer, benchmark_model_inference
from .utils.results_aggregator import ResultsAggregator, create_results_aggregator, aggregate_evaluation_results

# Main evaluation functions
__all__ = [
    # Core services
    'EvaluationService',
    'ScenarioManager', 
    'CheckpointSelector',
    
    # New modular components (SRP-compliant)
    'ScenarioEvaluator',
    'InferenceProcessor',
    'EvaluationDataLoader',
    
    # Utils
    'EvaluationProgressBridge',
    'InferenceTimer',
    'ResultsAggregator',
    
    # Factory functions
    'create_evaluation_service',
    'create_scenario_manager',
    'create_checkpoint_selector',
    'create_scenario_evaluator',
    'create_inference_processor',
    'create_evaluation_data_loader',
    'create_evaluation_progress_bridge',
    'create_inference_timer',
    'create_results_aggregator',
    
    # One-liner functions
    'run_evaluation_pipeline',
    'setup_research_scenarios',
    'get_available_checkpoints',
    'benchmark_model_inference',
    'aggregate_evaluation_results',
]