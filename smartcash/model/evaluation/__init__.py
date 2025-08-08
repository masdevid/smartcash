"""
File: smartcash/model/evaluation/__init__.py
Deskripsi: Evaluation pipeline exports untuk research scenarios
"""

# Core evaluation components - refactored SRP version
from .evaluation_service import EvaluationService, create_evaluation_service, run_evaluation_pipeline
from .scenario_manager import ScenarioManager, create_scenario_manager, setup_research_scenarios
from .checkpoint_selector import CheckpointSelector, create_checkpoint_selector, get_available_checkpoints
from .processors.scenario_data_source_selector import ScenarioDataSourceSelector, create_scenario_data_source_selector, get_scenario_data_directory, validate_all_scenario_data_sources

# New hierarchical metrics (replaces evaluation_metrics)
from .metrics.hierarchical_metrics_calculator import HierarchicalMetricsCalculator, create_hierarchical_metrics_calculator

# New modular components
from .evaluators.scenario_evaluator import ScenarioEvaluator, create_scenario_evaluator
from .processors.inference_processor import InferenceProcessor, create_inference_processor
from .processors.data_loader import EvaluationDataLoader, create_evaluation_data_loader
from .converters.yolov5_format_converter import YOLOv5FormatConverter, create_yolov5_format_converter
from .utils.model_config_extractor import ModelConfigExtractor, create_model_config_extractor

# Evaluation utils
from .utils.evaluation_progress_bridge import EvaluationProgressBridge, create_evaluation_progress_bridge
from .utils.inference_timer import InferenceTimer, create_inference_timer, benchmark_model_inference
from .utils.results_aggregator import ResultsAggregator, create_results_aggregator, aggregate_evaluation_results

# New managers for backbone comparison and chart generation
from .managers.backbone_comparison_manager import BackboneComparisonManager, create_backbone_comparison_manager, run_backbone_comparison_evaluation
from .managers.chart_generation_manager import ChartGenerationManager, create_chart_generation_manager, generate_backbone_comparison_charts

# Main evaluation functions
__all__ = [
    # Core services
    'EvaluationService',
    'ScenarioManager', 
    'CheckpointSelector',
    'ScenarioDataSourceSelector',
    
    # New modular components (SRP-compliant)
    'HierarchicalMetricsCalculator',
    'ScenarioEvaluator',
    'InferenceProcessor',
    'EvaluationDataLoader',
    'YOLOv5FormatConverter',
    'ModelConfigExtractor',
    
    # Utils
    'EvaluationProgressBridge',
    'InferenceTimer',
    'ResultsAggregator',
    
    # Managers
    'BackboneComparisonManager',
    'ChartGenerationManager',
    
    # Factory functions
    'create_evaluation_service',
    'create_scenario_manager',
    'create_checkpoint_selector',
    'create_scenario_data_source_selector',
    'create_hierarchical_metrics_calculator',
    'create_scenario_evaluator',
    'create_inference_processor',
    'create_evaluation_data_loader',
    'create_yolov5_format_converter',
    'create_model_config_extractor',
    'create_evaluation_progress_bridge',
    'create_inference_timer',
    'create_results_aggregator',
    'create_backbone_comparison_manager',
    'create_chart_generation_manager',
    
    # One-liner functions
    'run_evaluation_pipeline',
    'setup_research_scenarios',
    'get_available_checkpoints',
    'get_scenario_data_directory',
    'validate_all_scenario_data_sources',
    'benchmark_model_inference',
    'aggregate_evaluation_results',
    'run_backbone_comparison_evaluation',
    'generate_backbone_comparison_charts'
]