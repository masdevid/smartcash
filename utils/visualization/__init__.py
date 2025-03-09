"""
File: smartcash/utils/visualization/__init__.py
Author: Alfrida Sabar
Deskripsi: File inisialisasi untuk paket visualisasi SmartCash
"""

from smartcash.utils.visualization.base import VisualizationHelper
from smartcash.utils.visualization.detection import DetectionVisualizer, visualize_detection
from smartcash.utils.visualization.metrics import MetricsVisualizer, plot_confusion_matrix
from smartcash.utils.visualization.research import ResearchVisualizer, visualize_scenario_comparison
from smartcash.utils.visualization.experiment_visualizer import ExperimentVisualizer
from smartcash.utils.visualization.scenario_visualizer import ScenarioVisualizer
from smartcash.utils.visualization.research_utils import (
    clean_dataframe, format_metric_name, find_common_metrics, 
    create_benchmark_table, create_win_rate_table
)

# Import dari paket analysis
from smartcash.utils.visualization.analysis import ExperimentAnalyzer, ScenarioAnalyzer
from smartcash.utils.visualization.result_visualizer import ResultVisualizer
from smartcash.utils.visualization.evaluation_visualizer import EvaluationVisualizer


# Fungsi-fungsi helper untuk visualisasi cepat
def setup_visualization():
    """Setup visualisasi secara global"""
    VisualizationHelper.set_plot_style()

__all__ = [
    'VisualizationHelper',
    'DetectionVisualizer',
    'visualize_detection',
    'MetricsVisualizer',
    'plot_confusion_matrix',
    'ResearchVisualizer',
    'visualize_scenario_comparison',
    'ExperimentVisualizer',
    'ScenarioVisualizer',
    'ExperimentAnalyzer',
    'ScenarioAnalyzer',
    'clean_dataframe',
    'format_metric_name',
    'find_common_metrics',
    'create_benchmark_table',
    'create_win_rate_table',
    'setup_visualization',
    'ResultVisualizer'  ,
    'EvaluationVisualizer'
]