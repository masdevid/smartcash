"""
File: smartcash/utils/visualization/__init__.py
Author: Alfrida Sabar
Deskripsi: File inisialisasi untuk paket visualisasi SmartCash
"""

# Import dari base dan helper
from smartcash.utils.visualization.base import VisualizationHelper

# Import dari visualizer utama
from smartcash.utils.visualization.detection import DetectionVisualizer, visualize_detection
from smartcash.utils.visualization.metrics import MetricsVisualizer, plot_confusion_matrix
from smartcash.utils.visualization.research import ResearchVisualizer, visualize_scenario_comparison
from smartcash.utils.visualization.experiment_visualizer import ExperimentVisualizer
from smartcash.utils.visualization.scenario_visualizer import ScenarioVisualizer
from smartcash.utils.visualization.evaluation_visualizer import EvaluationVisualizer
from smartcash.utils.visualization.result_visualizer import ResultVisualizer

# Import dari utilitas visualisasi
from smartcash.utils.visualization.research_utils import (
    clean_dataframe, format_metric_name, find_common_metrics, 
    create_benchmark_table, create_win_rate_table
)

# Import dari analisis penelitian
from smartcash.utils.visualization.research_analysis import ExperimentAnalyzer, ScenarioAnalyzer

# Fungsi-fungsi helper untuk visualisasi cepat
def setup_visualization():
    """Setup visualisasi secara global."""
    VisualizationHelper.set_plot_style()

__all__ = [
    # Kelas dasar dan utilitas
    'VisualizationHelper',
    'setup_visualization',
    
    # Visualizer utama
    'DetectionVisualizer',
    'visualize_detection',
    'MetricsVisualizer',
    'plot_confusion_matrix',
    'ResearchVisualizer',
    'visualize_scenario_comparison',
    'ExperimentVisualizer',
    'ScenarioVisualizer',
    'EvaluationVisualizer',
    'ResultVisualizer',
    
    # Analisis
    'ExperimentAnalyzer',
    'ScenarioAnalyzer',
    
    # Utilitas
    'clean_dataframe',
    'format_metric_name',
    'find_common_metrics',
    'create_benchmark_table',
    'create_win_rate_table'
]