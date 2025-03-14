"""
File: smartcash/model/visualization/__init__.py
Deskripsi: Modul untuk file __init__.py
"""

# Import dari base dan helper
from smartcash.model.visualization.base_visualizer import VisualizationHelper

# Import dari visualizer utama
from smartcash.model.visualization.metrics_visualizer import MetricsVisualizer, plot_confusion_matrix
from smartcash.model.visualization.research_visualizer import ResearchVisualizer
from smartcash.model.visualization.experiment_visualizer import ExperimentVisualizer
from smartcash.model.visualization.scenario_visualizer import ScenarioVisualizer
from smartcash.model.visualization.evaluation_visualizer import EvaluationVisualizer
from smartcash.model.visualization.detection_visualizer import DetectionVisualizer


# Fungsi-fungsi helper untuk visualisasi cepat
def setup_visualization():
    """Setup visualisasi secara global."""
    VisualizationHelper.set_plot_style()

__all__ = [
    # Kelas dasar dan utilitas
    'VisualizationHelper',
    'setup_visualization',
    
    # Visualizer utama
    'MetricsVisualizer',
    'plot_confusion_matrix',
    'ResearchVisualizer',
    'ExperimentVisualizer',
    'ScenarioVisualizer',
    'EvaluationVisualizer',
    'DetectionVisualizer'
]