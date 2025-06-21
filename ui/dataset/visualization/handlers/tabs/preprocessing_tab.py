"""
File: smartcash/ui/dataset/visualization/handlers/tabs/preprocessing_tab.py
Deskripsi: Handler untuk tab preprocessing
"""

from smartcash.ui.dataset.visualization.components import PreprocessingStatsVisualizer
from smartcash.common.logger import get_logger

logger = get_logger(__name__)


def create_preprocessing_tab(dataset_name: str):
    """Buat tab untuk visualisasi preprocessing"""
    # Buat komponen visualizer
    visualizer = PreprocessingStatsVisualizer(dataset_name)
    
    # Buat tab
    tab = widgets.VBox([
        widgets.HTML(f'<h2>Statistik Preprocessing - {dataset_name}</h2>'),
        visualizer.get_ui_components()['main_container']
    ])
    
    return tab
