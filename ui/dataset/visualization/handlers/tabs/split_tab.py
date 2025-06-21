"""
File: smartcash/ui/dataset/visualization/handlers/tabs/split_tab.py
Deskripsi: Handler untuk tab split dataset
"""

from smartcash.ui.dataset.visualization.components import DatasetSplitVisualizer
from smartcash.common.logger import get_logger

logger = get_logger(__name__)


def create_split_tab(dataset_name: str):
    """Buat tab untuk visualisasi split dataset"""
    # Buat komponen visualizer
    visualizer = DatasetSplitVisualizer(dataset_name)
    
    # Buat tab
    tab = widgets.VBox([
        widgets.HTML(f'<h2>Split Dataset - {dataset_name}</h2>'),
        visualizer.get_ui_components()['main_container']
    ])
    
    return tab
