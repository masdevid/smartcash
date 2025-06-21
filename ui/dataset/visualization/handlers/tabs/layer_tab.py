"""
File: smartcash/ui/dataset/visualization/handlers/tabs/layer_tab.py
Deskripsi: Handler untuk tab distribusi layer
"""

from smartcash.ui.dataset.visualization.components import LayerDistributionVisualizer
from smartcash.common.logger import get_logger

logger = get_logger(__name__)


def create_layer_tab(dataset_name: str):
    """Buat tab untuk visualisasi distribusi layer"""
    # Buat komponen visualizer
    visualizer = LayerDistributionVisualizer(dataset_name)
    
    # Buat tab
    tab = widgets.VBox([
        widgets.HTML(f'<h2>Distribusi Layer - {dataset_name}</h2>'),
        visualizer.get_ui_components()['main_container']
    ])
    
    return tab
