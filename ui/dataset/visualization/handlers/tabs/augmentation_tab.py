"""
File: smartcash/ui/dataset/visualization/handlers/tabs/augmentation_tab.py
Deskripsi: Handler untuk tab augmentasi
"""

from smartcash.ui.dataset.visualization.components import AugmentationVisualizer
from smartcash.common.logger import get_logger

logger = get_logger(__name__)


def create_augmentation_tab(dataset_name: str):
    """Buat tab untuk visualisasi augmentasi"""
    # Buat komponen visualizer
    visualizer = AugmentationVisualizer(dataset_name)
    
    # Buat tab
    tab = widgets.VBox([
        widgets.HTML(f'<h2>Visualisasi Augmentasi - {dataset_name}</h2>'),
        visualizer.get_ui_components()['main_container']
    ])
    
    return tab
