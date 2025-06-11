"""
File: smartcash/ui/dataset/visualization/handlers/tabs/distribution_tab.py
Deskripsi: Handler untuk tab distribusi kelas
"""

from smartcash.ui.dataset.visualization.components import ClassDistributionVisualizer
from smartcash.common.logger import get_logger

logger = get_logger(__name__)


def create_distribution_tab(dataset_name: str):
    """Buat tab untuk visualisasi distribusi kelas"""
    # Buat komponen visualizer
    visualizer = ClassDistributionVisualizer(dataset_name)
    
    # Buat tab
    tab = widgets.VBox([
        widgets.HTML(f'<h2>Distribusi Kelas - {dataset_name}</h2>'),
        visualizer.get_ui_components()['main_container']
    ])
    
    return tab
