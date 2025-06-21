"""
File: smartcash/ui/dataset/visualization/__init__.py
Deskripsi: Modul visualisasi dataset untuk SmartCash

Modul ini menyediakan antarmuka visual untuk menganalisis dataset, menampilkan statistik,
dan memvisualisasikan hasil augmentasi data.
"""

from .visualization_controller import VisualizationController
from .visualization_initializer import setup_dataset_visualization

# Ekspor utama
__all__ = ['VisualizationController', 'setup_dataset_visualization']

# Buat instance controller default
_controller = None

def show_visualization():
    """Tampilkan antarmuka visualisasi dataset
    
    Returns:
        VisualizationController: Instance controller visualisasi
    """
    global _controller
    if _controller is None:
        _controller = VisualizationController()
    
    _controller.display()
    return _controller
