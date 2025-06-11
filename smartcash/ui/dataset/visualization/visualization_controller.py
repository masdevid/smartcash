"""
File: smartcash/ui/dataset/visualization/visualization_controller.py
Deskripsi: Controller utama untuk visualisasi dataset
"""

from typing import Dict, Any, Optional
from smartcash.common.logger import get_logger
from typing import Optional, Dict

logger = get_logger(__name__)

class VisualizationController:
    """Controller utama untuk visualisasi dataset"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Inisialisasi controller visualisasi
        
        Args:
            config: Konfigurasi visualisasi
        """
        self.config = config or {}
        self.ui_components = {}
        self.current_dataset = None
        self.dataset_stats = {}
        
        # Inisialisasi handler visualisasi
        from smartcash.ui.dataset.visualization.handlers.visualization_handler import DatasetVisualizationHandler
        self.handler = DatasetVisualizationHandler(self.config)
        
    def load_dataset(self, dataset_name: str):
        """Memuat dataset dan statistik terkait"""
        self.current_dataset = dataset_name
        
        # Gunakan data dummy
        self.dataset_stats = {
            "class_distribution": {
                "train": {"class1": 100, "class2": 200},
                "val": {"class1": 30, "class2": 40}
            },
            "image_sizes": {
                "train": [(640, 480), (800, 600)],
                "val": [(640, 480), (800, 600)]
            }
        }

    def get_ui_components(self) -> Dict[str, Any]:
        return self.handler.get_ui_components()
