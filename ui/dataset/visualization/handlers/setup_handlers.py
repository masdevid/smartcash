"""
File: smartcash/ui/dataset/visualization/handlers/setup_handlers.py
Deskripsi: Setup semua handler untuk visualisasi dataset
"""

from typing import Dict, Any

from smartcash.common.logger import get_logger
from smartcash.ui.utils.constants import ICONS

from smartcash.ui.dataset.visualization.handlers.dashboard_handlers import setup_dashboard_handlers
from smartcash.ui.dataset.visualization.handlers.distribution_handlers import setup_distribution_handlers
from smartcash.ui.dataset.visualization.handlers.split_handlers import setup_split_handlers
from smartcash.ui.dataset.visualization.handlers.advanced_visualization_handlers import setup_advanced_visualization_handlers
from smartcash.ui.dataset.visualization.handlers.bbox_handlers import setup_bbox_handlers
from smartcash.ui.dataset.visualization.handlers.layer_handlers import setup_layer_handlers

logger = get_logger(__name__)

def setup_visualization_handlers(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Setup semua handler untuk visualisasi dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary komponen UI yang telah diupdate dengan semua handler
    """
    try:
        # Setup handler untuk dashboard visualisasi
        ui_components = setup_dashboard_handlers(ui_components)
        
        # Setup handler untuk distribusi kelas
        ui_components = setup_distribution_handlers(ui_components)
        
        # Setup handler untuk distribusi split
        ui_components = setup_split_handlers(ui_components)
        
        # Setup handler untuk visualisasi lanjutan
        ui_components = setup_advanced_visualization_handlers(ui_components)
        
        # Setup handler untuk visualisasi bounding box
        ui_components = setup_bbox_handlers(ui_components)
        
        # Setup handler untuk visualisasi layer
        ui_components = setup_layer_handlers(ui_components)
        
        logger.info(f"{ICONS.get('success', '✅')} Semua handler visualisasi berhasil disetup")
        return ui_components
        
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat setup handler visualisasi: {str(e)}")
        return ui_components 