"""
File: smartcash/ui/dataset/visualization/handlers/setup_handlers.py
Deskripsi: Setup semua handler untuk visualisasi dataset
"""

from typing import Dict, Any

from smartcash.common.logger import get_logger
from smartcash.ui.utils.constants import ICONS

from smartcash.ui.dataset.visualization.handlers.dashboard_handlers import setup_dashboard_handlers
from smartcash.ui.dataset.visualization.handlers.bbox_handlers import setup_bbox_handlers
from smartcash.ui.dataset.visualization.handlers.layer_handlers import setup_layer_handlers
from smartcash.ui.dataset.visualization.handlers.tabs import (
    on_distribution_click,
    on_split_click,
    on_layer_click,
    on_bbox_click,
    on_preprocessing_samples_click,
    on_augmentation_comparison_click
)

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
        
        # Setup handler untuk visualisasi bounding box
        ui_components = setup_bbox_handlers(ui_components)
        
        # Setup handler untuk visualisasi layer
        ui_components = setup_layer_handlers(ui_components)
        
        # Setup handlers untuk semua tab visualisasi
        visualization_components = ui_components.get('visualization_components', {})
        
        # Setup handlers untuk tab distribusi kelas
        if 'distribution_tab' in visualization_components:
            distribution_tab = visualization_components['distribution_tab']
            if 'button' in distribution_tab:
                distribution_tab['button'].on_click(
                    lambda b: on_distribution_click(b, ui_components)
                )
        
        # Setup handlers untuk tab distribusi split
        if 'split_tab' in visualization_components:
            split_tab = visualization_components['split_tab']
            if 'button' in split_tab:
                split_tab['button'].on_click(
                    lambda b: on_split_click(b, ui_components)
                )
        
        # Setup handlers untuk tab distribusi layer
        if 'layer_tab' in visualization_components:
            layer_tab = visualization_components['layer_tab']
            if 'button' in layer_tab:
                layer_tab['button'].on_click(
                    lambda b: on_layer_click(b, ui_components)
                )
        
        # Setup handlers untuk tab analisis bounding box
        if 'bbox_tab' in visualization_components:
            bbox_tab = visualization_components['bbox_tab']
            if 'button' in bbox_tab:
                bbox_tab['button'].on_click(
                    lambda b: on_bbox_click(b, ui_components)
                )
        
        # Setup handlers untuk tab sampel preprocessing
        if 'preprocessing_samples_tab' in visualization_components:
            preprocessing_samples_tab = visualization_components['preprocessing_samples_tab']
            if 'button' in preprocessing_samples_tab:
                preprocessing_samples_tab['button'].on_click(
                    lambda b: on_preprocessing_samples_click(b, ui_components)
                )
        
        # Setup handlers untuk tab perbandingan augmentasi
        if 'augmentation_comparison_tab' in visualization_components:
            augmentation_comparison_tab = visualization_components['augmentation_comparison_tab']
            if 'button' in augmentation_comparison_tab:
                augmentation_comparison_tab['button'].on_click(
                    lambda b: on_augmentation_comparison_click(b, ui_components)
                )
        
        logger.info(f"{ICONS.get('success', '✅')} Semua handler visualisasi berhasil disetup")
        return ui_components
        
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat setup handler visualisasi: {str(e)}")
        return ui_components 