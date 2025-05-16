"""
File: smartcash/ui/dataset/augmentation/visualization/visualization_initializer.py
Deskripsi: Initializer untuk visualisasi augmentasi dataset
"""

import os
from typing import Dict, Any, Optional
from IPython.display import display

from smartcash.common.logger import get_logger
from smartcash.common.config.manager import get_config_manager
from smartcash.ui.dataset.augmentation.visualization.visualization_manager import AugmentationVisualizationManager
from smartcash.ui.utils.header_utils import create_header


def initialize_augmentation_visualization(config: Dict[str, Any] = None) -> None:
    """
    Inisialisasi dan tampilkan UI visualisasi augmentasi.
    
    Args:
        config: Konfigurasi aplikasi (opsional)
    """
    logger = get_logger("augmentation_visualization_init")
    
    # Tampilkan header
    header = create_header(
        title="Visualisasi Augmentasi Dataset 🎨",
        description="Visualisasi sampel hasil augmentasi dan perbandingan dengan preprocessing"
    )
    display(header)
    
    # Dapatkan konfigurasi
    if config is None:
        config_manager = get_config_manager()
        config = config_manager.get_module_config('augmentation')
    
    # Inisialisasi manager visualisasi
    try:
        visualization_manager = AugmentationVisualizationManager.get_instance(config, logger)
        
        # Tampilkan UI visualisasi
        visualization_manager.display_visualization_ui()
        
        logger.info("✅ Berhasil menginisialisasi visualisasi augmentasi")
    except Exception as e:
        logger.error(f"❌ Gagal menginisialisasi visualisasi augmentasi: {str(e)}")
        raise
