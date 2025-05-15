"""
File: smartcash/ui/dataset/preprocessing/handlers/visualization_handler.py
Deskripsi: Handler visualisasi untuk preprocessing dataset dengan pendekatan SRP
"""

from typing import Dict, Any  # Optional tidak digunakan langsung
# display, clear_output tidak digunakan langsung dalam file ini (digunakan dalam handler yang diimport)
from smartcash.ui.utils.constants import ICONS
# create_status_indicator tidak digunakan langsung dalam file ini
from smartcash.common.logger import get_logger

def setup_visualization_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk visualisasi dataset preprocessing.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    logger = ui_components.get('logger', get_logger('preprocessing_visualization'))
    
    # Import handler terpisah untuk SRP
    from smartcash.ui.dataset.preprocessing.handlers.visualization_sample_handler import setup_sample_visualization_button
    from smartcash.ui.dataset.preprocessing.handlers.visualization_compare_handler import setup_comparison_visualization_button
    
    # Setup tombol visualisasi sampel
    if 'visualize_sample_button' in ui_components:
        setup_sample_visualization_button(ui_components)
        logger.debug(f"{ICONS['info']} Handler visualisasi sampel berhasil disetup")
    
    # Setup tombol visualisasi perbandingan
    if 'visualize_compare_button' in ui_components:
        setup_comparison_visualization_button(ui_components)
        logger.debug(f"{ICONS['info']} Handler visualisasi perbandingan berhasil disetup")
    
    # Untuk kompatibilitas dengan pendekatan lama, jika tombol visualize_button dan compare_button ada
    # tapi tombol visualize_sample_button dan visualize_compare_button tidak ada
    if 'visualize_button' in ui_components and 'visualize_sample_button' not in ui_components:
        ui_components['visualize_sample_button'] = ui_components['visualize_button']
        setup_sample_visualization_button(ui_components)
    
    if 'compare_button' in ui_components and 'visualize_compare_button' not in ui_components:
        ui_components['visualize_compare_button'] = ui_components['compare_button']
        setup_comparison_visualization_button(ui_components)
    
    return ui_components