"""
File: smartcash/ui/cells/cell_2_5_dataset_visualization.py
Deskripsi: Entry point untuk visualisasi dataset dengan pendekatan minimalis
"""

from IPython.display import display, clear_output
import ipywidgets as widgets
import sys
import os

# Pastikan path modul dapat diakses
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from smartcash.ui.utils.loading_indicator import create_loading_indicator
from smartcash.common.logger import get_logger
from smartcash.ui.dataset.visualization.visualization_initializer_new import initialize_visualization_ui

def setup_dataset_visualization():
    """Setup dan tampilkan UI visualisasi dataset dengan pendekatan minimalis."""
    logger = get_logger(__name__)
    
    # Buat dan tampilkan indikator loading animasi
    loading_indicator = create_loading_indicator(
        message="Mempersiapkan visualisasi dataset", 
        is_indeterminate=True,
        auto_hide=True
    )
    display(loading_indicator)
    
    try:
        # Inisialisasi UI dengan loading indicator
        ui_components = initialize_visualization_ui(loading_indicator=loading_indicator)
        
        # Tampilkan UI
        clear_output(wait=True)
        display(ui_components['main_container'])
        
        return ui_components
    except Exception as e:
        logger.error(f"❌ Error saat setup visualisasi dataset: {str(e)}")
        loading_indicator.error(f"Error: {str(e)}")
        return {'error': str(e)}

# Eksekusi saat modul diimpor
if __name__ == "__main__":
    ui_components = setup_dataset_visualization()
else:
    ui_components = setup_dataset_visualization()
