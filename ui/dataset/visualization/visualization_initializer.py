"""
File: smartcash/ui/dataset/visualization/visualization_initializer.py
Deskripsi: Inisialisasi komponen UI untuk visualisasi dataset
"""

from typing import Dict, Any
import ipywidgets as widgets
from IPython.display import display

from smartcash.common.logger import get_logger
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.dataset.visualization.visualization_controller import VisualizationController

logger = get_logger(__name__)

def setup_dataset_visualization(dataset_name: str = None) -> Dict[str, Any]:
    """
    Setup dan tampilkan visualisasi dataset.
    
    Args:
        dataset_name: Nama dataset yang akan dimuat (opsional)
        
    Returns:
        Dictionary berisi komponen UI dan controller
    """
    try:
        # Buat instance controller
        controller = VisualizationController()
        
        # Jika nama dataset disediakan, muat dataset
        if dataset_name:
            success = controller.load_dataset(dataset_name)
            if not success:
                logger.warning(f"{ICONS.get('warning', '⚠️')} Gagal memuat dataset: {dataset_name}")
        
        # Dapatkan komponen UI
        ui_components = controller.get_ui_components()
        
        # Tampilkan UI
        if 'main_container' in ui_components:
            display(ui_components['main_container'])
            logger.info(f"{ICONS.get('success', '✅')} UI visualisasi dataset berhasil ditampilkan")
        
        # Tambahkan controller ke output
        ui_components['controller'] = controller
        
        return ui_components
    
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat setup visualisasi dataset: {str(e)}")
        
        # Tampilkan pesan error
        error_widget = widgets.HTML(
            f"""<div style="color: red; padding: 10px; border-left: 5px solid red;">
                <h3>Error saat setup visualisasi dataset</h3>
                <p>{str(e)}</p>
            </div>"""
        )
        display(error_widget)
        
        return {} 