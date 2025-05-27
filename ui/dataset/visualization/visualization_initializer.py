"""
File: smartcash/ui/dataset/visualization/setup.py
Deskripsi: Setup komponen UI untuk visualisasi dataset
"""

from typing import Dict, Any
import ipywidgets as widgets
from IPython.display import display

from smartcash.common.logger import get_logger
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.dataset.visualization.components.main_layout import create_visualization_layout

logger = get_logger(__name__)

def setup_dataset_visualization() -> Dict[str, Any]:
    """
    Setup visualisasi dataset.
    
    Returns:
        Dictionary berisi komponen UI
    """
    try:
        # Buat komponen UI
        ui_components = create_visualization_layout()
        
        # Tampilkan UI
        if 'main_container' in ui_components:
            display(ui_components['main_container'])
            logger.info(f"{ICONS.get('success', '✅')} UI visualisasi dataset berhasil ditampilkan")
        
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