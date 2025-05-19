"""
File: smartcash/ui/dataset/augmentation/augmentation_initializer.py
Deskripsi: Initializer untuk modul augmentasi dataset
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from IPython.display import display
from smartcash.ui.utils.constants import ICONS
from smartcash.common.logger import get_logger

logger = get_logger("augmentation_initializer")

def initialize_augmentation(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inisialisasi modul augmentasi dataset.

    Args:
        ui_components: Dictionary komponen UI

    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Tambahkan logger ke ui_components jika belum ada
    if 'logger' not in ui_components:
        ui_components['logger'] = logger

    # Buat container untuk UI augmentasi
    ui_components['augmentation_container'] = widgets.VBox([
        widgets.HTML(value="<h2>Augmentasi Dataset</h2>"),
        widgets.HTML(value="<p>Modul ini memungkinkan Anda untuk melakukan augmentasi pada dataset yang telah dipreprocessing.</p>")
    ])

    # Tambahkan status panel
    status_panel = widgets.HTML(
        value=f"<div style='padding:10px; background-color:#e8f4f8; border-left:4px solid #5bc0de; margin:10px 0;'>{ICONS['info']} Siap untuk augmentasi dataset</div>",
        layout=widgets.Layout(margin='10px 0')
    )
    ui_components['status_panel'] = status_panel

    # Tambahkan status panel ke container
    ui_components['augmentation_container'].children += (status_panel,)

    # Tambahkan output untuk status dan log
    ui_components['status'] = widgets.Output(layout=widgets.Layout(margin='10px 0'))
    ui_components['log_output'] = widgets.Output(layout=widgets.Layout(margin='10px 0', max_height='200px', overflow='auto'))

    # Tambahkan output ke container
    ui_components['augmentation_container'].children += (ui_components['status'],)

    # Setup handlers
    try:
        from smartcash.ui.dataset.augmentation.handlers.setup_handlers import setup_augmentation_handlers
        ui_components = setup_augmentation_handlers(ui_components)
        logger.info(f"{ICONS['success']} Modul augmentasi berhasil diinisialisasi")
    except Exception as e:
        logger.error(f"{ICONS['error']} Error saat setup handlers: {str(e)}")
        with ui_components['status']:
            display(widgets.HTML(
                value=f"<div style='padding:10px; background-color:#f2dede; border-left:4px solid #d9534f; margin:10px 0;'>{ICONS['error']} Error saat setup handlers: {str(e)}</div>"
            ))

    # Tambahkan tombol-tombol ke container
    if 'button_container' in ui_components:
        ui_components['augmentation_container'].children += (ui_components['button_container'],)

    # Tambahkan accordion options ke container
    if 'options_accordion' in ui_components:
        ui_components['augmentation_container'].children += (ui_components['options_accordion'],)

    # Tambahkan progress bar ke container
    if 'progress_container' in ui_components:
        ui_components['augmentation_container'].children += (ui_components['progress_container'],)

    # Tambahkan summary container ke container
    if 'summary_container' in ui_components:
        ui_components['augmentation_container'].children += (ui_components['summary_container'],)

    # Tampilkan container
    display(ui_components['augmentation_container'])

    return ui_components
