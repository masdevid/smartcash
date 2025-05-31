"""
File: smartcash/ui/pretrained_model/components/pretrained_components.py
Deskripsi: Komponen UI untuk pretrained model dengan pendekatan DRY
"""

import ipywidgets as widgets
from typing import Dict, Any
from IPython.display import display, clear_output

from smartcash.ui.utils.constants import ICONS, COLORS
from smartcash.ui.utils.header_utils import create_header
from smartcash.ui.components.progress_tracking import create_progress_tracking
from smartcash.ui.components.log_accordion import create_log_accordion
from smartcash.common.logger import get_logger

logger = get_logger(__name__)

def create_pretrained_ui() -> Dict[str, Any]:
    """
    Membuat komponen UI untuk pretrained model.
    
    Returns:
        Dictionary berisi komponen UI
    """
    try:
        # Buat komponen UI dengan progress tracking dan log accordion
        main_container = widgets.VBox()
        status_panel = widgets.Output()
        
        # Buat progress tracking dengan style yang konsisten
        progress_tracking = create_progress_tracking()
        
        # Buat log accordion untuk menampilkan log dengan lebih terorganisir
        log_accordion = create_log_accordion(module_name="pretrained_model", height="200px")
        log_output = log_accordion['log_output']
        
        # Buat tombol download & sync dengan layout di tengah
        download_sync_button = widgets.Button(
            description="Download & Sync Model",
            button_style='primary',
            icon='download',
            tooltip='Download dan sinkronisasi model pretrained',
            layout=widgets.Layout(
                margin='15px auto',  # Auto margin untuk centering
                display='flex'
            )
        )
        
        # Buat header dengan create_header
        header = create_header(
            title="Persiapan Model Pre-trained",
            description="Download dan sinkronisasi model pre-trained YOLOv5 dan EfficientNet-B4",
            icon=ICONS.get('brain', '🧠')
        )
        
        # Buat info panel
        info_html = widgets.HTML(
            f"""<div style="padding:10px; background-color:{COLORS['alert_info_bg']}; 
                 color:{COLORS['alert_info_text']}; border-radius:4px; margin:5px 0;
                 border-left:4px solid {COLORS['alert_info_text']}">
                <p style="margin:5px 0">{ICONS.get('info', 'ℹ️')} Klik tombol <b>Download & Sync Model</b> untuk memeriksa, 
                mengunduh, dan menyinkronkan model pre-trained yang diperlukan.</p>
                <p style="margin:5px 0">Model yang akan diunduh:</p>
                <ul>
                    <li>YOLOv5s (14 MB)</li>
                    <li>EfficientNet-B4 (75 MB)</li>
                </ul>
                <p style="margin:5px 0">Jika model sudah ada di Google Drive, akan langsung disinkronkan tanpa mengunduh ulang.</p>
            </div>"""
        )
        
        # Buat container utama dengan semua komponen
        main_container = widgets.VBox([
            header, 
            info_html,
            download_sync_button,
            status_panel,
            widgets.VBox([progress_tracking['container']], 
                        layout=widgets.Layout(margin='10px 0')),
            widgets.VBox([log_accordion['log_accordion']], 
                        layout=widgets.Layout(margin='10px 0'))
        ], layout=widgets.Layout(padding='10px'))
        
        # Definisikan direktori untuk model
        models_dir = '/content/models'
        drive_models_dir = '/content/drive/MyDrive/SmartCash/models'
        
        # Kumpulkan komponen UI dengan progress tracking dan log accordion
        ui_components = {
            'main_container': main_container,
            'status': status_panel,
            'log': log_output,
            'log_accordion': log_accordion['log_accordion'],
            'download_sync_button': download_sync_button,
            'models_dir': models_dir,
            'drive_models_dir': drive_models_dir,
            # Tambahkan alias untuk kompatibilitas dengan kode lama
            'progress_bar': progress_tracking['tracker'],
            'progress_label': progress_tracking['status_widget']
        }
        
        # Tambahkan komponen progress tracking dan log accordion lengkap
        ui_components.update(progress_tracking)
        ui_components.update(log_accordion)
        
        return ui_components
        
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat membuat UI pretrained model: {str(e)}")
        
        # Buat container minimal untuk menampilkan error
        error_container = widgets.VBox([
            widgets.HTML(f"<h3>{ICONS.get('error', '❌')} Error saat membuat UI pretrained model</h3>"),
            widgets.HTML(f"<p>{str(e)}</p>")
        ])
        
        display(error_container)
        
        return {'main_container': error_container}
