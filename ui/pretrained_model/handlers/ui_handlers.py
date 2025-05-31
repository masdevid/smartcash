"""
File: smartcash/ui/pretrained_model/handlers/ui_handlers.py
Deskripsi: Handler UI untuk komponen model pretrained
"""

from typing import Dict, Any
from IPython.display import display, HTML
import threading

from smartcash.ui.utils.constants import ICONS, COLORS
from smartcash.common.logger import get_logger
from smartcash.ui.pretrained_model.services.process_orchestrator import process_download_sync

logger = get_logger(__name__)

def handle_download_sync_button(b, ui_components: Dict[str, Any]) -> None:
    """
    Handler sederhana untuk tombol download dan sinkronisasi model pretrained.
    
    Args:
        b: Button widget yang dipicu
        ui_components: Dictionary berisi komponen UI
    """
    # Pastikan status_panel tersedia
    status_panel = ui_components.get('status')
    if not status_panel:
        logger.warning(f"{ICONS.get('warning', '⚠️')} Status panel tidak tersedia")
        return
    
    # Tampilkan status processing dengan emoji kontekstual
    status_panel.clear_output(wait=True)
    with status_panel:
        display(HTML(f"""<div style="padding:10px; background-color:{COLORS['alert_info_bg']}; 
                 color:{COLORS['alert_info_text']}; border-radius:4px; margin:5px 0;
                 border-left:4px solid {COLORS['alert_info_text']}">
            <p style="margin:5px 0">{ICONS.get('processing', '⏳')} Memeriksa dan memproses model pretrained...</p>
        </div>"""))
    
    # Reset progress tracking jika tersedia - one-liner style
    if 'reset_progress_bar' in ui_components and callable(ui_components['reset_progress_bar']):
        ui_components['reset_progress_bar'](0, "Bersiap untuk download dan sinkronisasi model...")
    
    # Buka log accordion secara otomatis jika tersedia - one-liner style
    if 'log_accordion' in ui_components and hasattr(ui_components['log_accordion'], 'selected_index'):
        ui_components['log_accordion'].selected_index = 0  # Expand log accordion
    
    # Nonaktifkan tombol selama proses berjalan
    b.disabled = True
    b.description = "Sedang Memproses..."
    
    # Jalankan proses download dan sync dalam thread terpisah agar UI tetap responsif
    def run_process():
        try:
            process_download_sync(ui_components)
        finally:
            # Aktifkan kembali tombol setelah proses selesai
            b.disabled = False
            b.description = "Download & Sync Model"
    
    # Mulai thread untuk proses download
    threading.Thread(target=run_process).start()
