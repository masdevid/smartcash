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
    Handler untuk tombol download dan sinkronisasi model pretrained dengan UI yang responsif.
    
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
    
    # Reset progress tracking dengan parameter show_progress=True untuk menampilkan progress bar yang bergerak
    if 'reset_progress_bar' in ui_components and callable(ui_components['reset_progress_bar']):
        ui_components['reset_progress_bar'](0, "Inisialisasi proses download model...", show_progress=True)
    
    # Buka log accordion secara otomatis jika tersedia
    if 'log_accordion' in ui_components and hasattr(ui_components['log_accordion'], 'selected_index'):
        ui_components['log_accordion'].selected_index = 0  # Expand log accordion
    
    # Nonaktifkan tombol selama proses berjalan
    b.disabled = True
    b.description = "Sedang Memproses..."
    b.icon = ICONS.get('processing', '⏳')
    
    # Tambahkan observer manager jika tersedia
    observer_manager = ui_components.get('observer_manager')
    if observer_manager and hasattr(observer_manager, 'notify'):
        try:
            observer_manager.notify('MODEL_PROCESS_START', None, {
                'message': "Memulai proses download dan sinkronisasi model",
                'timestamp': __import__('time').time()
            })
        except Exception:
            pass  # Silent fail untuk observer notification
    
    # Jalankan proses download dan sync dalam thread terpisah agar UI tetap responsif
    def run_process():
        try:
            # Jalankan proses download dan sync
            process_download_sync(ui_components)
            
            # Notify complete via observer
            if observer_manager and hasattr(observer_manager, 'notify'):
                try:
                    observer_manager.notify('MODEL_PROCESS_COMPLETE', None, {
                        'message': "Proses download dan sinkronisasi model selesai",
                        'timestamp': __import__('time').time()
                    })
                except Exception:
                    pass  # Silent fail untuk observer notification
        except Exception as e:
            logger.error(f"Error dalam proses download dan sinkronisasi: {str(e)}")
            
            # Notify error via observer
            if observer_manager and hasattr(observer_manager, 'notify'):
                try:
                    observer_manager.notify('MODEL_PROCESS_ERROR', None, {
                        'message': f"Error dalam proses download dan sinkronisasi: {str(e)}",
                        'timestamp': __import__('time').time(),
                        'error': str(e)
                    })
                except Exception:
                    pass  # Silent fail untuk observer notification
        finally:
            # Aktifkan kembali tombol setelah proses selesai
            b.disabled = False
            b.description = "Download & Sync Model"
            b.icon = ICONS.get('download', '📥')
    
    # Mulai thread untuk proses download
    thread = threading.Thread(target=run_process)
    thread.daemon = True
    thread.start()
