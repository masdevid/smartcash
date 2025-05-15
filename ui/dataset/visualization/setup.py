"""
File: smartcash/ui/dataset/visualization/setup.py
Deskripsi: Fungsi setup untuk visualisasi dataset yang dipisahkan untuk menghindari konflik
"""

from IPython.display import display, clear_output
import ipywidgets as widgets
import gc
import threading
import time

from smartcash.ui.dataset.visualization.conflict_resolver import check_and_resolve_conflicts

from smartcash.ui.utils.loading_indicator import create_loading_indicator
from smartcash.ui.dataset.visualization.visualization_manager import get_visualization_manager
from smartcash.common.logger import get_logger
from smartcash.ui.utils.constants import ICONS

logger = get_logger(__name__)

# Variabel global untuk menyimpan instance UI yang sedang aktif
_active_ui_components = None
_setup_lock = threading.Lock()

def is_restart_mode():
    """
    Deteksi apakah sedang dalam mode restart Colab.
    
    Returns:
        Boolean yang menunjukkan apakah sedang dalam mode restart
    """
    # Pendekatan yang lebih sederhana dan aman
    try:
        # Cek apakah kita berada di lingkungan Colab
        import google.colab
        
        # Di Colab, gunakan pendekatan alternatif yang lebih aman
        import IPython
        shell = IPython.get_ipython()
        if shell is not None:
            # Cek history input dan output
            history_in = shell.user_ns.get('_ih', [])
            history_out = shell.user_ns.get('_oh', {})
            
            # Jika history input lebih dari 10 dan history output kurang dari 5,
            # kemungkinan besar ini adalah restart
            if len(history_in) > 10 and len(history_out) < 5:
                return True
    except (ImportError, AttributeError, ReferenceError):
        # Bukan di Colab atau error lainnya
        pass
    
    return False

def setup_dataset_visualization(force_new=False):
    """
    Setup visualisasi dataset dengan pendekatan minimalis.
    Semua business logic dipindahkan ke visualization_manager.py.
    
    Args:
        force_new: Jika True, paksa membuat instance baru meskipun sudah ada
        
    Returns:
        Dictionary berisi komponen UI
    """
    global _active_ui_components
    
    # Gunakan lock untuk menghindari race condition
    with _setup_lock:
        # Periksa dan selesaikan konflik UI dengan modul lain
        conflicts_resolved = check_and_resolve_conflicts()
        if conflicts_resolved:
            # Jika konflik diselesaikan, paksa membuat instance baru
            force_new = True
            logger.info("🔄 Konflik UI terdeteksi dan diselesaikan, membuat instance baru")
        
        # Jika sudah ada instance aktif dan tidak dipaksa membuat baru, gunakan yang ada
        if _active_ui_components is not None and not force_new:
            logger.info("🔄 Menggunakan instance visualisasi dataset yang sudah ada")
            return _active_ui_components
        
        # Bersihkan output sebelumnya
        clear_output(wait=True)
        
        # Buat loading indicator
        loading_indicator = create_loading_indicator(
            "Memuat visualisasi dataset...", 
            "Visualisasi dataset berhasil dimuat"
        )
        
        # Bersihkan memori dari instance lama jika ada
        if _active_ui_components is not None:
            # Hapus referensi ke widget lama
            for key, component in _active_ui_components.items():
                if isinstance(component, widgets.Widget):
                    try:
                        component.close()
                    except Exception as e:
                        logger.warning(f"⚠️ Gagal menutup widget: {str(e)}")
            
            # Paksa garbage collection
            _active_ui_components = None
            gc.collect()
            
            # Tunggu sebentar untuk memastikan widget lama benar-benar dihapus
            time.sleep(0.5)
        
        # Cek dan resolve konflik UI jika ada
        has_conflict = check_and_resolve_conflicts()
        
        # Buat loading indicator
        loading_indicator = create_loading_indicator("Mempersiapkan visualisasi dataset...")
        
        try:
            # Dapatkan instance visualization manager
            visualization_manager = get_visualization_manager(loading_indicator)
            
            # Inisialisasi UI components
            ui_components = visualization_manager.initialize()
            
            # Tampilkan UI
            display(ui_components['main_container'])
            
            # Perbarui dashboard dengan data terbaru
            try:
                visualization_manager.update_dashboard()
            except Exception as e:
                logger.error(f"{ICONS.get('error', '❌')} Error saat memperbarui dashboard: {str(e)}")
                # Tampilkan pesan error di UI
                with ui_components['status_panel']:
                    error_html = f"<div style='padding: 10px; background-color: #ffebee; border-radius: 5px;'>"
                    error_html += f"<p><b>{ICONS.get('error', '❌')} Error:</b> {str(e)}</p>"
                    error_html += "</div>"
                    display(widgets.HTML(error_html))
            
            # Setup auto refresh download dengan delay 100ms
            from smartcash.ui.dataset.visualization.auto_refresh import trigger_auto_refresh
            
            def download_dataset_callback():
                try:
                    # Coba dapatkan dataset service
                    from smartcash.dataset.services.service_factory import get_dataset_service
                    dataset_service = get_dataset_service(service_name='downloader')
                    
                    # Jika dataset service tersedia, trigger download
                    if dataset_service and hasattr(dataset_service, 'download_dataset'):
                        logger.info("🔎 Memulai download dataset otomatis...")
                        dataset_service.download_dataset()
                    else:
                        logger.warning("⚠️ Dataset service tidak tersedia untuk download otomatis")
                except Exception as e:
                    logger.error(f"❌ Error saat download dataset otomatis: {str(e)}")
            
            # Trigger auto refresh dengan delay 100ms
            trigger_auto_refresh(download_dataset_callback, delay_ms=100)
            
            # Simpan referensi ke instance aktif
            _active_ui_components = ui_components
            
            return ui_components
        except Exception as e:
            logger.error(f"{ICONS.get('error', '❌')} Error saat setup visualisasi dataset: {str(e)}")
            error_container = widgets.VBox([
                widgets.HTML(f"<div style='color: red; padding: 10px;'><b>Error:</b> {str(e)}</div>")
            ])
            return {'main_container': error_container, 'error': str(e)}

def reset_visualization():
    """
    Reset visualisasi dataset, membuat instance baru.
    
    Returns:
        Dictionary berisi komponen UI baru
    """
    return setup_dataset_visualization(force_new=True)

def get_active_ui_components():
    """
    Dapatkan referensi ke komponen UI yang sedang aktif.
    
    Returns:
        Dictionary berisi komponen UI atau None jika belum diinisialisasi
    """
    return _active_ui_components
