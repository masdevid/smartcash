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
    import gc
    for obj in gc.get_objects():
        if isinstance(obj, dict) and '_oh' in obj and '_ih' in obj:
            # Jika history input lebih dari 10 dan history output kurang dari 5,
            # kemungkinan besar ini adalah restart
            if len(obj['_ih']) > 10 and len(obj['_oh']) < 5:
                return True
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
        
        # Dapatkan visualization manager
        visualization_manager = get_visualization_manager(loading_indicator)
        
        # Inisialisasi UI dan tampilkan
        ui_components = visualization_manager.initialize()
        display(ui_components['main_container'])
        
        # Simpan referensi ke instance aktif
        _active_ui_components = ui_components
        
        return ui_components

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
