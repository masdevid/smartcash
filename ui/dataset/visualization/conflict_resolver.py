"""
File: smartcash/ui/dataset/visualization/conflict_resolver.py
Deskripsi: Resolver konflik untuk mengatasi masalah antarmuka yang tercampur saat Colab restart
"""

import gc
import threading
import time
from IPython.display import clear_output
import ipywidgets as widgets

from smartcash.common.logger import get_logger

logger = get_logger(__name__)

# Daftar nama modul yang mungkin konflik
POTENTIAL_CONFLICT_MODULES = [
    'smartcash.ui.pretrained',
    'smartcash.ui.cells.cell_2_4_pretrained_model',
    'smartcash.ui.pretrained.setup'
]

# Lock untuk operasi resolusi konflik
_resolver_lock = threading.Lock()

def detect_ui_conflicts():
    """
    Deteksi kemungkinan konflik UI dengan modul lain.
    
    Returns:
        Tuple (has_conflict, conflict_widgets)
    """
    conflict_widgets = []
    
    # Cari semua widget yang mungkin dari modul lain
    for obj in gc.get_objects():
        if isinstance(obj, widgets.Widget):
            # Periksa apakah widget berasal dari modul yang berpotensi konflik
            widget_module = getattr(obj, '__module__', '')
            for conflict_module in POTENTIAL_CONFLICT_MODULES:
                if conflict_module in widget_module:
                    conflict_widgets.append(obj)
                    break
    
    return len(conflict_widgets) > 0, conflict_widgets

def resolve_ui_conflicts():
    """
    Resolve konflik UI dengan menutup widget dari modul lain.
    
    Returns:
        Boolean yang menunjukkan apakah konflik berhasil diselesaikan
    """
    with _resolver_lock:
        has_conflict, conflict_widgets = detect_ui_conflicts()
        
        if not has_conflict:
            return False
        
        # Log informasi konflik
        logger.warning(f"🔄 Terdeteksi {len(conflict_widgets)} widget yang berpotensi konflik")
        
        # Tutup semua widget yang konflik
        for widget in conflict_widgets:
            try:
                widget.close()
            except Exception as e:
                logger.error(f"❌ Gagal menutup widget: {str(e)}")
        
        # Paksa garbage collection
        gc.collect()
        
        # Tunggu sebentar untuk memastikan widget benar-benar dihapus
        time.sleep(0.5)
        
        # Bersihkan output
        clear_output(wait=True)
        
        # Periksa lagi apakah masih ada konflik
        has_conflict, remaining_widgets = detect_ui_conflicts()
        if has_conflict:
            logger.warning(f"⚠️ Masih tersisa {len(remaining_widgets)} widget konflik")
            return False
        
        logger.info("✅ Konflik UI berhasil diselesaikan")
        return True

def check_and_resolve_conflicts():
    """
    Periksa dan resolve konflik UI jika ditemukan.
    
    Returns:
        Boolean yang menunjukkan apakah konflik berhasil diselesaikan
    """
    has_conflict, _ = detect_ui_conflicts()
    
    if has_conflict:
        return resolve_ui_conflicts()
    
    return False
