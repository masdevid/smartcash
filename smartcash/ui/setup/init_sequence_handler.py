"""
File: smartcash/ui/setup/init_sequence_handler.py
Deskripsi: Pengendali urutan inisialisasi untuk memastikan logger UI diinisialisasi sebelum sinkronisasi drive
"""

import sys
import time
from typing import Dict, Any, Optional, Callable

def setup_inisialisasi_berurutan(ui_components: Dict[str, Any]) -> None:
    """
    Siapkan interceptor stdout dan inisialisasi komponen dalam urutan yang benar.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    logger = ui_components.get('logger')
    
    # 1. Pasang interceptor stdout terlebih dahulu
    try:
        from smartcash.ui.utils.ui_logger import intercept_stdout_to_ui
        intercept_stdout_to_ui(ui_components)
        
        if logger:
            logger.debug("🔧 Interceptor stdout terpasang")
    except ImportError:
        pass
    
    # 2. Tunda sinkronisasi drive sampai diinisialisasi secara eksplisit
    _disable_auto_sync()
    
    # 3. Tambahkan tombol untuk memulai sinkronisasi
    _tambahkan_tombol_sync(ui_components)

def _disable_auto_sync() -> None:
    """Nonaktifkan sinkronisasi otomatis pada import."""
    try:
        # Cara 1: Set flag initialized di drive_sync_initializer
        from smartcash.ui.setup.drive_sync_initializer import _initialized
        globals()['_initialized'] = True
    except (ImportError, AttributeError):
        pass
    
    try:
        # Cara 2: Buat modul tiruan untuk mencegah sinkronisasi
        import sys
        class NoOpModule:
            def __getattr__(self, name): return lambda *args, **kwargs: None
        
        # Pasang module palsu jika belum diimport
        if 'smartcash.common.drive_sync_initializer' not in sys.modules:
            sys.modules['smartcash.common.drive_sync_initializer'] = NoOpModule()
    except Exception:
        pass

def _tambahkan_tombol_sync(ui_components: Dict[str, Any]) -> None:
    """Tambahkan tombol sinkronisasi drive ke UI."""
    import ipywidgets as widgets
    
    # Skip jika tombol sudah ada
    if 'sync_button' in ui_components:
        return
    
    # Buat tombol sinkronisasi
    sync_button = widgets.Button(
        description='Sinkronisasi Drive',
        button_style='info',
        icon='sync',
        layout=widgets.Layout(margin='5px')
    )
    
    # Handler untuk tombol
    def on_sync_clicked(b):
        # Nonaktifkan tombol selama proses
        b.disabled = True
        
        # Tampilkan status
        from smartcash.ui.utils.ui_logger import log_to_ui
        log_to_ui(ui_components, "Memulai sinkronisasi konfigurasi dengan Drive...", "info", "🔄")
        
        # Jalankan sinkronisasi
        try:
            from smartcash.ui.setup.drive_sync_initializer import initialize_configs
            logger = ui_components.get('logger')
            # Reset flag initialized
            try:
                import smartcash.ui.setup.drive_sync_initializer
                smartcash.ui.setup.drive_sync_initializer._initialized = False
            except:
                pass
                
            # Jalankan sinkronisasi
            success, message = initialize_configs(logger)
            log_to_ui(ui_components, f"Sinkronisasi: {message}", "success" if success else "warning")
        except Exception as e:
            # Tampilkan error
            log_to_ui(ui_components, f"Error saat sinkronisasi: {str(e)}", "error", "❌")
        
        # Aktifkan tombol kembali setelah selesai
        b.disabled = False
    
    # Pasang handler
    sync_button.on_click(on_sync_clicked)
    
    # Tambahkan tombol ke UI
    if 'button_container' in ui_components:
        # Tambahkan ke container tombol yang sudah ada
        children_list = list(ui_components['button_container'].children)
        children_list.append(sync_button)
        ui_components['button_container'].children = tuple(children_list)
    else:
        # Buat container baru
        button_container = widgets.HBox([sync_button], 
                                      layout=widgets.Layout(
                                          display='flex',
                                          flex_flow='row',
                                          align_items='center',
                                          width='100%'
                                      ))
        
        # Cari posisi untuk menyisipkan container
        if 'ui' in ui_components and hasattr(ui_components['ui'], 'children'):
            # Temukan posisi yang tepat untuk menyisipkan (setelah header)
            children = list(ui_components['ui'].children)
            insert_pos = min(1, len(children))
            
            # Sisipkan container
            children.insert(insert_pos, button_container)
            ui_components['ui'].children = tuple(children)
        
        # Simpan referensi
        ui_components['button_container'] = button_container
    
    # Simpan referensi ke tombol
    ui_components['sync_button'] = sync_button

def jalankan_sinkronisasi_tertunda(ui_components: Dict[str, Any], delay_seconds: int = 3) -> None:
    """
    Jalankan sinkronisasi tertunda setelah menunggu beberapa detik.
    
    Args:
        ui_components: Dictionary komponen UI
        delay_seconds: Jumlah detik untuk menunggu
    """
    import threading
    
    def _sinkronisasi_tertunda():
        # Tunggu beberapa detik
        time.sleep(delay_seconds)
        
        # Jalankan handler tombol sinkronisasi secara manual
        if 'sync_button' in ui_components:
            ui_components['sync_button'].click()
    
    # Jalankan dalam thread terpisah
    thread = threading.Thread(target=_sinkronisasi_tertunda)
    thread.daemon = True
    thread.start()