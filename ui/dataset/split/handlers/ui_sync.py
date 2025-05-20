"""
File: smartcash/ui/dataset/split/handlers/ui_sync.py
Deskripsi: Utilitas untuk menyinkronkan UI dengan konfigurasi yang diupdate di Google Drive
"""

from typing import Dict, Any, Optional, Callable
import time
import threading
from smartcash.common.logger import get_logger
from smartcash.ui.dataset.split.handlers.config_handlers import (
    load_config, update_ui_from_config, is_colab_environment
)
from smartcash.ui.dataset.split.handlers.sync_logger import (
    log_sync_info, log_sync_success, log_sync_warning, log_sync_error
)

logger = get_logger(__name__)

class UISyncManager:
    """Manager untuk sinkronisasi UI dengan konfigurasi dari Google Drive."""
    
    def __init__(self, ui_components: Dict[str, Any], update_interval: int = 10):
        """
        Inisialisasi UI Sync Manager.
        
        Args:
            ui_components: Dictionary komponen UI
            update_interval: Interval update dalam detik
        """
        self.ui_components = ui_components
        self.update_interval = update_interval
        self.stop_flag = threading.Event()
        self.sync_thread = None
        self.last_config = None
        self.is_running = False
    
    def start(self):
        """Mulai sinkronisasi UI dengan konfigurasi dari Google Drive."""
        if self.is_running:
            log_sync_info(self.ui_components, "Sinkronisasi UI sudah berjalan")
            return
        
        # Periksa apakah di lingkungan Colab
        if not is_colab_environment():
            log_sync_info(self.ui_components, "Sinkronisasi UI tidak diperlukan (bukan di Google Colab)")
            return
        
        # Reset stop flag
        self.stop_flag.clear()
        
        # Simpan konfigurasi terakhir
        self.last_config = load_config()
        
        # Buat thread untuk sinkronisasi
        self.sync_thread = threading.Thread(target=self._sync_task, daemon=True)
        self.sync_thread.start()
        self.is_running = True
        
        log_sync_success(self.ui_components, f"Sinkronisasi UI dimulai (interval: {self.update_interval}s)")
    
    def stop(self):
        """Hentikan sinkronisasi UI."""
        if not self.is_running:
            return
            
        self.stop_flag.set()
        if self.sync_thread and self.sync_thread.is_alive():
            self.sync_thread.join(timeout=2.0)
        
        self.is_running = False
        log_sync_info(self.ui_components, "Sinkronisasi UI dihentikan")
    
    def _sync_task(self):
        """Task untuk sinkronisasi UI dengan konfigurasi dari Google Drive."""
        log_sync_info(self.ui_components, "Memulai task sinkronisasi UI")
        
        while not self.stop_flag.is_set():
            try:
                # Load konfigurasi terbaru
                current_config = load_config()
                
                # Periksa apakah konfigurasi berubah
                if self._is_config_changed(current_config):
                    log_sync_info(self.ui_components, "Terdeteksi perubahan konfigurasi, mengupdate UI...")
                    
                    # Update UI dari konfigurasi
                    update_ui_from_config(self.ui_components, current_config)
                    
                    # Perbarui konfigurasi terakhir
                    self.last_config = current_config
                    
                    log_sync_success(self.ui_components, "UI berhasil diupdate dari konfigurasi yang diubah di Google Drive")
            except Exception as e:
                log_sync_error(self.ui_components, f"Error saat sinkronisasi UI: {str(e)}")
            
            # Tunggu interval waktu
            time.sleep(self.update_interval)
    
    def _is_config_changed(self, current_config: Dict[str, Any]) -> bool:
        """
        Periksa apakah konfigurasi berubah.
        
        Args:
            current_config: Konfigurasi saat ini
            
        Returns:
            Boolean yang menunjukkan apakah konfigurasi berubah
        """
        if not self.last_config or not current_config:
            return True
        
        # Periksa apakah struktur sama
        if 'split' not in self.last_config or 'split' not in current_config:
            return True
        
        # Periksa apakah nilai berubah
        last_split = self.last_config['split']
        current_split = current_config['split']
        
        # Periksa setiap key
        for key in ['enabled', 'train_ratio', 'val_ratio', 'test_ratio', 'random_seed', 'stratify']:
            if key not in last_split or key not in current_split:
                return True
            if last_split[key] != current_split[key]:
                return True
        
        return False
    
    def force_sync(self):
        """Paksa sinkronisasi UI dengan konfigurasi dari Google Drive."""
        try:
            # Load konfigurasi terbaru
            current_config = load_config()
            
            # Update UI dari konfigurasi
            update_ui_from_config(self.ui_components, current_config)
            
            # Perbarui konfigurasi terakhir
            self.last_config = current_config
            
            log_sync_success(self.ui_components, "UI berhasil disinkronkan dengan konfigurasi")
            return True
        except Exception as e:
            log_sync_error(self.ui_components, f"Error saat memaksa sinkronisasi UI: {str(e)}")
            return False

def setup_ui_sync(ui_components: Dict[str, Any], update_interval: int = 10) -> UISyncManager:
    """
    Setup sinkronisasi UI dengan konfigurasi dari Google Drive.
    
    Args:
        ui_components: Dictionary komponen UI
        update_interval: Interval update dalam detik
        
    Returns:
        UISyncManager: Manager untuk sinkronisasi UI
    """
    try:
        # Buat sync manager
        sync_manager = UISyncManager(ui_components, update_interval)
        
        # Mulai sinkronisasi jika di lingkungan Colab
        if is_colab_environment():
            sync_manager.start()
        
        return sync_manager
    except Exception as e:
        logger.error(f"❌ Error saat setup sinkronisasi UI: {str(e)}")
        
        # Log ke UI jika UI logger tersedia
        if 'logger' in ui_components:
            log_sync_error(ui_components, f"Error saat setup sinkronisasi UI: {str(e)}")
        
        # Kembalikan dummy manager
        return UISyncManager(ui_components, 0)

def add_sync_button(ui_components: Dict[str, Any], sync_manager: Optional[UISyncManager] = None) -> Dict[str, Any]:
    """
    Tambahkan tombol sinkronisasi ke UI.
    
    Args:
        ui_components: Dictionary komponen UI
        sync_manager: Manager untuk sinkronisasi UI (opsional)
        
    Returns:
        Dictionary komponen UI yang telah diupdate
    """
    try:
        import ipywidgets as widgets
        
        # Buat tombol sync
        sync_button = widgets.Button(
            description='🔄 Sinkronisasi',
            tooltip='Sinkronisasi UI dengan konfigurasi dari Google Drive',
            button_style='info'
        )
        
        # Tambahkan ke komponen UI
        ui_components['sync_button'] = sync_button
        
        # Tambahkan handler
        def on_sync_clicked(b):
            try:
                log_sync_info(ui_components, "Memulai sinkronisasi manual...")
                
                # Jika sync_manager tersedia, gunakan force_sync
                if sync_manager:
                    success = sync_manager.force_sync()
                else:
                    # Load konfigurasi terbaru
                    current_config = load_config()
                    
                    # Update UI dari konfigurasi
                    update_ui_from_config(ui_components, current_config)
                    success = True
                
                if success:
                    log_sync_success(ui_components, "Sinkronisasi manual berhasil")
                else:
                    log_sync_warning(ui_components, "Sinkronisasi manual tidak sepenuhnya berhasil")
            except Exception as e:
                log_sync_error(ui_components, f"Error saat sinkronisasi manual: {str(e)}")
        
        sync_button.on_click(on_sync_clicked)
        
        # Tambahkan ke layout jika ada
        if 'button_group' in ui_components:
            ui_components['button_group'].children = ui_components['button_group'].children + (sync_button,)
        elif 'ui' in ui_components and hasattr(ui_components['ui'], 'children'):
            ui_components['ui'].children = ui_components['ui'].children + (sync_button,)
        
        return ui_components
    except Exception as e:
        logger.error(f"❌ Error saat menambahkan tombol sinkronisasi: {str(e)}")
        
        # Log ke UI jika UI logger tersedia
        if 'logger' in ui_components:
            log_sync_error(ui_components, f"Error saat menambahkan tombol sinkronisasi: {str(e)}")
        
        return ui_components 