"""
File: smartcash/ui/dataset/visualization/auto_refresh.py
Deskripsi: Modul untuk menangani auto refresh download dataset dengan delay
"""

import threading
import time
from typing import Callable, Optional

from smartcash.common.logger import get_logger

logger = get_logger(__name__)

class AutoRefreshDownloader:
    """Kelas untuk menangani auto refresh download dataset dengan delay"""
    
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls) -> 'AutoRefreshDownloader':
        """
        Mendapatkan instance singleton dari AutoRefreshDownloader
        
        Returns:
            Instance AutoRefreshDownloader
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = AutoRefreshDownloader()
            return cls._instance
    
    def __init__(self):
        """Inisialisasi AutoRefreshDownloader"""
        self.refresh_thread = None
        self.is_running = False
        self.download_callback = None
        
    def start_auto_refresh(self, download_callback: Callable, delay_ms: int = 100):
        """
        Memulai auto refresh download dengan delay tertentu
        
        Args:
            download_callback: Fungsi callback untuk download
            delay_ms: Delay dalam milidetik sebelum memulai download
        """
        if self.is_running:
            logger.warning("🔄 Auto refresh sudah berjalan, menghentikan yang lama...")
            self.stop_auto_refresh()
        
        self.download_callback = download_callback
        self.is_running = True
        
        # Buat thread baru untuk auto refresh
        self.refresh_thread = threading.Thread(
            target=self._run_delayed_refresh,
            args=(delay_ms,),
            daemon=True
        )
        self.refresh_thread.start()
        logger.info(f"🕒 Auto refresh akan dimulai dalam {delay_ms}ms")
    
    def _run_delayed_refresh(self, delay_ms: int):
        """
        Menjalankan refresh dengan delay
        
        Args:
            delay_ms: Delay dalam milidetik
        """
        # Konversi ms ke detik
        delay_sec = delay_ms / 1000.0
        
        # Tunggu sesuai delay
        time.sleep(delay_sec)
        
        # Jalankan callback jika masih running
        if self.is_running and self.download_callback:
            try:
                logger.info("🔄 Menjalankan auto refresh download...")
                self.download_callback()
                logger.info("✅ Auto refresh download selesai")
            except Exception as e:
                logger.error(f"❌ Error saat auto refresh: {str(e)}")
        
        # Reset flag
        self.is_running = False
    
    def stop_auto_refresh(self):
        """Menghentikan auto refresh yang sedang berjalan"""
        if self.is_running:
            self.is_running = False
            logger.info("🛑 Auto refresh dihentikan")


# Fungsi helper untuk memulai auto refresh
def trigger_auto_refresh(download_callback: Callable, delay_ms: int = 100):
    """
    Memulai auto refresh download dengan delay tertentu
    
    Args:
        download_callback: Fungsi callback untuk download
        delay_ms: Delay dalam milidetik sebelum memulai download
    """
    auto_refresher = AutoRefreshDownloader.get_instance()
    auto_refresher.start_auto_refresh(download_callback, delay_ms)
