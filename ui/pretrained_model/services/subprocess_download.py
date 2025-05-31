"""
File: smartcash/ui/pretrained_model/services/subprocess_download.py
Deskripsi: Layanan untuk mengunduh model pretrained menggunakan subprocess
"""

import subprocess
import sys
import time
from pathlib import Path
import urllib.request
import tempfile
from typing import Dict, Any, List, Tuple, Optional, Callable, Union
import concurrent.futures
import threading
import json

from smartcash.ui.utils.constants import ICONS, COLORS
from smartcash.common.logger import get_logger
from smartcash.ui.pretrained_model.utils.progress import update_progress_ui

logger = get_logger(__name__)

class DownloadProcess:
    """Kelas untuk mengelola proses download model dengan subprocess"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        """Inisialisasi download process manager"""
        self.ui_components = ui_components
        self.process = None
        self.is_running = False
        self.download_thread = None
        self.observer_manager = ui_components.get('observer_manager')
        self.progress_tracker = ui_components.get('progress_bar')  # Alias untuk progress_tracker
        self.log_func = ui_components.get('log_message', self._default_log)
        self.on_model_download_start = ui_components.get('on_model_download_start')  # Callback untuk update tahapan proses
    
    def _default_log(self, message: str, level: str = "info") -> None:
        """Fungsi log default jika tidak ada log_func yang diberikan"""
        if 'log' in self.ui_components and hasattr(self.ui_components['log'], 'append_display_data'):
            from IPython.display import display, HTML
            level_color = {
                "info": COLORS.get('alert_info_text', '#0c5460'),
                "success": COLORS.get('alert_success_text', '#155724'),
                "warning": COLORS.get('alert_warning_text', '#856404'),
                "error": COLORS.get('alert_danger_text', '#721c24')
            }.get(level, '#0c5460')
            
            level_icon = {
                "info": ICONS.get('info', 'ℹ️'),
                "success": ICONS.get('success', '✅'),
                "warning": ICONS.get('warning', '⚠️'),
                "error": ICONS.get('error', '❌')
            }.get(level, 'ℹ️')
            
            with self.ui_components['log']:
                display(HTML(f"<div style='color:{level_color}'>{level_icon} {message}</div>"))
        else:
            logger.info(f"{message}")
    
    def _update_progress(self, progress: float, message: str) -> None:
        """Update progress tracker dengan nilai dan pesan"""
        # Gunakan API update_progress jika tersedia (API baru)
        if 'update_progress' in self.ui_components and callable(self.ui_components['update_progress']):
            self.ui_components['update_progress'](progress, message)
        # Gunakan progress_tracker langsung jika tersedia
        elif self.progress_tracker and hasattr(self.progress_tracker, 'update'):
            self.progress_tracker.update(progress, message)
        # Fallback ke update_progress_ui
        else:
            update_progress_ui(self.ui_components, int(progress), 100, message)
    
    def _notify_observer(self, event_type: str, data: Dict[str, Any]) -> None:
        """Notifikasi ke observer jika tersedia"""
        if self.observer_manager and hasattr(self.observer_manager, 'notify'):
            try:
                self.observer_manager.notify(event_type, None, {
                    'timestamp': time.time(),
                    **data
                })
            except Exception:
                pass  # Silent fail untuk observer notification
    
    def download_model(self, model_url: str, target_path: Union[str, Path], model_name: str) -> None:
        """
        Download model menggunakan urllib dengan progress tracking
        
        Args:
            model_url: URL model untuk diunduh
            target_path: Path tujuan untuk menyimpan model
            model_name: Nama model untuk ditampilkan
        """
        if isinstance(target_path, str):
            target_path = Path(target_path)
            
        # Buat direktori jika belum ada
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Buat temporary file untuk download
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        
        try:
            # Dapatkan ukuran file
            file_size = 0
            try:
                with urllib.request.urlopen(model_url) as response:
                    file_size = int(response.info().get('Content-Length', 0))
            except Exception as e:
                self.log_func(f"⚠️ Tidak dapat mendapatkan ukuran file: {str(e)}", "warning")
                file_size = 0
            
            # Fungsi untuk melaporkan progress download
            def report_progress(block_num, block_size, total_size):
                if not self.is_running:
                    raise Exception("Download dibatalkan")
                    
                downloaded = block_num * block_size
                percent = min(100, int(100 * downloaded / total_size if total_size > 0 else 0))
                message = f"Mengunduh {model_name}: {percent}% ({downloaded/(1024*1024):.1f}/{total_size/(1024*1024):.1f} MB)"
                
                # Update progress tracking
                self._update_progress(percent, message)
            
            # Download file
            urllib.request.urlretrieve(model_url, temp_path, reporthook=report_progress)
            
            # Setelah download selesai, pindahkan ke target
            temp_path.rename(target_path)
            
            # Log sukses
            size_mb = target_path.stat().st_size / (1024 * 1024)
            self.log_func(f"{model_name} berhasil diunduh (<span style='color:{COLORS.get('alert_success_text', '#155724')}'>{size_mb:.1f} MB</span>)", "success")
            
            # Notify observer
            self._notify_observer('MODEL_DOWNLOAD_SUCCESS', {
                'message': f"{model_name} berhasil diunduh ({size_mb:.1f} MB)",
                'model_name': model_name,
                'file_size': size_mb,
                'target_path': str(target_path)
            })
            
            return True
            
        except Exception as e:
            error_msg = f"Gagal mengunduh {model_name}: {str(e)}"
            self.log_func(error_msg, "error")
            
            # Notify observer
            self._notify_observer('MODEL_DOWNLOAD_ERROR', {
                'message': error_msg,
                'model_name': model_name,
                'error': str(e)
            })
            
            if temp_path.exists():
                temp_path.unlink()  # Hapus file temporary jika terjadi error
            
            return False
    
    def start_download(self, models_info: List[Dict[str, Any]]) -> None:
        """
        Mulai proses download untuk daftar model
        
        Args:
            models_info: List dictionary dengan informasi model
                         [{'name': 'nama_model', 'url': 'url_model', 'path': 'path_target'}]
        """
        if self.is_running:
            self.log_func("⚠️ Proses download sudah berjalan", "warning")
            return
        
        self.is_running = True
        
        # Reset progress tracking
        if 'reset_progress_bar' in self.ui_components and callable(self.ui_components['reset_progress_bar']):
            self.ui_components['reset_progress_bar'](0, "Memulai download model...")
        
        # Notify start
        self._notify_observer('MODEL_DOWNLOAD_START', {
            'message': f"Memulai download {len(models_info)} model",
            'models_count': len(models_info),
            'models': [model['name'] for model in models_info]
        })
        
        # Log start
        self.log_func(f"🚀 Memulai download {len(models_info)} model...", "info")
        
        # Jalankan download dalam thread terpisah
        self.download_thread = threading.Thread(
            target=self._download_thread_func,
            args=(models_info,)
        )
        self.download_thread.daemon = True
        self.download_thread.start()
    
    def _download_thread_func(self, models_info: List[Dict[str, Any]]) -> None:
        """Fungsi thread untuk menjalankan download"""
        try:
            total_models = len(models_info)
            success_count = 0
            failed_count = 0
            
            for i, model_info in enumerate(models_info):
                if not self.is_running:
                    break
                
                model_name = model_info.get('name', f"Model-{i+1}")
                model_url = model_info.get('url', '')
                target_path = model_info.get('path', '')
                
                if not model_url or not target_path:
                    self.log_func(f"⚠️ Informasi model tidak lengkap untuk {model_name}", "warning")
                    failed_count += 1
                    continue
                
                # Update progress
                progress_pct = (i / total_models) * 100
                self._update_progress(progress_pct, f"Memulai download {model_name}...")
                
                # Panggil callback on_model_download_start jika tersedia
                if self.on_model_download_start and callable(self.on_model_download_start):
                    self.on_model_download_start(model_name)
                
                # Download model
                success = self.download_model(model_url, target_path, model_name)
                
                if success:
                    success_count += 1
                else:
                    failed_count += 1
            
            # Update final progress
            if self.is_running:
                self._update_progress(100, "Download selesai")
                
                # Log summary
                summary = f"✅ Download selesai: {success_count}/{total_models} berhasil, {failed_count} gagal"
                self.log_func(summary, "success" if failed_count == 0 else "warning")
                
                # Notify complete
                self._notify_observer('MODEL_DOWNLOAD_COMPLETE', {
                    'message': summary,
                    'total': total_models,
                    'success': success_count,
                    'failed': failed_count
                })
        
        except Exception as e:
            error_msg = f"Error dalam proses download: {str(e)}"
            self.log_func(error_msg, "error")
            
            # Notify error
            self._notify_observer('MODEL_DOWNLOAD_ERROR', {
                'message': error_msg,
                'error': str(e)
            })
        
        finally:
            self.is_running = False
    
    def stop_download(self) -> None:
        """Hentikan proses download yang sedang berjalan"""
        if not self.is_running:
            return
            
        self.is_running = False
        
        # Log stop
        self.log_func("🛑 Menghentikan proses download...", "warning")
        
        # Notify stop
        self._notify_observer('MODEL_DOWNLOAD_STOP', {
            'message': "Proses download dihentikan oleh pengguna"
        })
        
        # Wait for thread to finish
        if self.download_thread and self.download_thread.is_alive():
            self.download_thread.join(timeout=1.0)
        
        # Reset progress
        if 'reset_progress_bar' in self.ui_components and callable(self.ui_components['reset_progress_bar']):
            self.ui_components['reset_progress_bar'](0, "Download dihentikan")


def download_models(models_info: List[Dict[str, Any]], ui_components: Dict[str, Any]) -> None:
    """
    Fungsi helper untuk mendownload model dengan subprocess
    
    Args:
        models_info: List dictionary dengan informasi model
                     [{'name': 'nama_model', 'url': 'url_model', 'path': 'path_target'}]
        ui_components: Dictionary komponen UI
    """
    # Buat instance DownloadProcess
    download_process = DownloadProcess(ui_components)
    
    # Simpan instance ke ui_components untuk digunakan oleh handler lain
    ui_components['download_process'] = download_process
    
    # Mulai proses download
    download_process.start_download(models_info)
