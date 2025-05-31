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
import json

from smartcash.ui.utils.constants import ICONS, COLORS
from smartcash.common.logger import get_logger
from smartcash.ui.utils.ui_logger_namespace import PRETRAINED_MODEL_LOGGER_NAMESPACE, KNOWN_NAMESPACES
from smartcash.ui.pretrained_model.utils.progress import update_progress_ui

# Gunakan namespace yang benar untuk logger
MODULE_LOGGER_NAME = KNOWN_NAMESPACES[PRETRAINED_MODEL_LOGGER_NAMESPACE]
logger = get_logger(PRETRAINED_MODEL_LOGGER_NAMESPACE)

class DownloadProcess:
    """Kelas untuk mengelola proses download model dengan subprocess"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        """Inisialisasi download process manager"""
        self.ui_components = ui_components
        self.process = None
        self.is_running = False
        self.observer_manager = ui_components.get('observer_manager')
        self.progress_tracker = ui_components.get('progress_bar')  # Alias untuk progress_tracker
        self.log_func = ui_components.get('log_message', self._default_log)
        self.on_model_download_start = ui_components.get('on_model_download_start')  # Callback untuk update tahapan proses
    
    def _default_log(self, message: str, level: str = "info") -> None:
        """Fungsi log default jika tidak ada log_func yang diberikan"""
        if 'log' in self.ui_components and hasattr(self.ui_components['log'], 'append_display_data'):
            from IPython.display import display, HTML
            from smartcash.ui.utils.ui_logger_namespace import create_namespace_badge
            from smartcash.ui.utils.logging_utils import format_log_message_html
            
            # Gunakan format_log_message_html jika tersedia di ui_components
            if 'format_log_message_html' in self.ui_components and callable(self.ui_components['format_log_message_html']):
                html_message = self.ui_components['format_log_message_html'](message, level)
                with self.ui_components['log']:
                    display(HTML(html_message))
            else:
                # Gunakan styling yang konsisten dengan namespace badge
                namespace_badge = create_namespace_badge("PRETRAIN")
                
                level_icon = {
                    "info": ICONS.get('info', 'ℹ️'),
                    "success": ICONS.get('success', '✅'),
                    "warning": ICONS.get('warning', '⚠️'),
                    "error": ICONS.get('error', '❌')
                }.get(level, 'ℹ️')
                
                # Styling yang konsisten dengan modul lain
                with self.ui_components['log']:
                    display(HTML(f"""
                    <div style="margin:2px 0;padding:4px 8px;border-radius:4px;
                               background-color:rgba(248,249,250,0.8);
                               border-left:3px solid #D7BDE2;">
                        <span style="margin-right:5px;">{namespace_badge}</span>
                        <span>{level_icon} {message}</span>
                    </div>
                    """))
        else:
            # Gunakan logger dengan namespace yang benar
            if level == "info":
                logger.info(f"{message}")
            elif level == "success":
                logger.info(f"✅ {message}")
            elif level == "warning":
                logger.warning(f"{message}")
            elif level == "error":
                logger.error(f"{message}")
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
        
        # Jalankan download langsung (tanpa threading)
        self._download_models(models_info)
    
    def _download_models(self, models_info: List[Dict[str, Any]]) -> None:
        """Fungsi untuk menjalankan download"""
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
