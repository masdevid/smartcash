"""
File: smartcash/ui/pretrained_model/services/subprocess_download.py
Deskripsi: Layanan untuk mengunduh model pretrained menggunakan subprocess
"""

import subprocess
import os
import time
import shutil
import requests
from pathlib import Path
from typing import Dict, Any, List, Callable, Optional, Tuple
from tqdm.notebook import tqdm
from concurrent.futures import ThreadPoolExecutor

from smartcash.ui.pretrained_model.utils.logger_utils import get_module_logger, log_message
from smartcash.ui.pretrained_model.utils.model_utils import ModelManager
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.pretrained_model.utils.progress import update_progress_ui
from smartcash.ui.pretrained_model.config.model_config import get_model_config

# Gunakan logger dari utils
logger = get_module_logger()

class DownloadProcess:
    """Kelas untuk mengelola proses download model dengan subprocess"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        """Inisialisasi download process manager"""
        self.ui_components = ui_components
        self.process = None
        self.is_running = False
        self.success_count = 0  # Counter untuk model yang berhasil diunduh
        self.failed_count = 0   # Counter untuk model yang gagal diunduh
        self.total_models = 0   # Total model yang akan diunduh
        self.current_model_progress = 0  # Progress untuk model yang sedang diunduh (0-100)
        self.observer_manager = ui_components.get('observer_manager')
        self.progress_tracker = ui_components.get('progress_bar')  # Alias untuk progress_tracker
        self.log_func = ui_components.get('log_message', lambda msg, level="info": log_message(self.ui_components, msg, level))
        self.on_model_download_start = ui_components.get('on_model_download_start')  # Callback untuk update tahapan proses
    
    def _update_progress(self, progress: float, message: str) -> None:
        """Update progress tracker dengan nilai dan pesan"""
        # Simpan progress model saat ini
        self.current_model_progress = progress
        
        # Hitung progress keseluruhan berdasarkan model yang sudah selesai dan progress model saat ini
        if self.total_models > 0:
            # Bobot untuk model saat ini dan model yang sudah selesai
            completed_weight = self.success_count / self.total_models * 100
            current_weight = 1 / self.total_models * progress
            
            # Progress keseluruhan adalah jumlah dari progress model yang sudah selesai dan sebagian dari model saat ini
            overall_progress = completed_weight + current_weight
        else:
            overall_progress = progress
            
        # Gunakan API update_progress jika tersedia (API baru)
        if 'update_progress' in self.ui_components and callable(self.ui_components['update_progress']):
            self.ui_components['update_progress'](overall_progress, message)
        # Gunakan progress_tracker langsung jika tersedia
        elif self.progress_tracker and hasattr(self.progress_tracker, 'update'):
            self.progress_tracker.update(overall_progress, message)
        # Fallback ke update_progress_ui
        else:
            update_progress_ui(self.ui_components, int(overall_progress), 100, message)
    
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
    
    def download_model(self, model_url: str, target_path: Path, model_name: str, model_info: Dict[str, Any] = None) -> bool:
        """
        Download model menggunakan requests dengan progress tracking
        
        Args:
            model_url: URL model untuk diunduh
            target_path: Path tujuan untuk menyimpan model
            model_name: Nama model untuk ditampilkan
            model_info: Informasi tambahan tentang model (opsional)
            
        Returns:
            bool: True jika berhasil, False jika gagal
        """
        # Konversi target_path ke Path jika string
        if isinstance(target_path, str):
            target_path = Path(target_path)
            
        # Buat direktori jika belum ada
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Mulai download dengan streaming untuk progress bar
            model_source = model_info.get('source', '') if model_info else get_model_config(model_name).get('source', '')
            self.log_func(f"📥 Mengunduh {model_name} dari {model_source}", "info")
            
            # Buat koneksi dengan streaming
            response = requests.get(model_url, stream=True)
            response.raise_for_status()  # Raise exception jika status bukan 200 OK
            
            # Dapatkan ukuran file jika tersedia
            total_size = int(response.headers.get('content-length', 0))
            
            # Jika ukuran tidak tersedia dari header, gunakan ukuran dari konfigurasi model
            if total_size == 0 and model_info and 'size' in model_info:
                total_size = model_info.get('size', 0)
            
            # Download langsung ke file tujuan (tanpa temporary file)
            downloaded = 0
            with open(target_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if not self.is_running:
                        raise Exception("Download dibatalkan")
                        
                    if chunk:  # Filter out keep-alive chunks
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Update progress tracking
                        if total_size > 0:
                            percent = min(100, int(100 * downloaded / total_size))
                            message = f"Mengunduh {model_name}: {percent}% ({downloaded/(1024*1024):.1f}/{total_size/(1024*1024):.1f} MB)"
                            self._update_progress(percent, message)
            
            # Update metadata model jika tersedia
            if model_info:
                try:
                    # Inisialisasi ModelManager untuk metadata
                    model_manager = ModelManager(target_path.parent)
                    
                    # Dapatkan informasi model
                    model_id = model_info.get('id', f"{model_name}_downloaded")
                    version = model_info.get('version', '1.0')
                    source = model_info.get('source', model_url)
                    
                    # Update metadata
                    model_manager.update_model_metadata(target_path, model_id, version, source)
                    self.log_func(f"📋 Metadata untuk {model_name} berhasil diperbarui", "info")
                except Exception as e:
                    self.log_func(f"⚠️ Gagal memperbarui metadata: {str(e)}", "warning")
            
            # Log sukses dengan styling yang konsisten
            size_mb = target_path.stat().st_size / (1024 * 1024)
            success_msg = f"✅ {model_name} berhasil diunduh ({size_mb:.1f} MB)"
            self.log_func(success_msg, "success")
            
            # Update progress ke 100%
            self._update_progress(100, f"Selesai mengunduh {model_name}")
            
            # Notify success
            self._notify_observer('MODEL_DOWNLOAD_SUCCESS', {
                'model_name': model_name,
                'model_path': str(target_path),
                'size_mb': size_mb
            })
            
            return True
            
        except Exception as e:
            error_msg = f"❌ Gagal mengunduh {model_name}: {str(e)}"
            self.log_func(error_msg, "error")
            
            # Notify observer
            self._notify_observer('MODEL_DOWNLOAD_ERROR', {
                'message': error_msg,
                'model_name': model_name,
                'error': str(e)
            })
            
            # Hapus file yang tidak lengkap jika ada
            if target_path.exists():
                target_path.unlink()
            
            return False
    
    def start_download(self, models_info: List[Dict[str, Any]]) -> None:
        """
        Mulai proses download untuk daftar model
        
        Args:
            models_info: List dictionary dengan informasi model
                         [{'name': 'nama_model', 'url': 'url_model', 'path': 'path_target'}]
        """
        if self.is_running:
            self.log_func("🔄 Proses download sedang berjalan", "warning")
            return
            
        if not models_info:
            self.log_func("⚠️ Tidak ada model yang perlu diunduh", "warning")
            return
            
        # Set flag running dan reset counters
        self.is_running = True
        self.success_count = 0
        self.failed_count = 0
        self.total_models = len(models_info)
        self.current_model_progress = 0
        
        # Reset progress
        self._update_progress(0, "Memulai download...")
        
        # Notify start
        self._notify_observer('MODEL_DOWNLOAD_START', {
            'message': f"Memulai download {len(models_info)} model",
            'models': [model.get('name', '') for model in models_info]
        })
        
        # Jalankan download di thread terpisah
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(self._download_models, models_info)
    
    def _download_models(self, models_info: List[Dict[str, Any]]) -> None:
        """Fungsi untuk menjalankan download model secara berurutan"""
        try:
            # Proses setiap model secara berurutan
            for i, model_info in enumerate(models_info):
                if not self.is_running:
                    self.log_func("🛑 Download dihentikan", "warning")
                    break
                    
                # Dapatkan informasi model
                model_name = model_info.get('name', f"model_{i}")
                model_url = model_info.get('url', '')
                target_path = model_info.get('path', '')
                
                # Validasi informasi model
                if not model_url or not target_path:
                    self.log_func(f"⚠️ Informasi model tidak lengkap untuk {model_name}", "warning")
                    self.failed_count += 1
                    continue
                
                # Panggil callback on_model_download_start jika tersedia
                if self.on_model_download_start and callable(self.on_model_download_start):
                    self.on_model_download_start(model_name)
                
                # Siapkan metadata model dengan semua field yang diperlukan
                model_metadata = {
                    'id': model_info.get('id', f"{model_name}_downloaded"),
                    'version': model_info.get('version', '1.0'),
                    'source': model_info.get('source', model_url)
                }
                
                # Download model dengan metadata
                success = self.download_model(model_url, target_path, model_name, model_metadata)
                
                # Update counter berdasarkan hasil
                if success:
                    self.success_count += 1
                else:
                    self.failed_count += 1
            
            # Update final progress jika proses masih berjalan
            if self.is_running:
                self._update_progress(100, "Download selesai")
                
                # Log summary dengan emoji yang sesuai
                if self.failed_count == 0:
                    summary = f"✅ Download selesai: Semua {self.success_count} model berhasil diunduh"
                    log_level = "success"
                else:
                    summary = f"⚠️ Download selesai: {self.success_count}/{self.total_models} berhasil, {self.failed_count} gagal"
                    log_level = "warning"
                    
                self.log_func(summary, log_level)
                
                # Notify complete dengan informasi lengkap
                self._notify_observer('MODEL_DOWNLOAD_COMPLETE', {
                    'message': summary,
                    'total': self.total_models,
                    'success': self.success_count,
                    'failed': self.failed_count
                })
        
        except Exception as e:
            error_msg = f"❌ Error dalam proses download: {str(e)}"
            self.log_func(error_msg, "error")
            
            # Notify error dengan detail
            self._notify_observer('MODEL_DOWNLOAD_ERROR', {
                'message': error_msg,
                'error': str(e)
            })
        
        finally:
            # Pastikan flag is_running diatur ke False
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
