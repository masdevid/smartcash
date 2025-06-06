"""
File: smartcash/ui/pretrained_model/services/model_downloader.py
Deskripsi: Service untuk download model dengan progress tracker integration
"""

import requests
from pathlib import Path
from typing import Dict, Any, List
from smartcash.ui.pretrained_model.utils.model_utils import ModelUtils, SimpleProgressDelegate

class ModelDownloader:
    """Service untuk download model dengan UI progress tracker integration"""
    
    def __init__(self, ui_components: Dict[str, Any], logger=None):
        self.ui_components, self.logger = ui_components, logger
        self.progress_delegate = SimpleProgressDelegate(ui_components)
        self.config = ModelUtils.get_models_from_ui_config(ui_components)
        self.models_dir = Path(self.config['models_dir'])
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def download_models(self, model_names: List[str]) -> Dict[str, Any]:
        """Download multiple models dengan UI progress tracking"""
        try:
            self._start_download_operation(len(model_names))
            
            downloaded_count = 0
            for i, model_name in enumerate(model_names):
                self._update_download_progress(i, len(model_names), model_name)
                downloaded_count += (1 if self._download_single_model(model_name) else 0)
            
            self._complete_download_operation(downloaded_count, len(model_names))
            return {'success': True, 'downloaded_count': downloaded_count, 'total_count': len(model_names)}
            
        except Exception as e:
            self._error_download_operation(str(e))
            return {'success': False, 'message': str(e)}
    
    def _start_download_operation(self, model_count: int) -> None:
        """Start download operation dengan progress tracker"""
        tracker = self.ui_components.get('tracker')
        tracker and tracker.show("Model Download", ["Preparation", "Download", "Validation"])
        self._safe_update_progress(10, f"📥 Memulai download {model_count} model")
    
    def _update_download_progress(self, current: int, total: int, model_name: str) -> None:
        """Update progress untuk current download"""
        progress = int((current / total) * 100) if total > 0 else 0
        self._safe_update_progress(progress, f"Download {model_name} ({current+1}/{total})")
    
    def _complete_download_operation(self, downloaded: int, total: int) -> None:
        """Complete download operation"""
        summary_msg = f"Download selesai: {downloaded}/{total} model berhasil"
        self._safe_update_progress(100, summary_msg)
        
        tracker = self.ui_components.get('tracker')
        tracker and tracker.complete(summary_msg)
    
    def _error_download_operation(self, error_msg: str) -> None:
        """Error download operation"""
        tracker = self.ui_components.get('tracker')
        tracker and tracker.error(f"Download gagal: {error_msg}")
    
    def _safe_update_progress(self, progress: int, message: str) -> None:
        """Safe update progress dengan fallback"""
        update_fn = self.ui_components.get('update_primary')
        update_fn and update_fn(progress, message)
    
    def _download_single_model(self, model_name: str) -> bool:
        """Download single model dengan UI config"""
        try:
            config = self.config['models'][model_name]
            if not config or not config.get('url'):
                self.logger and self.logger.error(f"❌ Konfigurasi {model_name} tidak valid")
                return False
            
            url, filename = config['url'], config['filename']
            file_path = self.models_dir / filename
            
            # Skip jika sudah ada dan valid
            if ModelUtils.validate_model_file(file_path, model_name):
                size_str = ModelUtils.format_file_size(file_path.stat().st_size)
                self.logger and self.logger.info(f"⏭️ Skip {config['name']} - sudah tersedia ({size_str})")
                return True
            
            self.logger and self.logger.info(f"📥 Mengunduh {config['name']} dari {url}")
            return self._download_with_progress(url, file_path, config['name'])
            
        except Exception as e:
            self.logger and self.logger.error(f"❌ Gagal download {model_name}: {str(e)}")
            return False
    
    def _download_with_progress(self, url: str, file_path: Path, model_name: str) -> bool:
        """Download file dengan progress tracking"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size, downloaded = int(response.headers.get('content-length', 0)), 0
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk), setattr(self, 'downloaded', downloaded := downloaded + len(chunk))
                        
                        # Update progress setiap 10%
                        if total_size > 0 and downloaded % (total_size // 10) < 8192:
                            percent = int(100 * downloaded / total_size)
                            size_downloaded, size_total = ModelUtils.format_file_size(downloaded), ModelUtils.format_file_size(total_size)
                            self._safe_update_progress(percent, f"Download {model_name}: {percent}% ({size_downloaded}/{size_total})")
            
            # Validate download
            if self._validate_download(file_path, total_size):
                size_str = ModelUtils.format_file_size(file_path.stat().st_size)
                self.logger and self.logger.success(f"✅ {model_name} berhasil diunduh ({size_str})")
                return True
            else:
                file_path.unlink(missing_ok=True)
                return False
            
        except Exception as e:
            self.logger and self.logger.error(f"❌ Download {model_name} gagal: {str(e)}")
            file_path.unlink(missing_ok=True)
            return False
    
    def _validate_download(self, file_path: Path, expected_size: int) -> bool:
        """Validate downloaded file dengan size check"""
        if not file_path.exists():
            return False
        actual_size = file_path.stat().st_size
        return actual_size > (expected_size * 0.8) if expected_size > 0 else actual_size > 0