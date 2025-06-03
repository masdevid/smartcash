"""
File: smartcash/ui/pretrained_model/services/model_downloader.py
Deskripsi: Service khusus untuk download model dengan step-by-step progress tracking
"""

import requests
from pathlib import Path
from typing import Dict, Any, List, Callable
from tqdm.auto import tqdm
from smartcash.ui.pretrained_model.constants.model_constants import MODEL_CONFIGS
from smartcash.ui.pretrained_model.utils.model_utils import ModelUtils, ProgressTracker

class ModelDownloader:
    """Service untuk download model dengan detailed progress tracking"""
    
    def __init__(self, config: Dict[str, Any], logger=None, progress_tracker: ProgressTracker = None):
        self.config = config
        self.logger = logger
        self.progress_tracker = progress_tracker
        self.models_dir = Path(config.get('models_dir', '/content/models'))
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def download_models(self, model_names: List[str]) -> Dict[str, Any]:
        """Download multiple models dengan step-by-step progress"""
        try:
            if self.progress_tracker:
                self.progress_tracker.next_step('DOWNLOAD_START', f"Memulai download {len(model_names)} model")
            
            downloaded_count = 0
            total_models = len(model_names)
            
            for i, model_name in enumerate(model_names):
                if self.progress_tracker:
                    self.progress_tracker.update_current_step(
                        (i * 100) // total_models, 
                        f"Download {model_name} ({i+1}/{total_models})"
                    )
                
                if self._download_single_model(model_name, i + 1, total_models):
                    downloaded_count += 1
            
            return {'success': True, 'downloaded_count': downloaded_count, 'total_count': total_models}
            
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    def _download_single_model(self, model_name: str, current: int, total: int) -> bool:
        """Download single model dengan detailed progress"""
        try:
            config = ModelUtils.get_model_config(model_name)
            if not config:
                self.logger and self.logger.error(f"❌ Konfigurasi {model_name} tidak ditemukan")
                return False
            
            url = config['url']
            filename = config['filename']
            file_path = self.models_dir / filename
            
            # Skip jika sudah ada dan valid
            if ModelUtils.validate_model_file(file_path, model_name):
                size_str = ModelUtils.format_file_size(file_path.stat().st_size)
                self.logger and self.logger.info(f"⏭️ Skip {config['name']} - sudah tersedia ({size_str})")
                return True
            
            # Download dengan progress
            self.logger and self.logger.info(f"📥 Mengunduh {config['name']} dari {config['url']}")
            return self._download_with_progress(url, file_path, config['name'])
            
        except Exception as e:
            self.logger and self.logger.error(f"❌ Gagal download {model_name}: {str(e)}")
            return False
    
    def _download_with_progress(self, url: str, file_path: Path, model_name: str) -> bool:
        """Download file dengan detailed progress tracking"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Update progress dalam download
                        if total_size > 0 and self.progress_tracker:
                            percent = int(100 * downloaded / total_size)
                            size_downloaded = ModelUtils.format_file_size(downloaded)
                            size_total = ModelUtils.format_file_size(total_size)
                            self.progress_tracker.update_current_step(
                                percent, f"Download {model_name}: {percent}% ({size_downloaded}/{size_total})"
                            )
            
            # Validate download
            if not self._validate_download(file_path, total_size):
                file_path.unlink(missing_ok=True)
                return False
            
            size_str = ModelUtils.format_file_size(file_path.stat().st_size)
            self.logger and self.logger.success(f"✅ {model_name} berhasil diunduh ({size_str})")
            return True
            
        except Exception as e:
            self.logger and self.logger.error(f"❌ Download {model_name} gagal: {str(e)}")
            file_path.unlink(missing_ok=True)
            return False
    
    def _validate_download(self, file_path: Path, expected_size: int) -> bool:
        """Validate downloaded file"""
        if not file_path.exists():
            return False
        
        actual_size = file_path.stat().st_size
        return actual_size > (expected_size * 0.8) if expected_size > 0 else actual_size > 0