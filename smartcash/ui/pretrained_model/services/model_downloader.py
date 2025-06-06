"""
File: smartcash/ui/pretrained_model/services/model_downloader.py
Deskripsi: Optimized model downloader dengan enhanced progress tracker integration
"""

import requests
from pathlib import Path
from typing import Dict, Any, List
from smartcash.ui.pretrained_model.utils.model_utils import ModelUtils, ProgressHelper

class ModelDownloader:
    """Service untuk download model dengan enhanced progress integration"""
    
    def __init__(self, ui_components: Dict[str, Any], logger=None):
        self.ui_components, self.logger = ui_components, logger
        self.progress_helper = ProgressHelper(ui_components)
        self.config = ModelUtils.get_models_from_ui_config(ui_components)
        self.models_dir = Path(self.config['models_dir'])
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def download_models(self, model_names: List[str]) -> Dict[str, Any]:
        """Download multiple models dengan enhanced progress tracking"""
        try:
            downloaded_count = 0
            
            for i, model_name in enumerate(model_names):
                # Update step progress untuk setiap model
                step_progress = ((i + 1) * 100) // len(model_names)
                
                # Download single model dengan progress update
                if self._download_single_model(model_name, i + 1, len(model_names)):
                    downloaded_count += 1
                
                # Update step progress after each model
                self.progress_helper.update_current_step(100, f"Model {i+1}/{len(model_names)} selesai")
            
            return {'success': True, 'downloaded_count': downloaded_count, 'total_count': len(model_names)}
            
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    def _download_single_model(self, model_name: str, current_idx: int, total_count: int) -> bool:
        """Download single model dengan enhanced progress tracking"""
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
            return self._download_with_enhanced_progress(url, file_path, config['name'], current_idx, total_count)
            
        except Exception as e:
            self.logger and self.logger.error(f"❌ Gagal download {model_name}: {str(e)}")
            return False
    
    def _download_with_enhanced_progress(self, url: str, file_path: Path, model_name: str, 
                                       current_idx: int, total_count: int) -> bool:
        """Download file dengan enhanced progress tracking"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size, downloaded = int(response.headers.get('content-length', 0)), 0
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Update current operation progress dengan enhanced tracking
                        if total_size > 0:
                            download_percent = int(100 * downloaded / total_size)
                            size_downloaded, size_total = ModelUtils.format_file_size(downloaded), ModelUtils.format_file_size(total_size)
                            
                            # Update current progress dengan detailed message
                            self.progress_helper.update_current_step(
                                download_percent, 
                                f"Download {model_name}: {download_percent}% ({size_downloaded}/{size_total}) - Model {current_idx}/{total_count}"
                            )
            
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