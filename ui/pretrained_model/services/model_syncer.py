"""
File: smartcash/ui/pretrained_model/services/model_syncer.py
Deskripsi: Service khusus untuk sync model ke Google Drive dengan progress tracking
"""

import shutil
from pathlib import Path
from typing import Dict, Any
from smartcash.ui.pretrained_model.utils.model_utils import ModelUtils, ProgressTracker

class ModelSyncer:
    """Service untuk sync model ke Google Drive dengan detailed progress"""
    
    def __init__(self, config: Dict[str, Any], logger=None, progress_tracker: ProgressTracker = None):
        self.config = config
        self.logger = logger
        self.progress_tracker = progress_tracker
        self.models_dir = Path(config.get('models_dir', '/content/models'))
        self.drive_models_dir = Path(config.get('drive_models_dir', '/content/drive/MyDrive/SmartCash/models'))
    
    def sync_to_drive(self) -> Dict[str, Any]:
        """Sync local models ke Google Drive dengan step-by-step progress"""
        try:
            if self.progress_tracker:
                self.progress_tracker.next_step('SYNC_START', "Memulai sinkronisasi ke Drive")
            
            # Check if Drive is available
            if not self._is_drive_available():
                self.logger and self.logger.warning("⚠️ Google Drive tidak tersedia")
                return {'success': False, 'message': 'Drive tidak tersedia'}
            
            # Ensure drive directory exists
            self.drive_models_dir.mkdir(parents=True, exist_ok=True)
            
            # Get local model files using constants
            model_files = []
            for model_name in ModelUtils.get_all_model_names():
                file_path = ModelUtils.get_model_file_path(model_name, str(self.models_dir))
                if file_path.exists():
                    model_files.append(file_path)
            
            if not model_files:
                self.logger and self.logger.info("ℹ️ Tidak ada model untuk disinkronkan")
                return {'success': True, 'synced_count': 0}
            
            synced_count = 0
            for i, model_file in enumerate(model_files):
                if self.progress_tracker:
                    self.progress_tracker.update_current_step(
                        (i * 100) // len(model_files),
                        f"Sync {model_file.name} ({i+1}/{len(model_files)})"
                    )
                
                if self._sync_single_file(model_file):
                    synced_count += 1
            
            if self.progress_tracker:
                self.progress_tracker.next_step('SYNC_COMPLETE', f"{synced_count} model disinkronkan")
            
            self.logger and self.logger.success(f"☁️ {synced_count} model disinkronkan ke Drive")
            return {'success': True, 'synced_count': synced_count}
            
        except Exception as e:
            self.logger and self.logger.error(f"💥 Sync gagal: {str(e)}")
            return {'success': False, 'message': str(e)}
    
    def _is_drive_available(self) -> bool:
        """Check if Google Drive is mounted and accessible"""
        return Path('/content/drive/MyDrive').exists()
    
    def _sync_single_file(self, source_file: Path) -> bool:
        """Sync single file to Drive dengan progress info"""
        try:
            target_file = self.drive_models_dir / source_file.name
            
            # Skip if already exists with same size
            if (target_file.exists() and 
                target_file.stat().st_size == source_file.stat().st_size):
                size_str = ModelUtils.format_file_size(target_file.stat().st_size)
                self.logger and self.logger.info(f"⏭️ Skip sync {source_file.name} - sudah tersedia ({size_str})")
                return True
            
            # Copy file to Drive
            shutil.copy2(source_file, target_file)
            size_str = ModelUtils.format_file_size(target_file.stat().st_size)
            self.logger and self.logger.info(f"📤 {source_file.name} → Drive ({size_str})")
            return True
            
        except Exception as e:
            self.logger and self.logger.error(f"❌ Gagal sync {source_file.name}: {str(e)}")
            return False