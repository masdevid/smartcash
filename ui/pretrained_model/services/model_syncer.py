"""
File: smartcash/ui/pretrained_model/services/model_syncer.py
Deskripsi: Optimized model syncer dengan enhanced progress tracker integration
"""

import shutil
from pathlib import Path
from typing import Dict, Any
from smartcash.ui.pretrained_model.utils.model_utils import ModelUtils, ProgressHelper

class ModelSyncer:
    """Service untuk sync model ke Google Drive dengan enhanced progress integration"""
    
    def __init__(self, ui_components: Dict[str, Any], logger=None):
        self.ui_components, self.logger = ui_components, logger
        self.progress_helper = ProgressHelper(ui_components)
        self.config = ModelUtils.get_models_from_ui_config(ui_components)
        self.models_dir = Path(self.config['models_dir'])
        self.drive_models_dir = Path(self.config['drive_models_dir'])
    
    def sync_to_drive(self) -> Dict[str, Any]:
        """Sync local models ke Google Drive dengan enhanced progress tracking"""
        try:
            # Check Drive availability
            if not self._is_drive_available():
                self.logger and self.logger.warning("⚠️ Google Drive tidak tersedia")
                return {'success': False, 'message': 'Drive tidak tersedia'}
            
            # Ensure drive directory exists
            self.drive_models_dir.mkdir(parents=True, exist_ok=True)
            
            # Get local model files
            model_files = [self.models_dir / config['filename'] 
                          for config in self.config['models'].values() 
                          if (self.models_dir / config['filename']).exists()]
            
            if not model_files:
                self.logger and self.logger.info("ℹ️ Tidak ada model untuk disinkronkan")
                return {'success': True, 'synced_count': 0}
            
            synced_count = 0
            for i, model_file in enumerate(model_files):
                # Update current operation progress
                current_progress = ((i + 1) * 100) // len(model_files)
                self.progress_helper.update_current_step(
                    current_progress, 
                    f"Sync {model_file.name} ({i+1}/{len(model_files)})"
                )
                
                if self._sync_single_file(model_file):
                    synced_count += 1
            
            self.logger and self.logger.success(f"☁️ {synced_count} model disinkronkan ke Drive")
            return {'success': True, 'synced_count': synced_count}
            
        except Exception as e:
            self.logger and self.logger.error(f"💥 Sync gagal: {str(e)}")
            return {'success': False, 'message': str(e)}
    
    def _is_drive_available(self) -> bool:
        """Check if Google Drive is mounted dengan one-liner"""
        return Path('/content/drive/MyDrive').exists()
    
    def _sync_single_file(self, source_file: Path) -> bool:
        """Sync single file to Drive dengan progress info"""
        try:
            target_file = self.drive_models_dir / source_file.name
            
            # Skip if already exists with same size
            if (target_file.exists() and target_file.stat().st_size == source_file.stat().st_size):
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