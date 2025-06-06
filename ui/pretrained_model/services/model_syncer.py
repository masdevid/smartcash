"""
File: smartcash/ui/pretrained_model/services/model_syncer.py
Deskripsi: Service untuk sync model ke Google Drive dengan progress tracker integration
"""

import shutil
from pathlib import Path
from typing import Dict, Any
from smartcash.ui.pretrained_model.utils.model_utils import ModelUtils, SimpleProgressDelegate

class ModelSyncer:
    """Service untuk sync model ke Google Drive dengan UI progress tracker"""
    
    def __init__(self, ui_components: Dict[str, Any], logger=None):
        self.ui_components, self.logger = ui_components, logger
        self.progress_delegate = SimpleProgressDelegate(ui_components)
        self.config = ModelUtils.get_models_from_ui_config(ui_components)
        self.models_dir = Path(self.config['models_dir'])
        self.drive_models_dir = Path(self.config['drive_models_dir'])
    
    def sync_to_drive(self) -> Dict[str, Any]:
        """Sync local models ke Google Drive dengan progress tracking"""
        try:
            self._start_sync_operation()
            
            # Check Drive availability
            if not self._is_drive_available():
                error_msg = "Google Drive tidak tersedia"
                self._error_sync_operation(error_msg)
                return {'success': False, 'message': error_msg}
            
            # Ensure drive directory exists
            self.drive_models_dir.mkdir(parents=True, exist_ok=True)
            
            # Get local model files
            model_files = [self.models_dir / config['filename'] 
                          for config in self.config['models'].values() 
                          if (self.models_dir / config['filename']).exists()]
            
            if not model_files:
                success_msg = "Tidak ada model untuk disinkronkan"
                self._complete_sync_operation(0, success_msg)
                return {'success': True, 'synced_count': 0}
            
            synced_count = 0
            for i, model_file in enumerate(model_files):
                self._update_sync_progress(i, len(model_files), model_file.name)
                synced_count += (1 if self._sync_single_file(model_file) else 0)
            
            self._complete_sync_operation(synced_count, f"{synced_count} model disinkronkan")
            return {'success': True, 'synced_count': synced_count}
            
        except Exception as e:
            self._error_sync_operation(str(e))
            return {'success': False, 'message': str(e)}
    
    def _start_sync_operation(self) -> None:
        """Start sync operation dengan progress tracker"""
        tracker = self.ui_components.get('tracker')
        tracker and tracker.show("Drive Sync", ["Check Drive", "Copy Files", "Validation"])
        self._safe_update_progress(10, "☁️ Memulai sinkronisasi ke Drive")
    
    def _update_sync_progress(self, current: int, total: int, filename: str) -> None:
        """Update progress untuk current sync"""
        progress = int((current / total) * 100) if total > 0 else 0
        self._safe_update_progress(progress, f"Sync {filename} ({current+1}/{total})")
    
    def _complete_sync_operation(self, synced_count: int, message: str) -> None:
        """Complete sync operation"""
        self._safe_update_progress(100, f"☁️ {message}")
        self.logger and self.logger.success(f"☁️ {message}")
        
        tracker = self.ui_components.get('tracker')
        tracker and tracker.complete(message)
    
    def _error_sync_operation(self, error_msg: str) -> None:
        """Error sync operation"""
        self.logger and self.logger.error(f"💥 Sync gagal: {error_msg}")
        
        tracker = self.ui_components.get('tracker')
        tracker and tracker.error(f"Sync gagal: {error_msg}")
    
    def _safe_update_progress(self, progress: int, message: str) -> None:
        """Safe update progress dengan fallback"""
        update_fn = self.ui_components.get('update_primary')
        update_fn and update_fn(progress, message)
    
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