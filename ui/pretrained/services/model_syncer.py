# File: smartcash/ui/pretrained/services/model_syncer.py
"""
File: smartcash/ui/pretrained/services/model_syncer.py
Deskripsi: Complete service untuk syncing models dengan progress tracking
"""

import os
import shutil
from typing import List, Optional, Callable, Dict, Any
from smartcash.common.logger import get_logger

logger = get_logger(__name__)

class PretrainedModelSyncer:
    """🔄 Service untuk syncing pretrained models dengan progress tracking"""
    
    def __init__(self):
        self.logger = logger
        self._model_extensions = ['.pt', '.bin', '.pth', '.ckpt']
    
    def sync_to_drive(self, source_dir: str, drive_dir: str,
                     progress_callback: Optional[Callable[[int, str], None]] = None,
                     status_callback: Optional[Callable[[str], None]] = None) -> bool:
        """📤 Sync models dari local ke drive dengan progress tracking
        
        Args:
            source_dir: Source directory (local models)
            drive_dir: Drive directory target
            progress_callback: Callback untuk update progress (progress_pct, message)
            status_callback: Callback untuk update status message
            
        Returns:
            True jika sync berhasil, False jika gagal
        """
        try:
            if status_callback:
                status_callback("📁 Preparing drive directory...")
            
            # Create drive directory jika belum ada
            os.makedirs(drive_dir, exist_ok=True)
            
            # Get model files
            model_files = self._get_model_files(source_dir)
            
            if not model_files:
                if status_callback:
                    status_callback("⚠️ No model files found to sync")
                if progress_callback:
                    progress_callback(100, "No files to sync")
                return True
            
            if status_callback:
                status_callback(f"📤 Syncing {len(model_files)} files to drive...")
            
            # Copy files dengan progress tracking
            for i, model_file in enumerate(model_files):
                source_path = os.path.join(source_dir, model_file)
                drive_path = os.path.join(drive_dir, model_file)
                
                # Update progress
                progress = int((i / len(model_files)) * 90) + 5  # 5-95%
                if progress_callback:
                    progress_callback(progress, f"Syncing {model_file}...")
                
                # Check jika file sudah ada di drive dengan ukuran sama
                if self._files_are_identical(source_path, drive_path):
                    self.logger.info(f"⏭️ Skipped {model_file} (already synced)")
                    continue
                
                # Copy to drive
                shutil.copy2(source_path, drive_path)
                self.logger.info(f"📤 Synced {model_file} to drive")
            
            if status_callback:
                status_callback(f"✅ Successfully synced {len(model_files)} files to drive")
            if progress_callback:
                progress_callback(100, "Sync to drive completed")
            
            return True
            
        except Exception as e:
            error_msg = f"❌ Drive sync failed: {str(e)}"
            self.logger.error(error_msg)
            if status_callback:
                status_callback(error_msg)
            return False
    
    def sync_from_drive(self, drive_dir: str, target_dir: str,
                       progress_callback: Optional[Callable[[int, str], None]] = None,
                       status_callback: Optional[Callable[[str], None]] = None) -> bool:
        """📥 Sync models dari drive ke local dengan progress tracking
        
        Args:
            drive_dir: Drive directory source
            target_dir: Target directory (local models)
            progress_callback: Callback untuk update progress (progress_pct, message)
            status_callback: Callback untuk update status message
            
        Returns:
            True jika sync berhasil, False jika gagal
        """
        try:
            if status_callback:
                status_callback("📁 Checking drive directory...")
            
            if not os.path.exists(drive_dir):
                error_msg = f"❌ Drive directory not found: {drive_dir}"
                if status_callback:
                    status_callback(error_msg)
                return False
            
            # Create target directory jika belum ada
            os.makedirs(target_dir, exist_ok=True)
            
            # Get model files dari drive
            model_files = self._get_model_files(drive_dir)
            
            if not model_files:
                if status_callback:
                    status_callback("⚠️ No model files found in drive")
                if progress_callback:
                    progress_callback(100, "No files to sync")
                return True
            
            if status_callback:
                status_callback(f"📥 Syncing {len(model_files)} files from drive...")
            
            # Copy files dengan progress tracking
            for i, model_file in enumerate(model_files):
                drive_path = os.path.join(drive_dir, model_file)
                target_path = os.path.join(target_dir, model_file)
                
                # Update progress
                progress = int((i / len(model_files)) * 90) + 5  # 5-95%
                if progress_callback:
                    progress_callback(progress, f"Syncing {model_file}...")
                
                # Check jika file sudah ada di local dengan ukuran sama
                if self._files_are_identical(drive_path, target_path):
                    self.logger.info(f"⏭️ Skipped {model_file} (already synced)")
                    continue
                
                # Copy from drive
                shutil.copy2(drive_path, target_path)
                self.logger.info(f"📥 Synced {model_file} from drive")
            
            if status_callback:
                status_callback(f"✅ Successfully synced {len(model_files)} files from drive")
            if progress_callback:
                progress_callback(100, "Sync from drive completed")
            
            return True
            
        except Exception as e:
            error_msg = f"❌ Drive sync failed: {str(e)}"
            self.logger.error(error_msg)
            if status_callback:
                status_callback(error_msg)
            return False
    
    def bi_directional_sync(self, local_dir: str, drive_dir: str,
                          progress_callback: Optional[Callable[[int, str], None]] = None,
                          status_callback: Optional[Callable[[str], None]] = None) -> bool:
        """🔄 Bi-directional sync antara local dan drive
        
        Args:
            local_dir: Local directory
            drive_dir: Drive directory
            progress_callback: Callback untuk update progress
            status_callback: Callback untuk update status
            
        Returns:
            True jika sync berhasil
        """
        try:
            if status_callback:
                status_callback("🔍 Analyzing sync requirements...")
            
            local_files = set(self._get_model_files(local_dir))
            drive_files = set(self._get_model_files(drive_dir)) if os.path.exists(drive_dir) else set()
            
            # Files to sync to drive (local only)
            to_drive = local_files - drive_files
            # Files to sync from drive (drive only)
            from_drive = drive_files - local_files
            
            total_operations = len(to_drive) + len(from_drive)
            
            if total_operations == 0:
                if status_callback:
                    status_callback("✅ Models already in sync")
                if progress_callback:
                    progress_callback(100, "Already synchronized")
                return True
            
            current_op = 0
            
            # Sync to drive
            if to_drive:
                if status_callback:
                    status_callback(f"📤 Syncing {len(to_drive)} files to drive...")
                
                os.makedirs(drive_dir, exist_ok=True)
                for file in to_drive:
                    source_path = os.path.join(local_dir, file)
                    drive_path = os.path.join(drive_dir, file)
                    shutil.copy2(source_path, drive_path)
                    
                    current_op += 1
                    progress = int((current_op / total_operations) * 90) + 5
                    if progress_callback:
                        progress_callback(progress, f"Synced {file} to drive")
            
            # Sync from drive
            if from_drive:
                if status_callback:
                    status_callback(f"📥 Syncing {len(from_drive)} files from drive...")
                
                os.makedirs(local_dir, exist_ok=True)
                for file in from_drive:
                    drive_path = os.path.join(drive_dir, file)
                    local_path = os.path.join(local_dir, file)
                    shutil.copy2(drive_path, local_path)
                    
                    current_op += 1
                    progress = int((current_op / total_operations) * 90) + 5
                    if progress_callback:
                        progress_callback(progress, f"Synced {file} from drive")
            
            if status_callback:
                status_callback("✅ Bi-directional sync completed")
            if progress_callback:
                progress_callback(100, "Synchronization completed")
            
            return True
            
        except Exception as e:
            error_msg = f"❌ Bi-directional sync failed: {str(e)}"
            self.logger.error(error_msg)
            if status_callback:
                status_callback(error_msg)
            return False
    
    def _get_model_files(self, directory: str) -> List[str]:
        """📋 Get list of model files dalam directory"""
        try:
            if not os.path.exists(directory):
                return []
            
            model_files = []
            for file in os.listdir(directory):
                if any(file.endswith(ext) for ext in self._model_extensions):
                    model_files.append(file)
            
            return sorted(model_files)
            
        except Exception as e:
            self.logger.error(f"❌ Error listing model files: {str(e)}")
            return []
    
    def _files_are_identical(self, file1: str, file2: str) -> bool:
        """🔍 Check apakah dua file identical (size dan timestamp)"""
        try:
            if not (os.path.exists(file1) and os.path.exists(file2)):
                return False
            
            stat1 = os.stat(file1)
            stat2 = os.stat(file2)
            
            # Compare size dan modification time
            return (stat1.st_size == stat2.st_size and 
                   abs(stat1.st_mtime - stat2.st_mtime) < 2)  # 2 second tolerance
            
        except Exception:
            return False
    
    def get_sync_status(self, local_dir: str, drive_dir: str) -> Dict[str, Any]:
        """📊 Get sync status information"""
        try:
            local_files = set(self._get_model_files(local_dir))
            drive_files = set(self._get_model_files(drive_dir)) if os.path.exists(drive_dir) else set()
            
            return {
                'local_files': list(local_files),
                'drive_files': list(drive_files),
                'local_only': list(local_files - drive_files),
                'drive_only': list(drive_files - local_files),
                'common_files': list(local_files & drive_files),
                'in_sync': local_files == drive_files,
                'needs_sync': len(local_files ^ drive_files) > 0
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error getting sync status: {str(e)}")
            return {
                'local_files': [],
                'drive_files': [],
                'local_only': [],
                'drive_only': [],
                'common_files': [],
                'in_sync': False,
                'needs_sync': True,
                'error': str(e)
            }