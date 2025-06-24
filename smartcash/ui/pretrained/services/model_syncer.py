"""
File: smartcash/ui/pretrained/services/model_syncer.py
Deskripsi: Service untuk syncing models ke Google Drive
"""

import os
import shutil
from typing import List
from smartcash.common.logger import get_logger

class PretrainedModelSyncer:
    """Service untuk syncing pretrained models ke Google Drive"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def sync_models_to_drive(self, source_dir: str, drive_dir: str) -> bool:
        """
        Sync models dari local ke Google Drive.
        
        Args:
            source_dir: Source directory (local models)
            drive_dir: Drive directory target
            
        Returns:
            bool: Sync success status
        """
        try:
            # Create drive directory jika belum ada
            os.makedirs(drive_dir, exist_ok=True)
            
            # Get model files
            model_files = [f for f in os.listdir(source_dir) if f.endswith(('.pt', '.bin', '.pth'))]
            
            if not model_files:
                self.logger.warning("⚠️ No model files found to sync")
                return True
            
            # Copy each model file
            for model_file in model_files:
                source_path = os.path.join(source_dir, model_file)
                drive_path = os.path.join(drive_dir, model_file)
                
                # Check jika file sudah ada di drive
                if os.path.exists(drive_path):
                    self.logger.info(f"⏭️ Skipped {model_file} (already in drive)")
                    continue
                
                # Copy to drive
                shutil.copy2(source_path, drive_path)
                self.logger.info(f"📤 Synced {model_file} to drive")
            
            self.logger.info(f"✅ Synced {len(model_files)} models to drive")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Drive sync failed: {str(e)}")
            return False
    
    def sync_models_from_drive(self, drive_dir: str, target_dir: str) -> bool:
        """
        Sync models dari Google Drive ke local.
        
        Args:
            drive_dir: Drive directory source
            target_dir: Target directory (local)
            
        Returns:
            bool: Sync success status
        """
        try:
            if not os.path.exists(drive_dir):
                self.logger.warning(f"⚠️ Drive directory not found: {drive_dir}")
                return False
            
            # Create target directory
            os.makedirs(target_dir, exist_ok=True)
            
            # Get model files dari drive
            model_files = [f for f in os.listdir(drive_dir) if f.endswith(('.pt', '.bin', '.pth'))]
            
            if not model_files:
                self.logger.warning("⚠️ No model files found in drive")
                return True
            
            # Copy each model file
            for model_file in model_files:
                drive_path = os.path.join(drive_dir, model_file)
                target_path = os.path.join(target_dir, model_file)
                
                # Check jika file sudah ada di local
                if os.path.exists(target_path):
                    self.logger.info(f"⏭️ Skipped {model_file} (already exists)")
                    continue
                
                # Copy from drive
                shutil.copy2(drive_path, target_path)
                self.logger.info(f"📥 Synced {model_file} from drive")
            
            self.logger.info(f"✅ Synced {len(model_files)} models from drive")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Drive sync failed: {str(e)}")
            return False