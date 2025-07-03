"""
File: smartcash/ui/setup/env_config/handlers/operations/folder_operation.py

Folder Operation Handler untuk stage-based setup.
"""

import os
import shutil
from typing import Dict, Any, List, Tuple
from pathlib import Path

from smartcash.ui.core.handlers.operation_handler import OperationHandler
from smartcash.ui.setup.env_config.constants import REQUIRED_FOLDERS, SYMLINK_MAP


class FolderOperation(OperationHandler):
    """Handler untuk folder creation operations."""
    
    def __init__(self):
        """Initialize folder operation handler."""
        super().__init__(
            module_name='folder_operation',
            parent_module='setup.env_config'
        )
        
    def create_folders(self) -> Dict[str, Any]:
        """Create required folders dan symlinks menggunakan constants.
        
        Returns:
            Dictionary berisi hasil creation
        """
        try:
            self.logger.info("📁 Creating required folders...")
            
            folders_created = []
            symlinks_created = []
            errors = []
            
            # Create required folders dari constants
            for folder_path in REQUIRED_FOLDERS:
                try:
                    path = Path(folder_path)
                    if not path.exists():
                        path.mkdir(parents=True, exist_ok=True)
                        folders_created.append(str(path))
                        self.logger.info(f"✅ Created folder: {path}")
                    else:
                        self.logger.debug(f"📁 Folder already exists: {path}")
                        
                except Exception as e:
                    error_msg = f"Failed to create folder {folder_path}: {str(e)}"
                    errors.append(error_msg)
                    self.logger.error(error_msg)
            
            # Create symlinks dari constants
            for source, target in SYMLINK_MAP.items():
                try:
                    source_path = Path(source)
                    target_path = Path(target)
                    
                    # Ensure source exists
                    if not source_path.exists():
                        source_path.mkdir(parents=True, exist_ok=True)
                    
                    # Create symlink jika belum ada
                    if not target_path.exists():
                        target_path.symlink_to(source_path)
                        symlinks_created.append((str(source_path), str(target_path)))
                        self.logger.info(f"🔗 Created symlink: {target_path} -> {source_path}")
                    else:
                        self.logger.debug(f"🔗 Symlink already exists: {target_path}")
                        
                except Exception as e:
                    error_msg = f"Failed to create symlink {source} -> {target}: {str(e)}"
                    errors.append(error_msg)
                    self.logger.error(error_msg)
            
            # Return results
            success = len(errors) == 0
            return {
                'status': success,
                'message': f'Created {len(folders_created)} folders, {len(symlinks_created)} symlinks' if success else f'Completed with {len(errors)} errors',
                'folders_created': len(folders_created),
                'symlinks_created': len(symlinks_created),
                'folder_list': folders_created,
                'symlink_list': symlinks_created,
                'errors': errors
            }
            
        except Exception as e:
            error_msg = f"Folder operation failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                'status': False,
                'message': error_msg,
                'error': str(e),
                'folders_created': 0,
                'symlinks_created': 0
            }
    
    def verify_folders(self) -> Dict[str, Any]:
        """Verify folders dan symlinks menggunakan constants.
        
        Returns:
            Dictionary berisi verification results
        """
        try:
            self.logger.info("🔍 Verifying folders and symlinks...")
            
            verified_folders = []
            missing_folders = []
            verified_symlinks = []
            missing_symlinks = []
            
            # Verify required folders
            for folder_path in REQUIRED_FOLDERS:
                path = Path(folder_path)
                if path.exists() and path.is_dir():
                    verified_folders.append(str(path))
                else:
                    missing_folders.append(str(path))
            
            # Verify symlinks
            for source, target in SYMLINK_MAP.items():
                source_path = Path(source)
                target_path = Path(target)
                
                if target_path.exists() and target_path.is_symlink():
                    verified_symlinks.append((str(source_path), str(target_path)))
                else:
                    missing_symlinks.append((str(source_path), str(target_path)))
            
            all_verified = len(missing_folders) == 0 and len(missing_symlinks) == 0
            
            return {
                'status': all_verified,
                'message': 'All folders and symlinks verified' if all_verified else 'Some folders/symlinks missing',
                'verified_folders': verified_folders,
                'missing_folders': missing_folders,
                'verified_symlinks': verified_symlinks,
                'missing_symlinks': missing_symlinks
            }
            
        except Exception as e:
            error_msg = f"Folder verification failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                'status': False,
                'message': error_msg,
                'error': str(e)
            }

    # === OperationHandler abstract method implementations ===
    def initialize(self) -> Dict[str, Any]:
        """Concrete initialize to satisfy BaseHandler contract."""
        # For this simple handler, initialization just verifies folders
        return self.verify_folders()

    def get_operations(self) -> Dict[str, Any]:
        """Return available operations mapping."""
        return {
            'create_folders': self.create_folders,
            'verify_folders': self.verify_folders,
            'cleanup_folders': self.cleanup_folders
        }
    
    def cleanup_folders(self) -> Dict[str, Any]:
        """Cleanup created folders (untuk rollback).
        
        Returns:
            Dictionary berisi cleanup results
        """
        try:
            self.logger.info("🧹 Cleaning up folders...")
            
            cleaned_folders = []
            cleaned_symlinks = []
            errors = []
            
            # Remove symlinks dulu
            for source, target in SYMLINK_MAP.items():
                try:
                    target_path = Path(target)
                    if target_path.exists() and target_path.is_symlink():
                        target_path.unlink()
                        cleaned_symlinks.append(str(target_path))
                        self.logger.info(f"🗑️ Removed symlink: {target_path}")
                except Exception as e:
                    errors.append(f"Failed to remove symlink {target}: {str(e)}")
            
            # Remove folders (hanya yang kosong)
            for folder_path in reversed(REQUIRED_FOLDERS):
                try:
                    path = Path(folder_path)
                    if path.exists() and path.is_dir():
                        # Only remove if empty
                        if not any(path.iterdir()):
                            path.rmdir()
                            cleaned_folders.append(str(path))
                            self.logger.info(f"🗑️ Removed empty folder: {path}")
                except Exception as e:
                    errors.append(f"Failed to remove folder {folder_path}: {str(e)}")
            
            success = len(errors) == 0
            return {
                'status': success,
                'message': f'Cleaned {len(cleaned_folders)} folders, {len(cleaned_symlinks)} symlinks' if success else f'Cleanup completed with {len(errors)} errors',
                'cleaned_folders': cleaned_folders,
                'cleaned_symlinks': cleaned_symlinks,
                'errors': errors
            }
            
        except Exception as e:
            error_msg = f"Folder cleanup failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                'status': False,
                'message': error_msg,
                'error': str(e)
            }