"""
Folder and symlink management handler for environment setup.

This module provides the FolderHandler class which manages the creation of
required folders and symlinks for the application, ensuring proper directory
structure and file system organization with proper error handling.
"""

import datetime
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict, Union

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

from smartcash.common.environment import get_environment_manager
from smartcash.ui.handlers.base_handler import BaseHandler
from smartcash.ui.setup.env_config.handlers.base_config_mixin import BaseConfigMixin
from smartcash.ui.setup.env_config.constants import SetupStage

class FolderOperationResult(TypedDict, total=False):
    """Type definition for folder operation results."""
    created_count: int
    symlinks_count: int
    backups_count: int
    folders_created: List[str]
    symlinks_created: List[Tuple[str, str]]
    backups_created: List[str]
    source_dirs_created: List[str]
    errors: List[str]
    status: str
    message: str

class FolderHandler(BaseHandler, BaseConfigMixin):
    """Handler for folder and symlink management.
    
    This handler manages the creation of required folders and symlinks,
    using centralized configuration from ConfigHandler.
    """
    
    # Default configuration for the handler
    DEFAULT_CONFIG = {
        'base_dir': '/content',
        'create_missing': True,
        'backup_existing': True,
        'backup_suffix': '.bak',
        'default_permissions': 0o755,
        'symlink_relative': True
    }
    
    def __init__(self, config_handler=None, **kwargs):
        """Initialize the FolderHandler with configuration.
        
        Args:
            config_handler: Instance of ConfigHandler for configuration
            **kwargs: Additional keyword arguments for BaseHandler
        """
        # Initialize BaseHandler first
        super().__init__(
            module_name='folder',
            parent_module='env_config',
            **kwargs
        )
        
        # Then initialize BaseConfigMixin
        BaseConfigMixin.__init__(self, config_handler=config_handler, **kwargs)
        
        # Get environment manager
        self.env_manager = get_environment_manager(logger=self.logger)
        
        # Initialize operation results
        self._last_operation_result = None
        
        # Log initialization
        self.logger.debug("FolderHandler initialized with configuration")
        self.logger.debug(f"Base directory: {self.get_config_value('base_dir')}")
        self.logger.debug(f"Create missing: {self.get_config_value('create_missing')}")
        self.logger.debug(f"Backup existing: {self.get_config_value('backup_existing')}")
        
        # Initialize from config
        self.base_dir = self.get_config_value('base_dir', '/content')
        self.create_missing = self.get_config_value('create_missing', True)
        self.backup_existing = self.get_config_value('backup_existing', True)
        self.backup_suffix = self.get_config_value('backup_suffix', '.bak')
        self.required_folders = self.config.get('required_folders', [])
        self.required_symlinks = self.config.get('symlinks', {})
        
        self.logger.debug("Initialized FolderHandler")
    
    async def setup_symlinks(self) -> Dict[str, any]:
        """Set up required symlinks before folder creation.
        
        This method should be called before create_required_folders to ensure
        symlinks are in place before any folder operations.
        
        Returns:
            Dict with operation status and details
        """
        self.set_stage(SetupStage.SYMLINK_SETUP, "Setting up symlinks")
        
        result = {
            'status': False,
            'symlinks_created': [],
            'errors': [],
            'message': 'Symlink setup started'
        }
        
        if not self.required_symlinks:
            result.update({
                'status': True,
                'message': 'No symlinks to create',
                'symlinks_created': []
            })
            return result
            
        try:
            symlinks_created = []
            
            for target, source in self.required_symlinks.items():
                target_path = Path(target).expanduser().resolve()
                source_path = Path(source).expanduser().resolve()
                
                # Skip if symlink already exists and points to the correct location
                if target_path.exists():
                    if target_path.is_symlink() and os.readlink(str(target_path)) == str(source_path):
                        self.logger.debug(f"Symlink already exists: {target} -> {source}")
                        continue
                    
                    # Backup existing file/directory if enabled
                    if self.enable_backup:
                        backup_path = await self._backup_path(target_path)
                        if backup_path:
                            result.setdefault('backups', []).append(str(backup_path))
                    
                    # Remove existing file/directory
                    if not self.dry_run:
                        if target_path.is_dir():
                            shutil.rmtree(str(target_path))
                        else:
                            target_path.unlink()
                
                # Create parent directory if it doesn't exist
                parent_dir = target_path.parent
                if not parent_dir.exists() and not self.dry_run:
                    parent_dir.mkdir(parents=True, exist_ok=True)
                
                # Create symlink
                if not self.dry_run:
                    os.symlink(str(source_path), str(target_path), target_is_directory=source_path.is_dir())
                
                symlinks_created.append((str(target_path), str(source_path)))
                self.logger.info(f"Created symlink: {target_path} -> {source_path}")
            
            result.update({
                'status': True,
                'symlinks_created': symlinks_created,
                'message': f'Successfully created {len(symlinks_created)} symlinks'
            })
            
        except Exception as e:
            error_msg = f"Error setting up symlinks: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            result.update({
                'status': False,
                'message': error_msg,
                'errors': [error_msg]
            })
        
        return result
        
    async def create_required_folders(self) -> FolderOperationResult:
        """Create all required folders.
        
        Note: This should be called after setup_symlinks() to ensure
        symlinks are in place before folder creation.
        
        Returns:
            FolderOperationResult with operation status and details
        """
        self.set_stage(SetupStage.FOLDER_SETUP, "Creating required folders")
        
        result: FolderOperationResult = {
            'created_count': 0,
            'symlinks_count': 0,
            'backups_count': 0,
            'folders_created': [],
            'symlinks_created': [],
            'backups_created': [],
            'source_dirs_created': [],
            'errors': [],
            'status': 'pending',
            'message': 'Starting folder setup'
        }
        
        try:
            # Create required folders
            if self.required_folders:
                created, errors = await self._create_folders(self.required_folders)
                result['folders_created'].extend(created)
                result['errors'].extend(errors)
                result['created_count'] = len(created)
            
            # Create symlinks if enabled
            if self.create_symlinks and self.required_symlinks:
                created, errors = await self._create_symlinks(self.required_symlinks)
                result['symlinks_created'].extend(created)
                result['errors'].extend(errors)
                result['symlinks_count'] = len(created)
            
            # Update status based on results
            if result['errors']:
                result['status'] = 'warning'
                result['message'] = f"Completed with {len(result['errors'])} errors"
            else:
                result['status'] = 'completed'
                result['message'] = 'All folders and symlinks created successfully'
            
            self.logger.info(result['message'])
            
        except Exception as e:
            error_msg = f"Error during folder setup: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            result['status'] = 'error'
            result['message'] = error_msg
            result['errors'].append(error_msg)
        
        return result
    
    async def _backup_path(self, path: Path) -> Optional[Path]:
        """Create a backup of the specified path.
        
        Args:
            path: Path to back up
            
        Returns:
            Path to the backup if successful, None otherwise
        """
        if not self.enable_backup or not path.exists():
            return None
            
        try:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir = path.parent / '.backups'
            backup_name = f"{path.name}.bak_{timestamp}"
            backup_path = backup_dir / backup_name
            
            # Create backup directory if it doesn't exist
            if not backup_dir.exists() and not self.dry_run:
                backup_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Creating backup of {path} at {backup_path}")
            
            if not self.dry_run:
                if path.is_dir():
                    shutil.copytree(str(path), str(backup_path))
                else:
                    shutil.copy2(str(path), str(backup_path))
            
            # Clean up old backups if max_backups is set
            if self.max_backups > 0 and not self.dry_run:
                await self._cleanup_old_backups(backup_dir, path.name)
            
            return backup_path
            
        except Exception as e:
            self.logger.warning(f"Failed to create backup of {path}: {str(e)}")
            return None
    
    async def _cleanup_old_backups(self, backup_dir: Path, base_name: str) -> None:
        """Remove old backups, keeping only the most recent max_backups.
        
        Args:
            backup_dir: Directory containing backup files
            base_name: Base name of the files to clean up
        """
        try:
            # Find all backup files for this path
            backup_pattern = f"{base_name}.bak_*"
            backups = sorted(
                backup_dir.glob(backup_pattern),
                key=os.path.getmtime,
                reverse=True
            )
            
            # Remove old backups beyond max_backups
            for backup in backups[self.max_backups:]:
                try:
                    if not self.dry_run:
                        if backup.is_dir():
                            shutil.rmtree(str(backup))
                        else:
                            backup.unlink()
                    self.logger.debug(f"Removed old backup: {backup}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove old backup {backup}: {str(e)}")
                    
        except Exception as e:
            self.logger.warning(f"Error cleaning up old backups: {str(e)}")
    
    async def _create_folders(self, folders: List[Union[str, Dict]]) -> Tuple[List[str], List[str]]:
        """Create multiple folders with error handling.
        
        Args:
            folders: List of folder paths or dicts with path and other options
            
        Returns:
            Tuple of (created_folders, error_messages)
        """
        created = []
        errors = []
        
        for folder in folders:
            try:
                folder_path = Path(folder).resolve()
                
                # Skip if folder already exists
                if folder_path.exists():
                    if folder_path.is_dir():
                        self.logger.debug(f"Folder already exists: {folder_path}")
                    else:
                        error_msg = f"Path exists but is not a directory: {folder_path}"
                        self.logger.error(error_msg)
                        errors.append(error_msg)
                    continue
                
                # Create the folder
                if not self.dry_run:
                    folder_path.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"Created folder: {folder_path}")
                else:
                    self.logger.info(f"[DRY RUN] Would create folder: {folder_path}")
                
                created.append(str(folder_path))
                
            except Exception as e:
                error_msg = f"Error creating folder {folder}: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                errors.append(error_msg)
        
        return created, errors
    
    async def _create_symlinks(self, symlinks: Dict[str, str]) -> Tuple[List[Tuple[str, str]], List[str]]:
        """Create symlinks from source to target.
        
        Args:
            symlinks: Dictionary of {source: target} symlink mappings
            
        Returns:
            Tuple of (created_symlinks, error_messages)
        """
        created = []
        errors = []
        
        for source, target in symlinks.items():
            try:
                source_path = Path(source).resolve()
                target_path = Path(target).resolve()
                
                # Skip if symlink already exists and points to the correct target
                if target_path.exists():
                    if target_path.is_symlink():
                        current_target = Path(os.readlink(str(target_path))).resolve()
                        if current_target == source_path:
                            self.logger.debug(f"Symlink already exists: {target} -> {source}")
                            created.append((str(source_path), str(target_path)))
                            continue
                        else:
                            self.logger.warning(
                                f"Symlink exists but points to different target: "
                                f"{target} -> {current_target} (expected: {source})"
                            )
                    else:
                        error_msg = f"Target exists but is not a symlink: {target}"
                        self.logger.error(error_msg)
                        errors.append(error_msg)
                        continue
                
                # Ensure source exists
                if not source_path.exists():
                    error_msg = f"Source does not exist: {source}"
                    self.logger.error(error_msg)
                    errors.append(error_msg)
                    continue
                
                # Create parent directory if it doesn't exist
                if not self.dry_run:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Create the symlink
                if not self.dry_run:
                    if target_path.exists():
                        if target_path.is_dir():
                            shutil.rmtree(str(target_path))
                        else:
                            target_path.unlink()
                    
                    os.symlink(str(source_path), str(target_path))
                    self.logger.info(f"Created symlink: {target} -> {source}")
                else:
                    self.logger.info(f"[DRY RUN] Would create symlink: {target} -> {source}")
                
                created.append((str(source_path), str(target_path)))
                
            except Exception as e:
                error_msg = f"Error creating symlink {target} -> {source}: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                errors.append(error_msg)
        
        return created, errors
    
    def verify_folder_structure(self) -> Dict[str, Any]:
        """Verify that all required folders and symlinks exist.
        
        Returns:
            Dictionary with verification results
        """
        result = {
            'missing_folders': [],
            'invalid_symlinks': [],
            'valid_folders': [],
            'valid_symlinks': [],
            'is_valid': True
        }
        
        # Check required folders
        for folder in self.required_folders:
            folder_path = Path(folder)
            if folder_path.exists() and folder_path.is_dir():
                result['valid_folders'].append(str(folder_path))
            else:
                result['missing_folders'].append(str(folder_path))
                result['is_valid'] = False
        
        # Check symlinks
        for source, target in self.required_symlinks.items():
            target_path = Path(target)
            if target_path.exists() and target_path.is_symlink():
                try:
                    actual_target = Path(os.readlink(str(target_path))).resolve()
                    expected_target = Path(source).resolve()
                    if actual_target == expected_target:
                        result['valid_symlinks'].append((str(source), str(target)))
                    else:
                        result['invalid_symlinks'].append({
                            'source': str(source),
                            'target': str(target),
                            'actual_target': str(actual_target),
                            'reason': 'wrong_target'
                        })
                        result['is_valid'] = False
                except OSError as e:
                    result['invalid_symlinks'].append({
                        'source': str(source),
                        'target': str(target),
                        'reason': 'invalid_symlink',
                        'error': str(e)
                    })
                    result['is_valid'] = False
            else:
                result['invalid_symlinks'].append({
                    'source': str(source),
                    'target': str(target),
                    'reason': 'not_a_symlink'
                })
                result['is_valid'] = False
        
        return result
