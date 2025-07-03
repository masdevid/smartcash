"""
File: smartcash/ui/setup/env_config/handlers/operations/config_operation.py

Config Operation Handler untuk stage-based setup.
"""

import json
from typing import Dict, Any, List
from pathlib import Path

from smartcash.ui.core.handlers.operation_handler import OperationHandler
from smartcash.ui.setup.env_config.constants import DEFAULT_CONFIG_DIR, DRIVE_PATH, COLAB_PATH


class ConfigOperation(OperationHandler):
    """Handler untuk config synchronization operations."""
    
    def __init__(self):
        """Initialize config operation handler."""
        super().__init__(
            module_name='config_operation',
            parent_module='setup.env_config'
        )
        
    def sync_configs(self) -> Dict[str, Any]:
        """Sync configuration files dari drive ke colab menggunakan config manager.
        
        Returns:
            Dictionary berisi hasil sync
        """
        try:
            self.logger.info("⚙️ Syncing configuration files...")
            
            # Import config manager
            from smartcash.common.config import get_config_manager
            config_manager = get_config_manager()
            
            drive_config_path = Path(DRIVE_PATH) / DEFAULT_CONFIG_DIR
            colab_config_path = Path(COLAB_PATH) / DEFAULT_CONFIG_DIR
            
            configs_synced = 0
            errors = []
            
            # Ensure colab config directory exists
            colab_config_path.mkdir(parents=True, exist_ok=True)
            
            # Check if drive config exists
            if not drive_config_path.exists():
                return {
                    'status': True,
                    'message': 'No config files to sync (drive config directory not found)',
                    'configs_synced': 0
                }
            
            # Sync semua config files yang terdeteksi (.yaml, .yml, .json)
            config_extensions = ['*.yaml', '*.yml', '*.json']
            
            for pattern in config_extensions:
                for config_file in drive_config_path.glob(pattern):
                    try:
                        target_file = colab_config_path / config_file.name
                        
                        # Copy file
                        import shutil
                        shutil.copy2(config_file, target_file)
                        configs_synced += 1
                        
                        self.logger.info(f"📄 Synced config: {config_file.name}")
                        
                    except Exception as e:
                        error_msg = f"Failed to sync {config_file.name}: {str(e)}"
                        errors.append(error_msg)
                        self.logger.error(error_msg)
            
            success = len(errors) == 0
            return {
                'status': success,
                'message': f'Synced {configs_synced} config files' if success else f'Sync completed with {len(errors)} errors',
                'configs_synced': configs_synced,
                'errors': errors
            }
            
        except Exception as e:
            error_msg = f"Config sync operation failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                'status': False,
                'message': error_msg,
                'error': str(e),
                'configs_synced': 0
            }
    
    def sync_templates(self, force_overwrite: bool = False) -> Dict[str, Any]:
        """Sync config templates dari colab ke drive menggunakan config manager.
        
        Args:
            force_overwrite: Force overwrite existing templates
            
        Returns:
            Dictionary berisi hasil sync
        """
        try:
            self.logger.info("📋 Syncing config templates...")
            
            # Import config manager
            from smartcash.common.config import get_config_manager
            config_manager = get_config_manager()
            
            drive_config_path = Path(DRIVE_PATH) / DEFAULT_CONFIG_DIR
            colab_config_path = Path(COLAB_PATH) / DEFAULT_CONFIG_DIR
            
            templates_synced = 0
            errors = []
            
            # Ensure drive config directory exists
            drive_config_path.mkdir(parents=True, exist_ok=True)
            
            # Check if colab config exists
            if not colab_config_path.exists():
                return {
                    'status': True,
                    'message': 'No config templates to sync (colab config directory not found)',
                    'templates_synced': 0
                }
            
            # Sync semua config template files (.yaml, .yml, .json)
            config_extensions = ['*.yaml', '*.yml', '*.json']
            
            for pattern in config_extensions:
                for template_file in colab_config_path.glob(pattern):
                    try:
                        target_file = drive_config_path / template_file.name
                        
                        # Skip if exists and not force overwrite
                        if target_file.exists() and not force_overwrite:
                            self.logger.debug(f"Skipping existing template: {template_file.name}")
                            continue
                        
                        # Copy file
                        import shutil
                        shutil.copy2(template_file, target_file)
                        templates_synced += 1
                        
                        self.logger.info(f"📄 Synced template: {template_file.name}")
                        
                    except Exception as e:
                        error_msg = f"Failed to sync template {template_file.name}: {str(e)}"
                        errors.append(error_msg)
                        self.logger.error(error_msg)
            
            success = len(errors) == 0
            return {
                'status': success,
                'message': f'Synced {templates_synced} config templates' if success else f'Template sync completed with {len(errors)} errors',
                'templates_synced': templates_synced,
                'errors': errors
            }
            
        except Exception as e:
            error_msg = f"Config template sync failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                'status': False,
                'message': error_msg,
                'error': str(e),
                'templates_synced': 0
            }
    
    def verify_configs(self) -> Dict[str, Any]:
        """Verify config files integrity menggunakan config manager.
        
        Returns:
            Dictionary berisi verification results
        """
        try:
            self.logger.info("🔍 Verifying config files...")
            
            # Import config manager
            from smartcash.common.config import get_config_manager
            config_manager = get_config_manager()
            
            colab_config_path = Path(COLAB_PATH) / DEFAULT_CONFIG_DIR
            
            if not colab_config_path.exists():
                return {
                    'status': False,
                    'message': 'Config directory not found',
                    'config_check': {}
                }
            
            config_check = {}
            all_valid = True
            
            # Check semua config files yang terdeteksi
            config_extensions = ['*.yaml', '*.yml', '*.json']
            
            for pattern in config_extensions:
                for config_file in colab_config_path.glob(pattern):
                    try:
                        # Use config manager untuk validasi
                        if config_file.suffix.lower() in ['.yaml', '.yml']:
                            import yaml
                            with open(config_file, 'r') as f:
                                data = yaml.safe_load(f)
                        else:  # .json
                            import json
                            with open(config_file, 'r') as f:
                                data = json.load(f)
                        
                        config_check[config_file.name] = {
                            'valid': True,
                            'size': config_file.stat().st_size,
                            'type': config_file.suffix.lower(),
                            'keys': len(data) if isinstance(data, dict) else 'not_dict'
                        }
                        
                    except Exception as e:
                        config_check[config_file.name] = {
                            'valid': False,
                            'error': str(e)
                        }
                        all_valid = False
            
            return {
                'status': all_valid,
                'message': 'All config files verified' if all_valid else 'Some config files invalid',
                'config_check': config_check
            }
            
        except Exception as e:
            error_msg = f"Config verification failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                'status': False,
                'message': error_msg,
                'error': str(e),
                'config_check': {}
            }
    
    def create_default_configs(self) -> Dict[str, Any]:
        """Create default config files menggunakan config manager auto-detection.
        
        Returns:
            Dictionary berisi hasil creation
        """
        try:
            self.logger.info("📝 Creating default config files...")
            
            # Import config manager
            from smartcash.common.config import get_config_manager
            config_manager = get_config_manager()
            
            colab_config_path = Path(COLAB_PATH) / DEFAULT_CONFIG_DIR
            colab_config_path.mkdir(parents=True, exist_ok=True)
            
            # Detect config files from repo (smartcash/configs)
            repo_config_path = Path(__file__).parent.parent.parent.parent.parent / 'configs'
            
            configs_created = 0
            errors = []
            
            if not repo_config_path.exists():
                return {
                    'status': True,
                    'message': 'No default config files found in repo',
                    'configs_created': 0
                }
            
            # Copy semua config files dari repo
            config_extensions = ['*.yaml', '*.yml', '*.json']
            
            for pattern in config_extensions:
                for config_file in repo_config_path.glob(pattern):
                    try:
                        target_file = colab_config_path / config_file.name
                        
                        # Skip if exists
                        if target_file.exists():
                            self.logger.debug(f"Config already exists: {config_file.name}")
                            continue
                        
                        # Copy file
                        import shutil
                        shutil.copy2(config_file, target_file)
                        configs_created += 1
                        
                        self.logger.info(f"📄 Created default config: {config_file.name}")
                        
                    except Exception as e:
                        error_msg = f"Failed to create {config_file.name}: {str(e)}"
                        errors.append(error_msg)
                        self.logger.error(error_msg)
            
            success = len(errors) == 0
            return {
                'status': success,
                'message': f'Created {configs_created} default config files' if success else f'Config creation completed with {len(errors)} errors',
                'configs_created': configs_created,
                'errors': errors
            }
            
        except Exception as e:
            error_msg = f"Default config creation failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                'status': False,
                'message': error_msg,
                'error': str(e),
                'configs_created': 0
            }