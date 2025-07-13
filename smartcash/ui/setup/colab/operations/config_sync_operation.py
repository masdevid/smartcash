"""
File: smartcash/ui/setup/colab/operations/config_sync_operation.py
Description: Sync configuration files using the config manager
"""

import os
from typing import Dict, Any, Optional, Callable
from smartcash.ui.core.handlers.operation_handler import OperationHandler
from smartcash.common.config.manager import get_config_manager
from smartcash.common.constants.paths import COLAB_DATA_ROOT


class ConfigSyncOperation(OperationHandler):
    """Sync configuration files using the config manager."""
    
    def __init__(self, config: Dict[str, Any], **kwargs):
        """Initialize config sync operation.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional arguments
        """
        super().__init__(
            module_name='config_sync_operation',
            parent_module='colab',
            **kwargs
        )
        self.config = config
        self.config_manager = get_config_manager(auto_sync=False)
    
    def initialize(self) -> None:
        """Initialize the config sync operation."""
        self.logger.info("🚀 Initializing config sync operation")
        # No specific initialization needed for config sync operation
        self.logger.info("✅ Config sync operation initialization complete")
    
    def get_operations(self) -> Dict[str, Callable]:
        """Get available operations."""
        return {
            'sync_configs': self.execute_sync_configs
        }
    
    def execute_sync_configs(self, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Sync configuration files using the config manager.
        
        Args:
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with operation results
        """
        try:
            if progress_callback:
                progress_callback(10, "🔍 Checking config templates...")
            
            # Check if COLAB_DATA_ROOT exists
            if not os.path.exists(COLAB_DATA_ROOT):
                os.makedirs(COLAB_DATA_ROOT, exist_ok=True)
                self.log(f"Created COLAB_DATA_ROOT directory: {COLAB_DATA_ROOT}", 'info')
            
            # Ensure config directory exists
            config_dir = Path('/content/smartcash/configs')
            if not config_dir.exists():
                self.log(f"Config directory not found at {config_dir}, creating...", 'warning')
                try:
                    config_dir.mkdir(parents=True, exist_ok=True)
                    self.log(f"Created config directory: {config_dir}", 'info')
                    
                    # Create a sample config file if none exist
                    sample_config = config_dir / 'sample_config.yaml'
                    if not sample_config.exists():
                        sample_config.write_text("# Sample Configuration\n# Add your configuration here\n")
                        self.log(f"Created sample config: {sample_config}", 'info')
                except Exception as e:
                    self.log(f"Error creating config directory: {str(e)}", 'error')
                    return {
                        'success': False,
                        'error': f'Failed to create config directory: {str(e)}',
                        'config_dir': str(config_dir),
                        'cwd': os.getcwd(),
                        'ls_content': os.listdir('/content/smartcash' if os.path.exists('/content/smartcash') else '/content')
                    }
            
            if progress_callback:
                progress_callback(25, "📋 Discovering available configs...")
            
            # Discover configs from repo
            available_configs = self.config_manager.discover_repo_configs()
            self.log(f"Found {len(available_configs)} config templates in {config_dir}", 'info')
            
            if not available_configs:
                # List directory contents for debugging
                try:
                    contents = os.listdir(config_dir)
                    self.log(f"No config files found in {config_dir}. Contents: {contents}", 'warning')
                    return {
                        'success': False,
                        'error': f'No configuration templates found in {config_dir}. Directory contents: {contents}',
                        'config_dir': str(config_dir),
                        'directory_contents': contents
                    }
                except Exception as e:
                    return {
                        'success': False,
                        'error': f'Error reading config directory: {str(e)}',
                        'config_dir': str(config_dir)
                    }
            
            if progress_callback:
                progress_callback(40, f"🔄 Syncing {len(available_configs)} configs...")
            
            # Perform sync with progress tracking
            configs_processed = []
            configs_failed = []
            
            for i, config_name in enumerate(available_configs):
                current_progress = 40 + ((i + 1) / len(available_configs)) * 50  # 40% to 90%
                
                try:
                    success, message = self.config_manager.sync_single_config(
                        config_name, 
                        force_overwrite=False
                    )
                    
                    if success:
                        configs_processed.append({
                            'name': config_name,
                            'status': 'synced' if 'berhasil' in message else 'skipped',
                            'message': message
                        })
                        self.log(f"✅ {message}", 'info')
                    else:
                        configs_failed.append({
                            'name': config_name,
                            'error': message
                        })
                        self.log(f"❌ {message}", 'error')
                    
                    if progress_callback:
                        status_icon = "✅" if success else "❌"
                        progress_callback(current_progress, f"{status_icon} {config_name}")
                        
                except Exception as e:
                    configs_failed.append({
                        'name': config_name,
                        'error': str(e)
                    })
                    self.log(f"❌ Error syncing {config_name}: {str(e)}", 'error')
                    
                    if progress_callback:
                        progress_callback(current_progress, f"❌ {config_name}")
            
            if progress_callback:
                progress_callback(100, f"✅ Config sync complete")
            
            success_count = len([c for c in configs_processed if c['status'] == 'synced'])
            skip_count = len([c for c in configs_processed if c['status'] == 'skipped'])
            
            overall_success = len(configs_failed) == 0
            
            return {
                'success': overall_success,
                'configs_processed': configs_processed,
                'configs_failed': configs_failed,
                'synced_count': success_count,
                'skipped_count': skip_count,
                'error_count': len(configs_failed),
                'total_count': len(available_configs),
                'message': f'Synced {success_count} configs, skipped {skip_count}, {len(configs_failed)} errors'
            }
            
        except Exception as e:
            self.log(f"Config sync failed: {str(e)}", 'error')
            return {
                'success': False,
                'error': f'Configuration sync failed: {str(e)}'
            }
    
    def check_config_integrity(self) -> Dict[str, Any]:
        """Check integrity of configuration files.
        
        Returns:
            Dictionary with integrity check results
        """
        try:
            available_configs = self.config_manager.discover_repo_configs()
            missing_configs = []
            existing_configs = []
            
            for config_name in available_configs:
                dest_file = self.config_manager.drive_config_dir / config_name
                if dest_file.exists():
                    existing_configs.append(config_name)
                else:
                    missing_configs.append(config_name)
            
            return {
                'all_synced': len(missing_configs) == 0,
                'existing_configs': existing_configs,
                'missing_configs': missing_configs,
                'total_count': len(available_configs),
                'existing_count': len(existing_configs),
                'missing_count': len(missing_configs)
            }
            
        except Exception as e:
            return {
                'error': f'Config integrity check failed: {str(e)}'
            }