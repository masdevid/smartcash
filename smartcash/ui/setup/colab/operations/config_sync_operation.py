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
                return {
                    'success': False,
                    'error': f'COLAB_DATA_ROOT does not exist: {COLAB_DATA_ROOT}'
                }
            
            if progress_callback:
                progress_callback(25, "📋 Discovering available configs...")
            
            # Discover configs from repo
            available_configs = self.config_manager.discover_repo_configs()
            self.log(f"Found {len(available_configs)} config templates", 'info')
            
            if not available_configs:
                return {
                    'success': False,
                    'error': 'No configuration templates found in repository'
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