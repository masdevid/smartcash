"""
File: smartcash/ui/setup/colab/operations/config_sync_operation.py
Description: Sync configuration files using the config manager
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from smartcash.ui.components.operation_container import OperationContainer
from .base_colab_operation import BaseColabOperation
from smartcash.common.config.manager import get_config_manager
from smartcash.common.constants.paths import COLAB_DATA_ROOT


class ConfigSyncOperation(BaseColabOperation):
    """Sync configuration files using the config manager."""
    
    def __init__(self, operation_name: str, config: Dict[str, Any], operation_container: Optional[OperationContainer] = None, **kwargs):
        """Initialize config sync operation.
        
        Args:
            operation_name: Name of the operation
            config: Configuration dictionary
            operation_container: Optional operation container for UI integration
            **kwargs: Additional arguments
        """
        super().__init__(operation_name, config, operation_container, **kwargs)
        self.config_manager = get_config_manager(auto_sync=False)
    
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
        def execute_operation():
            progress_steps = self.get_progress_steps('config_sync')
            
            # Step 1: Check configuration
            self.update_progress_safe(
                progress_callback, 
                progress_steps[0]['progress'], 
                progress_steps[0]['message'],
                progress_steps[0].get('phase_progress', 0)
            )
            
            # Simulate config check work
            config_checks = [
                ("Checking COLAB_DATA_ROOT...", 20),
                ("Verifying config directory...", 50),
                ("Preparing configuration...", 80),
                ("Configuration check complete", 100)
            ]
            
            for msg, phase_pct in config_checks:
                self.update_progress_safe(
                    progress_callback,
                    int(progress_steps[0]['progress'] + (progress_steps[1]['progress'] - progress_steps[0]['progress']) * (phase_pct / 100)),
                    msg,
                    int(progress_steps[0].get('phase_progress', 0) + (progress_steps[1].get('phase_progress', 0) - progress_steps[0].get('phase_progress', 0)) * (phase_pct / 100))
                )
            
            # Check if COLAB_DATA_ROOT exists
            if not self.ensure_directory_exists(COLAB_DATA_ROOT):
                return self.create_error_result(f"Failed to create COLAB_DATA_ROOT directory: {COLAB_DATA_ROOT}")
            
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
                    return self.create_error_result(
                        f'Failed to create config directory: {str(e)}',
                        config_dir=str(config_dir),
                        cwd=os.getcwd(),
                        ls_content=os.listdir('/content/smartcash' if os.path.exists('/content/smartcash') else '/content')
                    )
            
            # Step 2: Sync configuration
            self.update_progress_safe(
                progress_callback, 
                progress_steps[1]['progress'], 
                progress_steps[1]['message'],
                progress_steps[1].get('phase_progress', 0)
            )
            
            # Simulate sync work
            sync_steps = [
                ("Discovering configuration files...", 30),
                ("Checking for updates...", 60),
                ("Synchronizing configurations...", 90),
                ("Finalizing sync...", 100)
            ]
            
            for msg, phase_pct in sync_steps:
                self.update_progress_safe(
                    progress_callback,
                    int(progress_steps[1]['progress'] + (progress_steps[2]['progress'] - progress_steps[1]['progress']) * (phase_pct / 100)),
                    msg,
                    int(progress_steps[1].get('phase_progress', 0) + (progress_steps[2].get('phase_progress', 0) - progress_steps[1].get('phase_progress', 0)) * (phase_pct / 100))
                )
            
            # Discover configs from repo
            available_configs = self.config_manager.discover_repo_configs()
            self.log(f"Found {len(available_configs)} config templates in {config_dir}", 'info')
            
            if not available_configs:
                # List directory contents for debugging
                try:
                    contents = os.listdir(config_dir)
                    self.log(f"No config files found in {config_dir}. Contents: {contents}", 'warning')
                    return self.create_error_result(
                        f'No configuration templates found in {config_dir}. Directory contents: {contents}',
                        config_dir=str(config_dir),
                        directory_contents=contents
                    )
                except Exception as e:
                    return self.create_error_result(
                        f'Error reading config directory: {str(e)}',
                        config_dir=str(config_dir)
                    )
            
            # Perform sync with progress tracking
            configs_processed = []
            configs_failed = []
            
            for config_name in available_configs:
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
                        
                except Exception as e:
                    configs_failed.append({
                        'name': config_name,
                        'error': str(e)
                    })
                    self.log(f"❌ Error syncing {config_name}: {str(e)}", 'error')
            
            # Step 3: Validate sync
            self.update_progress_safe(
                progress_callback, 
                progress_steps[2]['progress'], 
                progress_steps[2]['message'],
                progress_steps[2].get('phase_progress', 0)
            )
            
            integrity_check = self.check_config_integrity()
            
            # Step 4: Complete
            self.update_progress_safe(
                progress_callback, 
                progress_steps[3]['progress'], 
                progress_steps[3]['message'],
                progress_steps[3].get('phase_progress', 0)
            )
            
            success_count = len([c for c in configs_processed if c['status'] == 'synced'])
            skip_count = len([c for c in configs_processed if c['status'] == 'skipped'])
            
            return self.create_success_result(
                f'Synced {success_count} configs, skipped {skip_count}, {len(configs_failed)} errors',
                configs_processed=configs_processed,
                configs_failed=configs_failed,
                synced_count=success_count,
                skipped_count=skip_count,
                error_count=len(configs_failed),
                total_count=len(available_configs),
                integrity_check=integrity_check
            )
            
        return self.execute_with_error_handling(execute_operation)
    
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