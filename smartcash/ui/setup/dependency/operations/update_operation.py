"""
Handler for package update operations.
"""
from typing import Dict, Any, List, Tuple, Optional, Callable
import time

from smartcash.ui.core.handlers.operation_handler import ProgressLevel
from .base_operation import BaseOperationHandler


class UpdateOperationHandler(BaseOperationHandler):
    """Handler for package update operations."""
    
    def __init__(self, ui_components: Dict[str, Any], config: Dict[str, Any]):
        """Initialize update operation handler.
        
        Args:
            ui_components: Dictionary of UI components
            config: Configuration dictionary with update settings
        """
        super().__init__('update', ui_components, config)
        self._cancelled = False
    
    def execute_operation(self) -> Dict[str, Any]:
        """Execute package update.
        
        Returns:
            Dictionary with operation results
        """
        self.log("🔄 Memeriksa pembaruan paket...", 'info')
        self._cancelled = False
        
        try:
            # Get packages to update
            packages = self._get_packages_to_process()
            if not packages:
                self.log("ℹ️ Tidak ada paket yang dipilih untuk diperbarui", 'info')
                return {'success': True, 'updated': 0, 'total': 0}
            
            # Update config with current packages
            selected, custom = self._categorize_packages(packages)
            self.config.update({
                'selected_packages': selected,
                'custom_packages': '\n'.join(custom)
            })
            
            # Save config to YAML file
            self._save_config_to_file(self.config)
            
            # Execute update
            start_time = time.time()
            results = self._update_packages(packages)
            duration = time.time() - start_time
            
            # Process results
            success_count = sum(1 for r in results if r.get('success'))
            total = len(results)
            
            if self._cancelled:
                status_msg = f"⏹️ Pembaruan dibatalkan: {success_count}/{total} paket berhasil diperbarui"
                self.log(status_msg, 'warning')
            elif success_count == total:
                status_msg = f"✅ Berhasil memperbarui {success_count}/{total} paket dalam {duration:.1f} detik"
                self.log(status_msg, 'success')
            else:
                status_msg = f"⚠️ Berhasil memperbarui {success_count}/{total} paket dalam {duration:.1f} detik"
                self.log(status_msg, 'warning')
            
            return {
                'success': success_count > 0,
                'cancelled': self._cancelled,
                'updated': success_count,
                'total': total,
                'duration': duration,
                'results': results
            }
            
        except KeyboardInterrupt:
            self.log("⏹️ Pembaruan dibatalkan oleh pengguna", 'warning')
            return {'success': False, 'cancelled': True, 'error': 'Dibatalkan oleh pengguna'}
            
        except Exception as e:
            error_msg = f"Gagal memperbarui paket: {str(e)}"
            self.log(error_msg, 'error')
            return {'success': False, 'error': error_msg}
    
    def _categorize_packages(self, packages: List[str]) -> Tuple[List[str], List[str]]:
        """Categorize packages into regular and custom (with version specifiers)."""
        selected, custom = [], []
        for pkg in packages:
            (custom if any(c in pkg for c in '><=') else selected).append(pkg)
        return selected, custom
    
    def _update_packages(self, packages: List[str]) -> List[Dict[str, Any]]:
        """Update multiple packages in parallel with progress tracking.
        
        Args:
            packages: List of package names/requirements to update
            
        Returns:
            List of update results with statistics
        """
        if not packages:
            return []
            
        # Process packages in parallel with progress tracking
        processed_results = self._process_packages(
            packages,
            self._update_single_package,
            progress_message="Memperbarui paket",
            max_workers=min(4, len(packages))  # Limit concurrent updates
        )
        
        # Extract and return the results
        return [r for r in processed_results['details'] if r.get('status') != 'error']
    
    def _update_single_package(self, package: str) -> Dict[str, Any]:
        """Update a single package.
        
        Args:
            package: Package name or requirement specifier to update
            
        Returns:
            Dictionary with update result
        """
        if self._cancelled:
            return {
                'package': package,
                'success': False,
                'cancelled': True,
                'message': 'Dibatalkan oleh pengguna'
            }
            
        start_time = time.time()
        
        # Build pip install --upgrade command
        command = ["pip", "install", "--upgrade"]
        if self.config.get('use_index_url'):
            command.extend(["-i", self.config['index_url']])
        command.append(package)
        
        try:
            # Execute update with progress tracking
            result = self._execute_command(
                command,
                timeout=600,  # 10 minute timeout
                progress_callback=lambda p, msg: self._update_progress(
                    message=f"Memperbarui {package}... {msg if msg else ''}",
                    current=p,
                    level_name='secondary'
                )
            )
            
            duration = time.time() - start_time
            
            if result['success']:
                self.log(f"✅ Berhasil memperbarui {package} dalam {duration:.1f} detik", 'success')
                return {
                    'success': True,
                    'package': package,
                    'duration': duration,
                    'output': result.get('stdout', ''),
                    'message': f"Berhasil diperbarui dalam {duration:.1f} detik"
                }
            else:
                error_msg = result.get('stderr', result.get('stdout', 'Gagal memperbarui'))
                self.log(f"❌ Gagal memperbarui {package}: {error_msg}", 'error')
                return {
                    'success': False,
                    'package': package,
                    'error': error_msg,
                    'message': f"Gagal: {error_msg}"
                }
                
        except KeyboardInterrupt:
            self._cancelled = True
            raise
            
        except Exception as e:
            error_msg = f"Kesalahan saat memperbarui {package}: {str(e)}"
            self.log(error_msg, 'error')
            return {
                'success': False,
                'package': package,
                'error': str(e),
                'message': f"Kesalahan: {str(e)}"
            }
    
    def _save_config_to_file(self, config: Dict[str, Any]) -> None:
        """Save configuration to dependency_config.yaml file."""
        try:
            import yaml
            import os
            
            config_path = os.path.join('/Users/masdevid/Projects/smartcash', 'configs', 'dependency_config.yaml')
            
            # Read existing config
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    existing_config = yaml.safe_load(f) or {}
            else:
                existing_config = {}
            
            # Update with new values
            existing_config.update({
                'selected_packages': config.get('selected_packages', []),
                'custom_packages': config.get('custom_packages', '')
            })
            
            # Write back to file
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(existing_config, f, default_flow_style=False, allow_unicode=True)
            
            self.log(f"💾 Configuration saved to {config_path}", 'info')
            
        except Exception as e:
            self.log(f"❌ Failed to save config: {str(e)}", 'error')

    def cancel_operation(self) -> None:
        """Cancel the current update operation."""
        self._cancelled = True
        self.log("Permintaan pembatalan diterima, menunggu proses saat ini selesai...", 'warning')
    
    def get_operations(self) -> Dict[str, Callable]:
        """Get available operations for this handler.
        
        Returns:
            Dictionary of operation name to callable mapping
        """
        return {
            'execute': self.execute_operation,
            'cancel': self.cancel_operation
        }
