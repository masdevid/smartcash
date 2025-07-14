"""
Handler for package uninstallation operations.
"""
from typing import Dict, Any, List, Optional, Callable
import re
import time

# Progress level import removed - not used in synchronous version
from .base_operation import BaseOperationHandler


class UninstallOperationHandler(BaseOperationHandler):
    """Handler for package uninstallation operations."""
    
    def __init__(self, ui_components: Dict[str, Any], config: Dict[str, Any]):
        """Initialize uninstall operation handler.
        
        Args:
            ui_components: Dictionary of UI components
            config: Configuration dictionary with uninstallation settings
        """
        super().__init__('uninstall', ui_components, config)
        self._cancelled = False
    
    def execute_operation(self) -> Dict[str, Any]:
        """Execute package uninstallation synchronously.
        
        Returns:
            Dictionary with operation results
        """
        self.log("🗑️ Memulai penghapusan paket...", 'info')
        self._cancelled = False
        
        try:
            # Get packages to uninstall
            packages = self._get_packages_to_process()
            if not packages:
                self.log("ℹ️ Tidak ada paket yang dipilih untuk dihapus", 'info')
                return {'success': True, 'uninstalled': 0, 'total': 0}
            
            # Execute uninstallation
            start_time = time.time()
            results = self._uninstall_packages(packages)
            duration = time.time() - start_time
            
            # Process results
            success_count = sum(1 for r in results if r.get('success'))
            total = len(results)
            
            # Update config by removing successfully uninstalled packages
            if success_count > 0:
                self._update_config_after_uninstall(results)
            
            if self._cancelled:
                status_msg = f"⏹️ Penghapusan dibatalkan: {success_count}/{total} paket berhasil dihapus"
                self.log(status_msg, 'warning')
            elif success_count == total:
                status_msg = f"✅ Berhasil menghapus {success_count}/{total} paket dalam {duration:.1f} detik"
                self.log(status_msg, 'success')
            else:
                status_msg = f"⚠️ Berhasil menghapus {success_count}/{total} paket dalam {duration:.1f} detik"
                self.log(status_msg, 'warning')
            
            return {
                'success': success_count > 0,
                'cancelled': self._cancelled,
                'uninstalled': success_count,
                'total': total,
                'duration': duration,
                'results': results
            }
            
        except KeyboardInterrupt:
            self.log("⏹️ Penghapusan dibatalkan oleh pengguna", 'warning')
            return {'success': False, 'cancelled': True, 'error': 'Dibatalkan oleh pengguna'}
            
        except Exception as e:
            error_msg = f"Gagal melakukan penghapusan: {str(e)}"
            self.log(error_msg, 'error')
            return {'success': False, 'error': error_msg}
    
    def _uninstall_packages(self, packages: List[str]) -> List[Dict[str, Any]]:
        """Uninstall multiple packages sequentially with progress tracking.
        
        Args:
            packages: List of package names to uninstall
            
        Returns:
            List of uninstallation results
        """
        if not packages:
            return []
            
        # Process packages sequentially with progress tracking
        processed_results = self._process_packages(
            packages,
            self._uninstall_single_package,
            progress_message="Menghapus paket",
            max_workers=min(4, len(packages))  # Limit concurrent uninstallations
        )
        
        # Extract and return the results
        return [r for r in processed_results['details'] if r.get('status') != 'error']
    
    def _uninstall_single_package(self, package: str) -> Dict[str, Any]:
        """Uninstall a single package synchronously.
        
        Args:
            package: Package name to uninstall
            
        Returns:
            Dictionary with uninstallation result
        """
        if self._cancelled:
            return {
                'package': package,
                'success': False,
                'cancelled': True,
                'message': 'Dibatalkan oleh pengguna'
            }
            
        start_time = time.time()
        
        # For packages with version specifiers, extract just the package name
        pkg_name = package.split('>')[0].split('<')[0].split('=')[0].strip()
        
        # Build pip uninstall command
        command = ["pip", "uninstall", "--yes", pkg_name]
        
        try:
            # Execute uninstallation with progress tracking
            result = self._execute_command(
                command,
                timeout=300,  # 5 minute timeout
                progress_callback=lambda p, msg: self._update_progress(
                    message=f"Menghapus {pkg_name}... {msg if msg else ''}",
                    current=p,
                    level_name='secondary'
                )
            )
            
            duration = time.time() - start_time
            
            # Check if uninstallation was successful or if package was not installed
            is_success = result['success'] or 'not installed' in result.get('stderr', '')
            
            if is_success:
                message = "Tidak terpasang" if 'not installed' in result.get('stderr', '') else f"Berhasil dihapus dalam {duration:.1f} detik"
                self.log(f"✅ {pkg_name} {message.lower()}", 'success')
                return {
                    'success': True,
                    'package': pkg_name,
                    'duration': duration,
                    'output': result.get('stdout', ''),
                    'message': message
                }
            else:
                error_msg = result.get('stderr', result.get('stdout', 'Gagal menghapus'))
                self.log(f"❌ Gagal menghapus {pkg_name}: {error_msg}", 'error')
                return {
                    'success': False,
                    'package': pkg_name,
                    'error': error_msg,
                    'message': f"Gagal: {error_msg}"
                }
                
        except KeyboardInterrupt:
            self._cancelled = True
            raise
            
        except Exception as e:
            error_msg = f"Kesalahan saat menghapus {pkg_name}: {str(e)}"
            self.log(error_msg, 'error')
            return {
                'success': False,
                'package': pkg_name,
                'error': str(e),
                'message': f"Kesalahan: {str(e)}"
            }
    
    def _update_config_after_uninstall(self, results: List[Dict[str, Any]]) -> None:
        """Update config by removing successfully uninstalled packages."""
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
            
            # Get successfully uninstalled packages
            uninstalled_packages = [r['package'] for r in results if r.get('success')]
            
            # Get default packages to check which ones should be preserved
            from ..configs.dependency_defaults import get_default_package_categories
            default_categories = get_default_package_categories()
            default_packages = set()
            for category in default_categories.values():
                for pkg in category.get('packages', []):
                    if pkg.get('is_default', False):
                        default_packages.add(pkg['name'])
            
            # Only remove non-default packages from selected_packages
            selected_packages = existing_config.get('selected_packages', [])
            updated_selected = [pkg for pkg in selected_packages 
                              if pkg not in uninstalled_packages or pkg in default_packages]
            
            # Add uninstalled default packages to uninstalled_defaults list
            uninstalled_defaults = existing_config.get('uninstalled_defaults', [])
            for pkg in uninstalled_packages:
                if pkg in default_packages and pkg not in uninstalled_defaults:
                    uninstalled_defaults.append(pkg)
            
            # Remove from custom packages (these can be fully removed)
            custom_packages = existing_config.get('custom_packages', '')
            if custom_packages:
                custom_lines = custom_packages.split('\n')
                updated_custom_lines = []
                for line in custom_lines:
                    line = line.strip()
                    if line and not any(pkg in line for pkg in uninstalled_packages):
                        updated_custom_lines.append(line)
                updated_custom = '\n'.join(updated_custom_lines)
            else:
                updated_custom = ''
            
            # Update config
            existing_config.update({
                'selected_packages': updated_selected,
                'custom_packages': updated_custom,
                'uninstalled_defaults': uninstalled_defaults
            })
            
            # Write back to file
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(existing_config, f, default_flow_style=False, allow_unicode=True)
            
            self.log(f"💾 Configuration updated after uninstalling {len(uninstalled_packages)} packages", 'info')
            
        except Exception as e:
            self.log(f"❌ Failed to update config after uninstall: {str(e)}", 'error')

    def cancel_operation(self) -> None:
        """Cancel the current uninstallation operation."""
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
