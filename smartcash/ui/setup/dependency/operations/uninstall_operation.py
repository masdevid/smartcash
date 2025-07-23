"""
Pure mixin-based handler for package uninstallation operations.
Uses core mixins instead of inheritance.
"""

from typing import Dict, Any, List
import time

from .base_operation import BaseOperationHandler


class UninstallOperationHandler(BaseOperationHandler):
    """
    Pure mixin-based handler for package uninstallation operations.
    
    Uses composition over inheritance with core mixins:
    - LoggingMixin for operation logging
    - ProgressTrackingMixin for progress updates  
    - OperationMixin for operation management
    """
    
    def __init__(self, ui_components: Dict[str, Any], config: Dict[str, Any]):
        """
        Initialize uninstall operation handler with mixin pattern.
        
        Args:
            ui_components: Dictionary of UI components
            config: Configuration dictionary with uninstallation settings
        """
        super().__init__('uninstall', ui_components, config)
    
    def execute_operation(self) -> Dict[str, Any]:
        """
        Execute package uninstallation using mixin pattern.
        
        Returns:
            Dictionary with operation results
        """
        self.log("🗑️ Memulai uninstal paket...", 'info')
        self._cancelled = False
        
        try:
            # Get packages to uninstall
            packages = self._get_packages_to_process()
            if not packages:
                self.log("⚠️ Tidak ada paket yang dipilih untuk diuninstal", 'warning')
                return {'success': False, 'error': 'Tidak ada paket yang dipilih'}
            
            # Execute uninstallation
            start_time = time.time()
            results = self._uninstall_packages(packages)
            duration = time.time() - start_time
            
            # Process results
            success_count = sum(1 for r in results if r.get('success'))
            total = len(results)
            
            # Update config to track uninstalled packages
            self._update_config_after_uninstall(packages)
            
            if self._cancelled:
                status_msg = f"⏹️ Uninstal dibatalkan: {success_count}/{total} paket berhasil diuninstal"
                self.log(status_msg, 'warning')
            elif success_count == total:
                status_msg = f"✅ Berhasil menguninstal {success_count}/{total} paket dalam {duration:.1f} detik"
                self.log(status_msg, 'success')
            else:
                status_msg = f"⚠️ Berhasil menguninstal {success_count}/{total} paket dalam {duration:.1f} detik"
                self.log(status_msg, 'warning')
            
            return {
                'success': success_count > 0,
                'cancelled': self._cancelled,
                'uninstalled_count': success_count,  # Added for consistency with dependency module
                'total': total,
                'duration': duration,
                'results': results
            }
            
        except KeyboardInterrupt:
            self.log("⏹️ Uninstal dibatalkan oleh pengguna", 'warning')
            return {'success': False, 'cancelled': True, 'error': 'Dibatalkan oleh pengguna'}
            
        except Exception as e:
            error_msg = f"Gagal melakukan uninstal: {str(e)}"
            self.log(error_msg, 'error')
            return {'success': False, 'error': error_msg}
    
    def _uninstall_packages(self, packages: List[str]) -> List[Dict[str, Any]]:
        """Uninstall multiple packages sequentially with progress tracking."""
        if not packages:
            return []
            
        results = []
        total = len(packages)
        
        for i, package in enumerate(packages, 1):
            if self._cancelled:
                break
                
            # Update progress using mixin method
            progress_value = int((i / total) * 100)
            progress_message = f"Menguninstal paket {i}/{total}: {package}"
            
            # Small delay to make progress visible in UI
            import time
            time.sleep(0.05)
            
            self.update_progress(progress_value, progress_message)
            
            result = self._uninstall_single_package(package)
            results.append(result)
            
            # Show result status
            status = "✅" if result.get('success') else "❌"
            final_progress = int((i / total) * 100)
            final_message = f"Selesai {i}/{total}: {status} {package}"
            
            self.update_progress(final_progress, final_message)
            
            # Brief delay for status visibility
            time.sleep(0.1)
        
        return results
    
    def _uninstall_single_package(self, package: str) -> Dict[str, Any]:
        """Uninstall a single package."""
        if self._cancelled:
            return {
                'package': package,
                'success': False,
                'cancelled': True,
                'message': 'Dibatalkan oleh pengguna'
            }
            
        start_time = time.time()
        
        # Build pip uninstall command
        command = ["pip", "uninstall", "-y", package]
        
        try:
            # Execute uninstallation
            result = self._execute_command(command)
            
            duration = time.time() - start_time
            
            if result['success']:
                return {
                    'success': True,
                    'package': package,
                    'duration': duration,
                    'message': f"Berhasil diuninstal dalam {duration:.1f} detik"
                }
            else:
                error_msg = result.get('stderr', result.get('stdout', 'Gagal menguninstal'))[:100]
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
            return {
                'success': False,
                'package': package,
                'error': str(e)[:100],
                'message': f"Kesalahan: {str(e)[:50]}"
            }
    
    def _update_config_after_uninstall(self, packages: List[str]) -> None:
        """Update configuration after uninstalling packages."""
        try:
            import yaml
            import os
            
            config_path = self._get_config_path('dependency_config.yaml')
            
            # Read existing config
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    existing_config = yaml.safe_load(f) or {}
            else:
                existing_config = {}
            
            # Get current lists
            selected_packages = existing_config.get('selected_packages', [])
            uninstalled_defaults = existing_config.get('uninstalled_defaults', [])
            
            # Remove uninstalled packages from selected_packages
            updated_selected_packages = [pkg for pkg in selected_packages if pkg not in packages]
            
            # Add uninstalled packages to uninstalled_defaults if they were default packages
            from ..configs.dependency_defaults import get_default_dependency_config
            default_config = get_default_dependency_config()
            default_categories = default_config.get('package_categories', {})
            
            # Check if any uninstalled packages were defaults
            all_default_packages = set()
            for category in default_categories.values():
                for pkg_info in category.get('packages', []):
                    if pkg_info.get('is_default', False):
                        all_default_packages.add(pkg_info['name'])
            
            # Add uninstalled default packages to uninstalled_defaults
            for pkg in packages:
                if pkg in all_default_packages and pkg not in uninstalled_defaults:
                    uninstalled_defaults.append(pkg)
            
            # Update config
            existing_config.update({
                'selected_packages': updated_selected_packages,
                'uninstalled_defaults': uninstalled_defaults
            })
            
            # Write back to file
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(existing_config, f, default_flow_style=False, allow_unicode=True)
            
            self.log(f"💾 Configuration updated after uninstallation", 'info')
            
        except Exception as e:
            self.log(f"❌ Failed to update config after uninstall: {str(e)}", 'error')