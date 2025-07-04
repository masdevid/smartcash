"""
File: smartcash/ui/setup/dependency/operations/update_handler.py
Deskripsi: Handler untuk update operations
"""

import subprocess
import time
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from smartcash.ui.core.handlers.operation_handler import OperationHandler

class UpdateOperationHandler(OperationHandler):
    """Handler untuk update package operations"""
    
    def __init__(self, ui_components: Dict[str, Any], config: Dict[str, Any]):
        super().__init__('update', 'dependency.setup')
        self.ui_components = ui_components
        self.config = config
    
    def execute_operation(self) -> Dict[str, Any]:
        """Execute update operation"""
        try:
            self._update_status("🔄 Memulai pengecekan update packages...")
            
            packages_to_check = self._get_packages_to_check()
            
            if not packages_to_check:
                self._update_status("⚠️ Tidak ada packages untuk dicek update", "warning")
                return {'success': False, 'message': 'Tidak ada packages yang dipilih'}
            
            self._show_progress(f"Checking updates for {len(packages_to_check)} packages", packages_to_check)
            
            # Check for updates
            update_results = self._check_for_updates(packages_to_check)
            packages_with_updates = [r for r in update_results if r.get('has_update')]
            
            if not packages_with_updates:
                self._update_status("✅ Semua packages sudah up-to-date", "success")
                return {'success': True, 'updated': 0, 'up_to_date': len(packages_to_check)}
            
            # Update packages
            self._update_status(f"⬆️ Mengupdate {len(packages_with_updates)} packages...")
            update_results = self._update_packages(packages_with_updates)
            
            success_count = sum(1 for r in update_results if r['success'])
            failed_count = len(update_results) - success_count
            
            if success_count == len(packages_with_updates):
                self._update_status(f"✅ Semua {success_count} packages berhasil diupdate", "success")
                return {'success': True, 'updated': success_count, 'failed': 0}
            else:
                self._update_status(f"⚠️ {success_count} berhasil, {failed_count} gagal", "warning")
                return {'success': False, 'updated': success_count, 'failed': failed_count}
                
        except Exception as e:
            self.handle_error(f"Update operation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _get_packages_to_check(self) -> List[str]:
        """Get packages untuk dicek update"""
        packages = []
        
        # Get selected packages
        selected_packages = self.config.get('selected_packages', [])
        packages.extend(selected_packages)
        
        # Get custom packages
        custom_packages = self.config.get('custom_packages', '')
        if custom_packages:
            for line in custom_packages.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    # Extract package name
                    pkg_name = line.split('==')[0].split('>=')[0].split('<=')[0].split('>')[0].split('<')[0]
                    packages.append(pkg_name.strip())
        
        return packages
    
    def _check_for_updates(self, packages: List[str]) -> List[Dict[str, Any]]:
        """Check packages yang memiliki update"""
        results = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_package = {
                executor.submit(self._check_single_package_update, pkg): pkg 
                for pkg in packages
            }
            
            for future in as_completed(future_to_package):
                package = future_to_package[future]
                try:
                    result = future.result()
                    results.append(result)
                    self._update_progress_step(f"Checked: {package}")
                except Exception as e:
                    results.append({
                        'package': package,
                        'has_update': False,
                        'error': str(e)
                    })
        
        return results
    
    def _check_single_package_update(self, package: str) -> Dict[str, Any]:
        """Check update untuk single package"""
        try:
            # Get current version
            current_version = self._get_installed_version(package)
            if not current_version:
                return {'package': package, 'has_update': False, 'error': 'Package not installed'}
            
            # Get latest version
            latest_version = self._get_latest_version(package)
            if not latest_version:
                return {'package': package, 'has_update': False, 'error': 'Cannot fetch latest version'}
            
            # Compare versions
            has_update = self._compare_versions(current_version, latest_version)
            
            return {
                'package': package,
                'has_update': has_update,
                'current_version': current_version,
                'latest_version': latest_version
            }
            
        except Exception as e:
            return {'package': package, 'has_update': False, 'error': str(e)}
    
    def _get_installed_version(self, package: str) -> str:
        """Get installed version dari package"""
        try:
            result = subprocess.run(
                ['pip', 'show', package],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if line.startswith('Version:'):
                        return line.split(':', 1)[1].strip()
            return ""
            
        except Exception:
            return ""
    
    def _get_latest_version(self, package: str) -> str:
        """Get latest version dari PyPI"""
        try:
            result = subprocess.run(
                ['pip', 'index', 'versions', package],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Parse output untuk get latest version
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'Available versions:' in line:
                        versions = line.split(':', 1)[1].strip()
                        if versions:
                            return versions.split(',')[0].strip()
            return ""
            
        except Exception:
            return ""
    
    def _compare_versions(self, current: str, latest: str) -> bool:
        """Compare versions untuk check update"""
        try:
            # Simple version comparison
            current_parts = [int(x) for x in current.split('.')]
            latest_parts = [int(x) for x in latest.split('.')]
            
            # Pad dengan zeros
            max_len = max(len(current_parts), len(latest_parts))
            current_parts.extend([0] * (max_len - len(current_parts)))
            latest_parts.extend([0] * (max_len - len(latest_parts)))
            
            return latest_parts > current_parts
            
        except Exception:
            return False
    
    def _update_packages(self, packages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Update packages"""
        results = []
        install_options = self.config.get('install_options', {})
        
        with ThreadPoolExecutor(max_workers=install_options.get('parallel_workers', 4)) as executor:
            future_to_package = {
                executor.submit(self._update_single_package, pkg['package']): pkg 
                for pkg in packages
            }
            
            for future in as_completed(future_to_package):
                package_info = future_to_package[future]
                try:
                    result = future.result()
                    results.append(result)
                    self._update_progress_step(f"Updated: {package_info['package']}")
                except Exception as e:
                    results.append({
                        'package': package_info['package'],
                        'success': False,
                        'error': str(e)
                    })
        
        return results
    
    def _update_single_package(self, package: str) -> Dict[str, Any]:
        """Update single package"""
        try:
            self.logger.info(f"⬆️ Updating {package}...")
            
            install_options = self.config.get('install_options', {})
            cmd = [
                install_options.get('python_path', 'python'),
                '-m', 'pip', 'install', '--upgrade', package
            ]
            
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=install_options.get('timeout', 300)
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                self.logger.success(f"✅ {package} updated successfully in {duration:.1f}s")
                return {
                    'package': package,
                    'success': True,
                    'duration': duration,
                    'output': result.stdout
                }
            else:
                self.logger.error(f"❌ Failed to update {package}: {result.stderr}")
                return {
                    'package': package,
                    'success': False,
                    'error': result.stderr,
                    'duration': duration
                }
                
        except Exception as e:
            self.logger.error(f"❌ Error updating {package}: {e}")
            return {
                'package': package,
                'success': False,
                'error': str(e)
            }
    
    def _update_status(self, message: str, status_type: str = "info") -> None:
        """Update status panel"""
        if 'status_panel' in self.ui_components:
            status_panel = self.ui_components['status_panel']
            if hasattr(status_panel, 'update_status'):
                status_panel.update_status(message, status_type)
    
    def _show_progress(self, title: str, steps: List[str]) -> None:
        """Show progress tracker"""
        if 'progress_tracker' in self.ui_components:
            progress_tracker = self.ui_components['progress_tracker']
            if hasattr(progress_tracker, 'show'):
                progress_tracker.show(title, steps)
    
    def _update_progress_step(self, message: str) -> None:
        """Update progress step"""
        if 'progress_tracker' in self.ui_components:
            progress_tracker = self.ui_components['progress_tracker']
            if hasattr(progress_tracker, 'update_step'):
                progress_tracker.update_step(message)