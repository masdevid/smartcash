"""
File: smartcash/ui/setup/dependency/operations/uninstall_handler.py
Deskripsi: Handler untuk uninstall operations
"""

import subprocess
import time
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from smartcash.ui.core.handlers.operation_handler import OperationHandler

class UninstallOperationHandler(OperationHandler):
    """Handler untuk uninstall package operations"""
    
    def __init__(self, ui_components: Dict[str, Any], config: Dict[str, Any]):
        super().__init__('uninstall', 'dependency.setup')
        self.ui_components = ui_components
        self.config = config
    
    def execute_operation(self) -> Dict[str, Any]:
        """Execute uninstall operation"""
        try:
            self._update_status("🗑️ Memulai uninstall packages...")
            
            packages_to_uninstall = self._get_packages_to_uninstall()
            
            if not packages_to_uninstall:
                self._update_status("⚠️ Tidak ada packages untuk diuninstall", "warning")
                return {'success': False, 'message': 'Tidak ada packages yang dipilih'}
            
            # Check installed packages
            installed_packages = self._filter_installed_packages(packages_to_uninstall)
            
            if not installed_packages:
                self._update_status("ℹ️ Tidak ada packages yang terinstall", "info")
                return {'success': True, 'uninstalled': 0, 'not_installed': len(packages_to_uninstall)}
            
            self._show_progress(f"Uninstalling {len(installed_packages)} packages", installed_packages)
            
            # Uninstall packages
            results = self._uninstall_packages(installed_packages)
            
            success_count = sum(1 for r in results if r['success'])
            failed_count = len(results) - success_count
            
            if success_count == len(installed_packages):
                self._update_status(f"✅ Semua {success_count} packages berhasil diuninstall", "success")
                return {'success': True, 'uninstalled': success_count, 'failed': 0}
            else:
                self._update_status(f"⚠️ {success_count} berhasil, {failed_count} gagal", "warning")
                return {'success': False, 'uninstalled': success_count, 'failed': failed_count}
                
        except Exception as e:
            self.handle_error(f"Uninstall operation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _get_packages_to_uninstall(self) -> List[str]:
        """Get packages untuk diuninstall"""
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
    
    def _filter_installed_packages(self, packages: List[str]) -> List[str]:
        """Filter packages yang sudah terinstall"""
        installed = []
        
        for package in packages:
            if self._is_package_installed(package):
                installed.append(package)
            else:
                self.logger.info(f"ℹ️ Package {package} tidak terinstall")
        
        return installed
    
    def _is_package_installed(self, package: str) -> bool:
        """Check apakah package sudah terinstall"""
        try:
            result = subprocess.run(
                ['pip', 'show', package],
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def _uninstall_packages(self, packages: List[str]) -> List[Dict[str, Any]]:
        """Uninstall packages dengan parallel processing"""
        results = []
        install_options = self.config.get('install_options', {})
        max_workers = install_options.get('parallel_workers', 4)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_package = {
                executor.submit(self._uninstall_single_package, pkg): pkg 
                for pkg in packages
            }
            
            for future in as_completed(future_to_package):
                package = future_to_package[future]
                try:
                    result = future.result()
                    results.append(result)
                    self._update_progress_step(f"Uninstalled: {package}")
                except Exception as e:
                    results.append({
                        'package': package,
                        'success': False,
                        'error': str(e)
                    })
        
        return results
    
    def _uninstall_single_package(self, package: str) -> Dict[str, Any]:
        """Uninstall single package"""
        try:
            self.logger.info(f"🗑️ Uninstalling {package}...")
            
            install_options = self.config.get('install_options', {})
            cmd = [
                install_options.get('python_path', 'python'),
                '-m', 'pip', 'uninstall', '-y', package
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
                self.logger.success(f"✅ {package} uninstalled successfully in {duration:.1f}s")
                return {
                    'package': package,
                    'success': True,
                    'duration': duration,
                    'output': result.stdout
                }
            else:
                self.logger.error(f"❌ Failed to uninstall {package}: {result.stderr}")
                return {
                    'package': package,
                    'success': False,
                    'error': result.stderr,
                    'duration': duration
                }
                
        except Exception as e:
            self.logger.error(f"❌ Error uninstalling {package}: {e}")
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