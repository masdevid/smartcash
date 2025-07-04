"""
File: smartcash/ui/setup/dependency/operations/install_handler.py
Deskripsi: Handler untuk install operations
"""

import subprocess
import time
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from smartcash.ui.core.handlers.operation_handler import OperationHandler

class InstallOperationHandler(OperationHandler):
    """Handler untuk install package operations"""
    
    def __init__(self, ui_components: Dict[str, Any], config: Dict[str, Any]):
        super().__init__('install', 'dependency.setup')
        self.ui_components = ui_components
        self.config = config
    
    def execute_operation(self) -> Dict[str, Any]:
        """Execute install operation"""
        try:
            # Update status
            self._update_status("🚀 Memulai instalasi packages...")
            
            # Get packages to install
            packages_to_install = self._get_packages_to_install()
            
            if not packages_to_install:
                self._update_status("⚠️ Tidak ada packages untuk diinstall", "warning")
                return {'success': False, 'message': 'Tidak ada packages yang dipilih'}
            
            # Show progress
            self._show_progress(f"Installing {len(packages_to_install)} packages", packages_to_install)
            
            # Install packages
            results = self._install_packages(packages_to_install)
            
            # Process results
            success_count = sum(1 for r in results if r['success'])
            failed_count = len(results) - success_count
            
            if success_count == len(packages_to_install):
                self._update_status(f"✅ Semua {success_count} packages berhasil diinstall", "success")
                return {'success': True, 'installed': success_count, 'failed': 0}
            else:
                self._update_status(f"⚠️ {success_count} berhasil, {failed_count} gagal", "warning")
                return {'success': False, 'installed': success_count, 'failed': failed_count}
                
        except Exception as e:
            self.handle_error(f"Install operation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _get_packages_to_install(self) -> List[str]:
        """Get packages yang akan diinstall"""
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
                    packages.append(line)
        
        return packages
    
    def _install_packages(self, packages: List[str]) -> List[Dict[str, Any]]:
        """Install packages dengan parallel processing"""
        results = []
        install_options = self.config.get('install_options', {})
        max_workers = install_options.get('parallel_workers', 4)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks
            future_to_package = {
                executor.submit(self._install_single_package, pkg): pkg 
                for pkg in packages
            }
            
            # Process results
            for future in as_completed(future_to_package):
                package = future_to_package[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Update progress
                    self._update_progress_step(f"Completed: {package}")
                    
                except Exception as e:
                    results.append({
                        'package': package,
                        'success': False,
                        'error': str(e)
                    })
                    self.logger.error(f"❌ Error installing {package}: {e}")
        
        return results
    
    def _install_single_package(self, package: str) -> Dict[str, Any]:
        """Install single package"""
        try:
            self.logger.info(f"📥 Installing {package}...")
            
            # Prepare command
            install_options = self.config.get('install_options', {})
            cmd = self._build_install_command(package, install_options)
            
            # Execute installation
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=install_options.get('timeout', 300)
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                self.logger.success(f"✅ {package} installed successfully in {duration:.1f}s")
                return {
                    'package': package,
                    'success': True,
                    'duration': duration,
                    'output': result.stdout
                }
            else:
                self.logger.error(f"❌ Failed to install {package}: {result.stderr}")
                return {
                    'package': package,
                    'success': False,
                    'error': result.stderr,
                    'duration': duration
                }
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"❌ Install timeout for {package}")
            return {
                'package': package,
                'success': False,
                'error': 'Installation timeout'
            }
        except Exception as e:
            self.logger.error(f"❌ Error installing {package}: {e}")
            return {
                'package': package,
                'success': False,
                'error': str(e)
            }
    
    def _build_install_command(self, package: str, options: Dict[str, Any]) -> List[str]:
        """Build install command"""
        cmd = [options.get('python_path', 'python'), '-m', 'pip', 'install']
        
        # Add options
        if options.get('upgrade_strategy') == 'eager':
            cmd.append('--upgrade')
        
        if options.get('force_reinstall'):
            cmd.append('--force-reinstall')
        
        if not options.get('use_cache', True):
            cmd.append('--no-cache-dir')
        
        # Add trusted hosts
        for host in options.get('trusted_hosts', []):
            cmd.extend(['--trusted-host', host])
        
        # Add package
        cmd.append(package)
        
        return cmd
    
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