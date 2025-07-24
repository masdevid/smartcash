"""
Pure mixin-based handler for package installation operations.
Uses core mixins instead of inheritance.
"""

from typing import Dict, Any, List
import time

from .base_operation import BaseOperationHandler


class InstallOperationHandler(BaseOperationHandler):
    """
    Pure mixin-based handler for package installation operations.
    
    Uses composition over inheritance with core mixins:
    - LoggingMixin for operation logging
    - ProgressTrackingMixin for progress updates  
    - OperationMixin for operation management
    """
    
    def __init__(self, ui_components: Dict[str, Any], config: Dict[str, Any]):
        """
        Initialize install operation handler with mixin pattern.
        
        Args:
            ui_components: Dictionary of UI components
            config: Configuration dictionary with installation settings
        """
        super().__init__('install', ui_components, config)
    
    def execute_operation(self) -> Dict[str, Any]:
        """
        Execute package installation using mixin pattern.
        
        Returns:
            Dictionary with operation results
        """
        self.log("🚀 Memulai instalasi paket...", 'info')
        self._cancelled = False
        
        try:
            # Get packages to install
            packages = self._get_packages_to_process()
            if not packages:
                self.log("⚠️ Tidak ada paket yang dipilih untuk diinstal", 'warning')
                return {'success': False, 'error': 'Tidak ada paket yang dipilih'}
            
            # Check which packages are missing and need installation
            missing_packages = self._filter_missing_packages(packages)
            if not missing_packages:
                self.log("✅ Semua paket sudah terinstal", 'success')
                return {'success': True, 'message': 'Semua paket sudah terinstal'}
            
            self.log(f"📋 {len(missing_packages)} dari {len(packages)} paket perlu diinstal: {', '.join(missing_packages)}", 'info')
            
            # Update and save config with current packages
            selected, custom = self._categorize_packages(packages)
            self.config.update({
                'selected_packages': selected,
                'custom_packages': '\n'.join(custom)
            })
            
            # Save config to YAML file
            self._save_config_to_file(self.config)
            
            # Execute installation only for missing packages
            start_time = time.time()
            results = self._install_packages(missing_packages)
            duration = time.time() - start_time
            
            # Process results
            success_count = sum(1 for r in results if r.get('success'))
            total = len(results)
            
            if self._cancelled:
                status_msg = f"⏹️ Instalasi dibatalkan: {success_count}/{total} paket berhasil diinstal"
                self.log(status_msg, 'warning')
            elif success_count == total:
                status_msg = f"✅ Berhasil menginstal {success_count}/{total} paket dalam {duration:.1f} detik"
                self.log(status_msg, 'success')
            else:
                status_msg = f"⚠️ Berhasil menginstal {success_count}/{total} paket dalam {duration:.1f} detik"
                self.log(status_msg, 'warning')
            
            return {
                'success': success_count > 0,
                'cancelled': self._cancelled,
                'installed_count': success_count,  # Added for consistency with dependency module
                'total': total,
                'duration': duration,
                'results': results
            }
            
        except KeyboardInterrupt:
            self.log("⏹️ Instalasi dibatalkan oleh pengguna", 'warning')
            return {'success': False, 'cancelled': True, 'error': 'Dibatalkan oleh pengguna'}
            
        except Exception as e:
            error_msg = f"Gagal melakukan instalasi: {str(e)}"
            self.log(error_msg, 'error')
            return {'success': False, 'error': error_msg}
    
    def _filter_missing_packages(self, packages: List[str]) -> List[str]:
        """Filter packages to only include those not already installed.
        
        Args:
            packages: List of package names to check
            
        Returns:
            List of packages that are not installed
        """
        missing_packages = []
        
        self.log("🔍 Memeriksa paket yang sudah terinstal...", 'info')
        
        for package in packages:
            if self._cancelled:
                break
            
            # Check if package is already installed
            if not self._is_package_installed(package):
                missing_packages.append(package)
            else:
                self.log(f"✅ Sudah terinstal: {package}", 'info')
        
        return missing_packages
    
    # _is_package_installed is now inherited from BaseOperationHandler
    
    def _install_packages(self, packages: List[str]) -> List[Dict[str, Any]]:
        """Install multiple packages sequentially with progress tracking."""
        if not packages:
            return []
            
        results = []
        total = len(packages)
        
        for i, package in enumerate(packages, 1):
            if self._cancelled:
                break
                
            # Update progress using mixin method
            progress_value = int((i / total) * 100)
            progress_message = f"Menginstal paket {i}/{total}: {package}"
            
            # Small delay to make progress visible in UI
            import time
            time.sleep(0.05)  # Reduced from 0.1s to 0.05s for better UX
            
            self.update_progress(progress_value, progress_message)
            
            result = self._install_single_package(package)
            results.append(result)
            
            # Show result status
            status = "✅" if result.get('success') else "❌"
            final_progress = int((i / total) * 100)
            final_message = f"Selesai {i}/{total}: {status} {package}"
            
            self.update_progress(final_progress, final_message)
            
            # Brief delay for status visibility
            time.sleep(0.1)
        
        return results
    
    def _install_single_package(self, package: str) -> Dict[str, Any]:
        """Install a single package."""
        if self._cancelled:
            return {
                'package': package,
                'success': False,
                'cancelled': True,
                'message': 'Dibatalkan oleh pengguna'
            }
            
        start_time = time.time()
        
        # Build pip install command
        command = ["pip", "install", "--upgrade"]
        if self.config.get('use_index_url'):
            command.extend(["-i", self.config['index_url']])
        command.append(package)
        
        try:
            # Execute installation
            result = self._execute_command(command)
            
            duration = time.time() - start_time
            
            if result['success']:
                return {
                    'success': True,
                    'package': package,
                    'duration': duration,
                    'message': f"Berhasil diinstal dalam {duration:.1f} detik"
                }
            else:
                error_msg = result.get('stderr', result.get('stdout', 'Gagal menginstal'))[:100]
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