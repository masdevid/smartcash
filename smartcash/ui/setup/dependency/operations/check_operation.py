"""
Pure mixin-based handler for package status check operations.
Uses core mixins instead of inheritance.
"""

from typing import Dict, Any, List
import time
import subprocess

from .base_operation import BaseOperationHandler


class CheckStatusOperationHandler(BaseOperationHandler):
    """
    Pure mixin-based handler for package status check operations.
    
    Uses composition over inheritance with core mixins:
    - LoggingMixin for operation logging
    - ProgressTrackingMixin for progress updates  
    - OperationMixin for operation management
    """
    
    def __init__(self, ui_components: Dict[str, Any], config: Dict[str, Any]):
        """
        Initialize check status operation handler with mixin pattern.
        
        Args:
            ui_components: Dictionary of UI components
            config: Configuration dictionary with check settings
        """
        super().__init__('check_status', ui_components, config)
    
    def execute_operation(self) -> Dict[str, Any]:
        """
        Execute package status check using mixin pattern.
        
        Returns:
            Dictionary with operation results
        """
        self.log("🔍 Memulai pemeriksaan status paket...", 'info')
        self._cancelled = False
        
        try:
            # Get packages to check
            packages = self._get_packages_to_process()
            if not packages:
                self.log("ℹ️ Tidak ada paket yang dipilih untuk dicek", 'info')
                return {
                    'success': True, 
                    'package_status': {},
                    'summary': {'total': 0, 'installed': 0, 'missing': 0}
                }
            
            # Execute status check
            start_time = time.time()
            package_status = self._check_package_status(packages)
            duration = time.time() - start_time
            
            # Create summary
            total = len(packages)
            installed = sum(1 for status in package_status.values() if status.get('installed', False))
            missing = total - installed
            
            summary = {
                'total': total,
                'installed': installed,
                'missing': missing
            }
            
            if self._cancelled:
                status_msg = f"⏹️ Pemeriksaan status dibatalkan"
                self.log(status_msg, 'warning')
            else:
                status_msg = f"✅ Selesai memeriksa {total} paket dalam {duration:.1f} detik"
                self.log(status_msg, 'success')
                self.log(f"📊 Status: {installed}/{total} terinstal, {missing} hilang", 'info')
            
            return {
                'success': not self._cancelled,
                'cancelled': self._cancelled,
                'package_status': package_status,
                'summary': summary,
                'duration': duration
            }
            
        except KeyboardInterrupt:
            self.log("⏹️ Pemeriksaan status dibatalkan oleh pengguna", 'warning')
            return {'success': False, 'cancelled': True, 'error': 'Dibatalkan oleh pengguna'}
            
        except Exception as e:
            error_msg = f"Gagal melakukan pemeriksaan status: {str(e)}"
            self.log(error_msg, 'error')
            return {'success': False, 'error': error_msg}
    
    def _check_package_status(self, packages: List[str]) -> Dict[str, Dict[str, Any]]:
        """Check status of multiple packages with threadpool for faster response."""
        if not packages:
            return {}
            
        package_status = {}
        total = len(packages)
        processed = 0
        
        # Use threadpool for parallel package checking (faster response)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time
        
        # Limit concurrent workers to prevent overwhelming the system
        max_workers = min(len(packages), 5)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all package checks
            futures = {
                executor.submit(self._check_single_package, pkg): pkg
                for pkg in packages
            }
            
            # Process results as they complete
            for future in as_completed(futures):
                if self._cancelled:
                    break
                    
                pkg = futures[future]
                processed += 1
                
                try:
                    status = future.result()
                    package_status[pkg] = status
                    
                    # Update progress
                    progress_value = int((processed / total) * 100)
                    status_icon = "✅" if status.get('installed') else "❌"
                    progress_message = f"Selesai {processed}/{total}: {status_icon} {pkg}"
                    
                    self.update_progress(progress_value, progress_message)
                    
                    # Brief delay for UI visibility
                    time.sleep(0.02)
                    
                except Exception as e:
                    self.log(f"Error checking {pkg}: {str(e)}", 'error')
                    package_status[pkg] = {
                        'installed': False,
                        'version': None,
                        'error': str(e)
                    }
        
        return package_status
    
    def _check_single_package(self, package: str) -> Dict[str, Any]:
        """Check status of a single package."""
        if self._cancelled:
            return {
                'package': package,
                'installed': False,
                'cancelled': True,
                'message': 'Dibatalkan oleh pengguna'
            }
        
        try:
            # Use pip show to check if package is installed
            result = subprocess.run(
                ['pip', 'show', package],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Package is installed, parse version info
                version_info = self._parse_pip_show_output(result.stdout)
                return {
                    'package': package,
                    'installed': True,
                    'version': version_info.get('version', 'Unknown'),
                    'location': version_info.get('location', 'Unknown'),
                    'message': f"Terinstal (v{version_info.get('version', 'Unknown')})"
                }
            else:
                return {
                    'package': package,
                    'installed': False,
                    'message': 'Tidak terinstal'
                }
                
        except subprocess.TimeoutExpired:
            return {
                'package': package,
                'installed': False,
                'error': 'Timeout saat memeriksa paket',
                'message': 'Timeout'
            }
        except Exception as e:
            return {
                'package': package,
                'installed': False,
                'error': str(e),
                'message': f"Error: {str(e)[:50]}"
            }
    
    def _parse_pip_show_output(self, output: str) -> Dict[str, str]:
        """Parse output from pip show command."""
        info = {}
        for line in output.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                info[key.strip().lower()] = value.strip()
        return info