"""
Pure mixin-based handler for package update operations.
Uses core mixins instead of inheritance.
"""

from typing import Dict, Any, List
import time

from .base_operation import BaseOperationHandler


class UpdateOperationHandler(BaseOperationHandler):
    """
    Pure mixin-based handler for package update operations.
    
    Uses composition over inheritance with core mixins:
    - LoggingMixin for operation logging
    - ProgressTrackingMixin for progress updates  
    - OperationMixin for operation management
    """
    
    def __init__(self, ui_components: Dict[str, Any], config: Dict[str, Any]):
        """
        Initialize update operation handler with mixin pattern.
        
        Args:
            ui_components: Dictionary of UI components
            config: Configuration dictionary with update settings
        """
        super().__init__('update', ui_components, config)
    
    def execute_operation(self) -> Dict[str, Any]:
        """
        Execute package update using mixin pattern.
        
        Returns:
            Dictionary with operation results
        """
        self.log("⬆️ Memulai update paket...", 'info')
        self._cancelled = False
        
        try:
            # Get packages to update
            packages = self._get_packages_to_process()
            if not packages:
                self.log("ℹ️ Tidak ada paket yang dipilih untuk diupdate", 'info')
                return {'success': True, 'updated_count': 0, 'total': 0}
            
            # Execute update
            start_time = time.time()
            results = self._update_packages(packages)
            duration = time.time() - start_time
            
            # Process results
            success_count = sum(1 for r in results if r.get('success'))
            total = len(results)
            
            if self._cancelled:
                status_msg = f"⏹️ Update dibatalkan: {success_count}/{total} paket berhasil diupdate"
                self.log(status_msg, 'warning')
            elif success_count == total:
                status_msg = f"✅ Berhasil mengupdate {success_count}/{total} paket dalam {duration:.1f} detik"
                self.log(status_msg, 'success')
            else:
                status_msg = f"⚠️ Berhasil mengupdate {success_count}/{total} paket dalam {duration:.1f} detik"
                self.log(status_msg, 'warning')
            
            return {
                'success': success_count > 0,
                'cancelled': self._cancelled,
                'updated_count': success_count,  # Added for consistency with dependency module
                'total': total,
                'duration': duration,
                'results': results
            }
            
        except KeyboardInterrupt:
            self.log("⏹️ Update dibatalkan oleh pengguna", 'warning')
            return {'success': False, 'cancelled': True, 'error': 'Dibatalkan oleh pengguna'}
            
        except Exception as e:
            error_msg = f"Gagal melakukan update: {str(e)}"
            self.log(error_msg, 'error')
            return {'success': False, 'error': error_msg}
    
    def _update_packages(self, packages: List[str]) -> List[Dict[str, Any]]:
        """Update multiple packages with threadpool optimization for faster response."""
        if not packages:
            return []
            
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        
        results = []
        total = len(packages)
        completed_count = 0
        lock = threading.Lock()
        
        # Use threadpool for parallel updates (max 3 to avoid pip conflicts)
        max_workers = min(len(packages), 3)
        
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="PackageUpdate") as executor:
            # Submit all update tasks
            future_to_package = {
                executor.submit(self._update_single_package, package): package
                for package in packages
            }
            
            # Process completed updates
            for future in as_completed(future_to_package):
                if self._cancelled:
                    break
                    
                package = future_to_package[future]
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    with lock:
                        completed_count += 1
                        progress = (completed_count / total) * 100
                        
                        # Update progress using mixin method
                        status = "✅" if result.get('success') else "❌"
                        self.update_progress(
                            progress,
                            f"{status} Update paket {completed_count}/{total}: {package}"
                        )
                        
                except Exception as e:
                    results.append({
                        'package': package,
                        'success': False,
                        'error': str(e)
                    })
                    
                    with lock:
                        completed_count += 1
                        progress = (completed_count / total) * 100
                        self.update_progress(
                            progress,
                            f"❌ Update paket {completed_count}/{total}: {package} - {str(e)}"
                        )
        
        return results
    
    def _update_single_package(self, package: str) -> Dict[str, Any]:
        """Update a single package."""
        if self._cancelled:
            return {
                'package': package,
                'success': False,
                'cancelled': True,
                'message': 'Dibatalkan oleh pengguna'
            }
            
        start_time = time.time()
        
        # Build pip install --upgrade command
        command = ["pip", "install", "--upgrade", package]
        
        try:
            # Execute update
            result = self._execute_command(command)
            
            duration = time.time() - start_time
            
            if result['success']:
                return {
                    'success': True,
                    'package': package,
                    'duration': duration,
                    'message': f"Berhasil diupdate dalam {duration:.1f} detik"
                }
            else:
                error_msg = result.get('stderr', result.get('stdout', 'Gagal mengupdate'))[:100]
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