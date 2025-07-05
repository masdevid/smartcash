"""
Handler for package uninstallation operations.
"""
from typing import Dict, Any, List, Optional
import asyncio
import re
import time

from smartcash.ui.core.handlers.operation_handler import ProgressLevel
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
    
    async def execute_operation(self) -> Dict[str, Any]:
        """Execute package uninstallation asynchronously.
        
        Returns:
            Dictionary with operation results
        """
        await self.log("🗑️ Memulai penghapusan paket...", 'info')
        self._cancelled = False
        
        try:
            # Get packages to uninstall
            packages = await self._get_packages_to_process()
            if not packages:
                await self.log("ℹ️ Tidak ada paket yang dipilih untuk dihapus", 'info')
                return {'success': True, 'uninstalled': 0, 'total': 0}
            
            # Execute uninstallation
            start_time = time.time()
            results = await self._uninstall_packages(packages)
            duration = time.time() - start_time
            
            # Process results
            success_count = sum(1 for r in results if r.get('success'))
            total = len(results)
            
            if self._cancelled:
                status_msg = f"⏹️ Penghapusan dibatalkan: {success_count}/{total} paket berhasil dihapus"
                await self.log(status_msg, 'warning')
            elif success_count == total:
                status_msg = f"✅ Berhasil menghapus {success_count}/{total} paket dalam {duration:.1f} detik"
                await self.log(status_msg, 'success')
            else:
                status_msg = f"⚠️ Berhasil menghapus {success_count}/{total} paket dalam {duration:.1f} detik"
                await self.log(status_msg, 'warning')
            
            return {
                'success': success_count > 0,
                'cancelled': self._cancelled,
                'uninstalled': success_count,
                'total': total,
                'duration': duration,
                'results': results
            }
            
        except asyncio.CancelledError:
            await self.log("⏹️ Penghapusan dibatalkan oleh pengguna", 'warning')
            return {'success': False, 'cancelled': True, 'error': 'Dibatalkan oleh pengguna'}
            
        except Exception as e:
            error_msg = f"Gagal melakukan penghapusan: {str(e)}"
            await self.log(error_msg, 'error')
            return {'success': False, 'error': error_msg}
    
    async def _uninstall_packages(self, packages: List[str]) -> List[Dict[str, Any]]:
        """Uninstall multiple packages in parallel with progress tracking.
        
        Args:
            packages: List of package names to uninstall
            
        Returns:
            List of uninstallation results
        """
        if not packages:
            return []
            
        # Process packages in parallel with progress tracking
        processed_results = await self._process_packages(
            packages,
            self._uninstall_single_package,
            progress_message="Menghapus paket",
            max_workers=min(4, len(packages))  # Limit concurrent uninstallations
        )
        
        # Extract and return the results
        return [r for r in processed_results['details'] if r.get('status') != 'error']
    
    async def _uninstall_single_package(self, package: str) -> Dict[str, Any]:
        """Uninstall a single package asynchronously.
        
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
            result = await self._execute_command(
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
                await self.log(f"✅ {pkg_name} {message.lower()}", 'success')
                return {
                    'success': True,
                    'package': pkg_name,
                    'duration': duration,
                    'output': result.get('stdout', ''),
                    'message': message
                }
            else:
                error_msg = result.get('stderr', result.get('stdout', 'Gagal menghapus'))
                await self.log(f"❌ Gagal menghapus {pkg_name}: {error_msg}", 'error')
                return {
                    'success': False,
                    'package': pkg_name,
                    'error': error_msg,
                    'message': f"Gagal: {error_msg}"
                }
                
        except asyncio.CancelledError:
            self._cancelled = True
            raise
            
        except Exception as e:
            error_msg = f"Kesalahan saat menghapus {pkg_name}: {str(e)}"
            await self.log(error_msg, 'error')
            return {
                'success': False,
                'package': pkg_name,
                'error': str(e),
                'message': f"Kesalahan: {str(e)}"
            }
    
    async def cancel_operation(self) -> None:
        """Cancel the current uninstallation operation."""
        self._cancelled = True
        await self.log("Permintaan pembatalan diterima, menunggu proses saat ini selesai...", 'warning')
