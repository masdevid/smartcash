"""
Handler for package update operations.
"""
from typing import Dict, Any, List, Tuple, Optional
import asyncio
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
    
    async def execute_operation(self) -> Dict[str, Any]:
        """Execute package update asynchronously.
        
        Returns:
            Dictionary with operation results
        """
        await self.log("🔄 Memeriksa pembaruan paket...", 'info')
        self._cancelled = False
        
        try:
            # Get packages to update
            packages = await self._get_packages_to_process()
            if not packages:
                await self.log("ℹ️ Tidak ada paket yang dipilih untuk diperbarui", 'info')
                return {'success': True, 'updated': 0, 'total': 0}
            
            # Update config with current packages
            selected, custom = self._categorize_packages(packages)
            self.config.update({
                'selected_packages': selected,
                'custom_packages': '\n'.join(custom)
            })
            
            # Execute update
            start_time = time.time()
            results = await self._update_packages(packages)
            duration = time.time() - start_time
            
            # Process results
            success_count = sum(1 for r in results if r.get('success'))
            total = len(results)
            
            if self._cancelled:
                status_msg = f"⏹️ Pembaruan dibatalkan: {success_count}/{total} paket berhasil diperbarui"
                await self.log(status_msg, 'warning')
            elif success_count == total:
                status_msg = f"✅ Berhasil memperbarui {success_count}/{total} paket dalam {duration:.1f} detik"
                await self.log(status_msg, 'success')
            else:
                status_msg = f"⚠️ Berhasil memperbarui {success_count}/{total} paket dalam {duration:.1f} detik"
                await self.log(status_msg, 'warning')
            
            return {
                'success': success_count > 0,
                'cancelled': self._cancelled,
                'updated': success_count,
                'total': total,
                'duration': duration,
                'results': results
            }
            
        except asyncio.CancelledError:
            await self.log("⏹️ Pembaruan dibatalkan oleh pengguna", 'warning')
            return {'success': False, 'cancelled': True, 'error': 'Dibatalkan oleh pengguna'}
            
        except Exception as e:
            error_msg = f"Gagal memperbarui paket: {str(e)}"
            await self.log(error_msg, 'error')
            return {'success': False, 'error': error_msg}
    
    def _categorize_packages(self, packages: List[str]) -> Tuple[List[str], List[str]]:
        """Categorize packages into regular and custom (with version specifiers)."""
        selected, custom = [], []
        for pkg in packages:
            (custom if any(c in pkg for c in '><=') else selected).append(pkg)
        return selected, custom
    
    async def _update_packages(self, packages: List[str]) -> List[Dict[str, Any]]:
        """Update multiple packages in parallel with progress tracking.
        
        Args:
            packages: List of package names/requirements to update
            
        Returns:
            List of update results with statistics
        """
        if not packages:
            return []
            
        # Process packages in parallel with progress tracking
        processed_results = await self._process_packages(
            packages,
            self._update_single_package,
            progress_message="Memperbarui paket",
            max_workers=min(4, len(packages))  # Limit concurrent updates
        )
        
        # Extract and return the results
        return [r for r in processed_results['details'] if r.get('status') != 'error']
    
    async def _update_single_package(self, package: str) -> Dict[str, Any]:
        """Update a single package asynchronously.
        
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
            result = await self._execute_command(
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
                await self.log(f"✅ Berhasil memperbarui {package} dalam {duration:.1f} detik", 'success')
                return {
                    'success': True,
                    'package': package,
                    'duration': duration,
                    'output': result.get('stdout', ''),
                    'message': f"Berhasil diperbarui dalam {duration:.1f} detik"
                }
            else:
                error_msg = result.get('stderr', result.get('stdout', 'Gagal memperbarui'))
                await self.log(f"❌ Gagal memperbarui {package}: {error_msg}", 'error')
                return {
                    'success': False,
                    'package': package,
                    'error': error_msg,
                    'message': f"Gagal: {error_msg}"
                }
                
        except asyncio.CancelledError:
            self._cancelled = True
            raise
            
        except Exception as e:
            error_msg = f"Kesalahan saat memperbarui {package}: {str(e)}"
            await self.log(error_msg, 'error')
            return {
                'success': False,
                'package': package,
                'error': str(e),
                'message': f"Kesalahan: {str(e)}"
            }
    
    async def cancel_operation(self) -> None:
        """Cancel the current update operation."""
        self._cancelled = True
        await self.log("Permintaan pembatalan diterima, menunggu proses saat ini selesai...", 'warning')
