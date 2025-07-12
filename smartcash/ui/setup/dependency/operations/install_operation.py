"""
Handler for package installation operations.
"""
from typing import Dict, Any, List, Tuple, Optional, Callable
import asyncio
import time

from smartcash.ui.core.handlers.operation_handler import ProgressLevel
from .base_operation import BaseOperationHandler


class InstallOperationHandler(BaseOperationHandler):
    """Handler for package installation operations."""
    
    def __init__(self, ui_components: Dict[str, Any], config: Dict[str, Any]):
        """Initialize install operation handler.
        
        Args:
            ui_components: Dictionary of UI components
            config: Configuration dictionary with installation settings
        """
        super().__init__('install', ui_components, config)
        self._cancelled = False
    
    async def execute_operation(self) -> Dict[str, Any]:
        """Execute package installation asynchronously.
        
        Returns:
            Dictionary with operation results
        """
        await self.log("🚀 Memulai instalasi paket...", 'info')
        self._cancelled = False
        
        try:
            # Get packages to install
            packages = await self._get_packages_to_process()
            if not packages:
                await self.log("⚠️ Tidak ada paket yang dipilih untuk diinstal", 'warning')
                return {'success': False, 'error': 'Tidak ada paket yang dipilih'}
            
            # Update and save config with current packages
            selected, custom = self._categorize_packages(packages)
            self.config.update({
                'selected_packages': selected,
                'custom_packages': '\n'.join(custom)
            })
            
            # Save config to YAML file
            await self._save_config_to_file(self.config)
            
            # Execute installation
            start_time = time.time()
            results = await self._install_packages(packages)
            duration = time.time() - start_time
            
            # Process results
            success_count = sum(1 for r in results if r.get('success'))
            total = len(results)
            
            if self._cancelled:
                status_msg = f"⏹️ Instalasi dibatalkan: {success_count}/{total} paket berhasil diinstal"
                await self.log(status_msg, 'warning')
            elif success_count == total:
                status_msg = f"✅ Berhasil menginstal {success_count}/{total} paket dalam {duration:.1f} detik"
                await self.log(status_msg, 'success')
            else:
                status_msg = f"⚠️ Berhasil menginstal {success_count}/{total} paket dalam {duration:.1f} detik"
                await self.log(status_msg, 'warning')
            
            return {
                'success': success_count > 0,
                'cancelled': self._cancelled,
                'installed': success_count,
                'total': total,
                'duration': duration,
                'results': results
            }
            
        except asyncio.CancelledError:
            await self.log("⏹️ Instalasi dibatalkan oleh pengguna", 'warning')
            return {'success': False, 'cancelled': True, 'error': 'Dibatalkan oleh pengguna'}
            
        except Exception as e:
            error_msg = f"Gagal melakukan instalasi: {str(e)}"
            await self.log(error_msg, 'error')
            return {'success': False, 'error': error_msg}
    
    def _categorize_packages(self, packages: List[str]) -> Tuple[List[str], List[str]]:
        """Categorize packages into regular and custom (with version specifiers).
        
        Args:
            packages: List of package names/requirements
            
        Returns:
            Tuple of (selected_packages, custom_packages)
        """
        selected: List[str] = []
        custom: List[str] = []
        for pkg in packages:
            if any(c in pkg for c in '><='):
                custom.append(pkg)
            else:
                selected.append(pkg)
        return selected, custom
    
    async def _install_packages(self, packages: List[str]) -> List[Dict[str, Any]]:
        """Install multiple packages sequentially with progress tracking.
        
        Args:
            packages: List of package names/requirements to install
            
        Returns:
            List of installation results with statistics
        """
        if not packages:
            return []
            
        results = []
        total = len(packages)
        
        for i, package in enumerate(packages, 1):
            if self._cancelled:
                break
                
            self._update_progress(
                message=f"Installing package {i}/{total}: {package}",
                current=(i-1) / total * 100,
                level_name='primary'
            )
            
            result = await self._install_single_package(package)
            results.append(result)
            
            # Update progress after completion
            self._update_progress(
                message=f"Completed {i}/{total}: {package}",
                current=i / total * 100,
                level_name='primary'
            )
        
        return results
    
    async def _install_single_package(self, package: str) -> Dict[str, Any]:
        """Install a single package asynchronously.
        
        Args:
            package: Package name or requirement specifier
            
        Returns:
            Dictionary with installation result
        """
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
            # Execute installation with progress tracking
            result = await self._execute_command(
                command,
                timeout=600,  # 10 minute timeout
                progress_callback=lambda p, msg: self._update_progress(
                    message=f"Menginstal {package}... {msg if msg else ''}",
                    current=p,
                    level_name='secondary'
                )
            )
            
            duration = time.time() - start_time
            
            if result['success']:
                await self.log(f"✅ Berhasil menginstal {package} dalam {duration:.1f} detik", 'success')
                return {
                    'success': True,
                    'package': package,
                    'duration': duration,
                    'output': result.get('stdout', ''),
                    'message': f"Berhasil diinstal dalam {duration:.1f} detik"
                }
            else:
                error_msg = result.get('stderr', result.get('stdout', 'Gagal menginstal'))
                await self.log(f"❌ Gagal menginstal {package}: {error_msg}", 'error')
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
            error_msg = f"Kesalahan saat menginstal {package}: {str(e)}"
            await self.log(error_msg, 'error')
            return {
                'success': False,
                'package': package,
                'error': str(e),
                'message': f"Kesalahan: {str(e)}"
            }
    
    async def _save_config_to_file(self, config: Dict[str, Any]) -> None:
        """Save configuration to dependency_config.yaml file."""
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
            
            # Remove successfully installed packages from uninstalled_defaults
            uninstalled_defaults = existing_config.get('uninstalled_defaults', [])
            selected_packages = config.get('selected_packages', [])
            custom_packages = config.get('custom_packages', '')
            
            # Parse installed packages from both selected and custom
            all_installed_packages = set(selected_packages)
            if custom_packages:
                for line in custom_packages.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        package_name = line.split('>')[0].split('<')[0].split('=')[0].strip()
                        all_installed_packages.add(package_name)
            
            # Remove any installed packages from uninstalled_defaults
            updated_uninstalled_defaults = [pkg for pkg in uninstalled_defaults 
                                          if pkg not in all_installed_packages]
            
            # Update with new values
            existing_config.update({
                'selected_packages': selected_packages,
                'custom_packages': custom_packages,
                'uninstalled_defaults': updated_uninstalled_defaults
            })
            
            # Write back to file
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(existing_config, f, default_flow_style=False, allow_unicode=True)
            
            await self.log(f"💾 Configuration saved to {config_path}", 'info')
            
        except Exception as e:
            await self.log(f"❌ Failed to save config: {str(e)}", 'error')

    async def cancel_operation(self) -> None:
        """Cancel the current installation operation."""
        self._cancelled = True
        await self.log("Permintaan pembatalan diterima, menunggu proses saat ini selesai...", 'warning')
    
    def get_operations(self) -> Dict[str, Callable]:
        """Get available operations for this handler.
        
        Returns:
            Dictionary of operation name to callable mapping
        """
        return {
            'execute': self.execute_operation,
            'cancel': self.cancel_operation
        }
