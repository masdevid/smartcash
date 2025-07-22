"""
File: smartcash/ui/setup/dependency/services/package_status_tracker.py
Deskripsi: Service untuk tracking package status dengan real-time updates
"""

import asyncio
import subprocess
from typing import Dict, Any, List, Callable, Optional
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from ..constants import PACKAGE_STATUS, get_status_config

class PackageStatusTracker:
    """Tracker untuk package status dengan real-time updates"""
    
    def __init__(self, config: Dict[str, Any], logger):
        self.config = config
        self.logger = logger
        self._status_cache: Dict[str, Dict[str, Any]] = {}
        self._callbacks: Dict[str, List[Callable]] = {}
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=4)
        
    def register_callback(self, package_name: str, callback: Callable) -> None:
        """Register callback untuk package status updates"""
        with self._lock:
            if package_name not in self._callbacks:
                self._callbacks[package_name] = []
            self._callbacks[package_name].append(callback)
    
    def unregister_callback(self, package_name: str, callback: Callable) -> None:
        """Unregister callback"""
        with self._lock:
            if package_name in self._callbacks:
                try:
                    self._callbacks[package_name].remove(callback)
                    if not self._callbacks[package_name]:
                        del self._callbacks[package_name]
                except ValueError:
                    pass
    
    def get_package_status(self, package_name: str) -> Dict[str, Any]:
        """Get package status dari cache atau check real-time"""
        with self._lock:
            if package_name in self._status_cache:
                return self._status_cache[package_name].copy()
        
        # Check real-time jika tidak ada di cache
        return self._check_package_status_sync(package_name)
    
    def update_package_status(self, package_name: str, status: str, **kwargs) -> None:
        """Update package status dan notify callbacks"""
        status_info = {
            'package': package_name,
            'status': status,
            'timestamp': asyncio.get_event_loop().time(),
            **kwargs
        }
        
        with self._lock:
            self._status_cache[package_name] = status_info
            callbacks = self._callbacks.get(package_name, [])
        
        # Notify callbacks
        for callback in callbacks:
            try:
                callback(status_info)
            except Exception as e:
                self.logger.error(f"❌ Error in status callback for {package_name}: {e}")
    
    def check_package_status_async(self, package_name: str) -> None:
        """Check package status secara async"""
        def _check_and_update():
            try:
                self.update_package_status(package_name, PACKAGE_STATUS['CHECKING'])
                status_info = self._check_package_status_sync(package_name)
                self.update_package_status(package_name, status_info['status'], **status_info)
            except Exception as e:
                self.logger.error(f"❌ Error checking status for {package_name}: {e}")
                self.update_package_status(package_name, PACKAGE_STATUS['ERROR'], error=str(e))
        
        self._executor.submit(_check_and_update)
    
    def _check_package_status_sync(self, package_name: str) -> Dict[str, Any]:
        """Check package status synchronously"""
        try:
            # Check if installed
            result = subprocess.run(
                ['pip', 'show', package_name],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Parse version
                version = None
                for line in result.stdout.split('\n'):
                    if line.startswith('Version:'):
                        version = line.split(':', 1)[1].strip()
                        break
                
                return {
                    'package': package_name,
                    'status': PACKAGE_STATUS['INSTALLED'],
                    'version': version,
                    'installed': True
                }
            else:
                # Check if this is an uninstalled default package
                uninstalled_defaults = self.config.get('uninstalled_defaults', [])
                if package_name in uninstalled_defaults:
                    return {
                        'package': package_name,
                        'status': 'uninstalled_default',  # Use the new status
                        'version': None,
                        'installed': False,
                        'is_default': True
                    }
                else:
                    return {
                        'package': package_name,
                        'status': PACKAGE_STATUS['NOT_INSTALLED'],
                        'version': None,
                        'installed': False
                    }
                
        except Exception as e:
            return {
                'package': package_name,
                'status': PACKAGE_STATUS['ERROR'],
                'error': str(e),
                'installed': False
            }
    
    def check_multiple_packages(self, package_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """Check status untuk multiple packages"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_package = {
                executor.submit(self._check_package_status_sync, pkg): pkg 
                for pkg in package_names
            }
            
            for future in future_to_package:
                package = future_to_package[future]
                try:
                    result = future.result()
                    results[package] = result
                    # Update cache
                    with self._lock:
                        self._status_cache[package] = result
                except Exception as e:
                    results[package] = {
                        'package': package,
                        'status': PACKAGE_STATUS['ERROR'],
                        'error': str(e)
                    }
        
        return results
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get summary dari cached status"""
        with self._lock:
            total = len(self._status_cache)
            installed = sum(1 for s in self._status_cache.values() if s.get('installed', False))
            not_installed = sum(1 for s in self._status_cache.values() if not s.get('installed', False))
            errors = sum(1 for s in self._status_cache.values() if s.get('status') == PACKAGE_STATUS['ERROR'])
            
            return {
                'total': total,
                'installed': installed,
                'not_installed': not_installed,
                'errors': errors,
                'cache_size': total
            }
    
    def clear_cache(self) -> None:
        """Clear status cache"""
        with self._lock:
            self._status_cache.clear()
        self.logger.info("🧹 Package status cache cleared")
    
    def refresh_all_status(self) -> None:
        """Refresh status untuk semua packages dalam cache"""
        with self._lock:
            packages = list(self._status_cache.keys())
        
        for package in packages:
            self.check_package_status_async(package)
    
    def is_package_installed(self, package_name: str) -> bool:
        """Check if package is installed"""
        status = self.get_package_status(package_name)
        return status.get('installed', False)
    
    def get_package_version(self, package_name: str) -> Optional[str]:
        """Get package version if installed"""
        status = self.get_package_status(package_name)
        return status.get('version') if status.get('installed') else None
    
    def get_installed_packages(self) -> List[str]:
        """Get list of installed packages"""
        try:
            result = subprocess.run(
                ['pip', 'list', '--format=freeze'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                packages = []
                for line in result.stdout.strip().split('\n'):
                    if '==' in line:
                        package_name = line.split('==')[0]
                        packages.append(package_name)
                return packages
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting installed packages: {e}")
            return []
    
    def create_compact_status_widget(self, package_name: str) -> 'widgets.HBox':
        """Create a compact status widget for a package.
        
        Args:
            package_name: Name of the package to create status widget for
            
        Returns:
            ipywidgets.HBox containing the status indicator and version
        """
        import ipywidgets as widgets
        from IPython.display import display, clear_output
        
        # Create status indicator
        status_indicator = widgets.HTML(
            value='<div style="width: 12px; height: 12px; border-radius: 50%; background: #ccc; display: inline-block;"></div>',
            layout=widgets.Layout(width='14px', margin='0 8px 0 0')
        )
        
        # Version label
        version_label = widgets.HTML(
            value='',
            layout=widgets.Layout(font_size='0.8em', color='#666')
        )
        
        # Container for status and version
        container = widgets.HBox(
            [status_indicator, version_label],
            layout=widgets.Layout(align_items='center')
        )
        
        # Store references
        container.status_indicator = status_indicator
        container.version_label = version_label
        
        # Update function
        def update_status(status_info: Dict[str, Any]) -> None:
            status = status_info.get('status', '')
            version = status_info.get('version', '')
            
            # Update status indicator
            status_colors = {
                'installed': '#4CAF50',  # Green
                'not_installed': '#f44336',  # Red
                'checking': '#FFC107',  # Yellow
                'error': '#9E9E9E',  # Grey
                'uninstalled_default': '#FF9800',  # Orange
            }
            
            color = status_colors.get(status.lower(), '#9E9E9E')
            status_indicator.value = f'<div style="width: 12px; height: 12px; border-radius: 50%; background: {color}; display: inline-block;" title="{status}"></div>'
            
            # Update version
            version_label.value = f'<span style="font-size: 0.8em; color: #666;">{version}</span>' if version else ''
        
        # Register for updates
        self.register_callback(package_name, update_status)
        
        # Set initial status
        status_info = self.get_package_status(package_name)
        update_status(status_info)
        
        return container
        
    def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            self._executor.shutdown(wait=False)
            with self._lock:
                self._callbacks.clear()
                self._status_cache.clear()
            self.logger.info("🧹 PackageStatusTracker cleaned up")
        except Exception as e:
            self.logger.error(f"❌ Error during cleanup: {e}")