"""
File: smartcash/ui/setup/dependency/utils/package_status_tracker.py
Deskripsi: Utility untuk tracking package status dengan real-time updates
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
    
    def create_status_widget(self, package_name: str) -> 'widgets.HTML':
        """Create widget untuk display package status"""
        import ipywidgets as widgets
        
        status_info = self.get_package_status(package_name)
        status_config = get_status_config(status_info.get('status', PACKAGE_STATUS['NOT_INSTALLED']))
        
        widget = widgets.HTML(
            value=f"""
            <div style="
                display: flex;
                align-items: center;
                gap: 8px;
                padding: 5px 10px;
                border-radius: 15px;
                background: {status_config['bg_color']};
                border: 1px solid {status_config['color']};
                font-size: 12px;
                font-weight: 500;
            ">
                <span style="font-size: 14px;">{status_config['icon']}</span>
                <span style="color: {status_config['color']};">{status_config['text']}</span>
            </div>
            """
        )
        
        # Register callback untuk auto-update
        def update_widget(status_info):
            config = get_status_config(status_info.get('status', PACKAGE_STATUS['NOT_INSTALLED']))
            widget.value = f"""
            <div style="
                display: flex;
                align-items: center;
                gap: 8px;
                padding: 5px 10px;
                border-radius: 15px;
                background: {config['bg_color']};
                border: 1px solid {config['color']};
                font-size: 12px;
                font-weight: 500;
            ">
                <span style="font-size: 14px;">{config['icon']}</span>
                <span style="color: {config['color']};">{config['text']}</span>
            </div>
            """
        
        self.register_callback(package_name, update_widget)
        
        return widget
    
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