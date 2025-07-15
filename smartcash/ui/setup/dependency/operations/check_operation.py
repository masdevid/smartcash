"""
Handler for package status check operations.
"""
from typing import Dict, Any, List, Optional, Tuple, Set, Callable
import re
import time
from datetime import datetime

from smartcash.ui.core.handlers.operation_handler import ProgressLevel
from .base_operation import BaseOperationHandler


class CheckStatusOperationHandler(BaseOperationHandler):
    """Handler for package status check operations."""
    
    def __init__(self, ui_components: Dict[str, Any], config: Dict[str, Any]):
        """Initialize status check operation handler.
        
        Args:
            ui_components: Dictionary of UI components
            config: Configuration dictionary with package information
        """
        super().__init__('check_status', ui_components, config)
        self._status_cache: Dict[str, Dict[str, Any]] = {}
    
    def execute_operation(self) -> Dict[str, Any]:
        """Execute package status check.
        
        Returns:
            Dictionary with status check results
        """
        self.log("🔍 Memeriksa status paket...", 'info')
        
        # Get packages to check
        packages = self._get_packages_to_check()
        if not packages:
            self.log("ℹ️ Tidak ada paket yang perlu diperiksa", 'info')
            return {'success': True, 'checked': 0, 'total': 0}
        
        # Execute status check with progress tracking
        start_time = time.time()
        results = self._check_packages_status(packages)
        duration = time.time() - start_time
        
        # Update UI with results
        self._update_package_statuses(results)
        
        # Generate summary
        installed = sum(1 for r in results if r.get('status') == 'installed')
        outdated = sum(1 for r in results if r.get('status') == 'outdated')
        missing = sum(1 for r in results if r.get('status') == 'not_installed')
        
        status_msg = (
            f"📊 Status: {installed} terpasang"
            f"{', ' + str(outdated) + ' perlu diperbarui' if outdated else ''}"
            f"{', ' + str(missing) + ' belum terpasang' if missing else ''}"
            f" dalam {duration:.1f} detik"
        )
        
        self.log(status_msg, 'success')
        
        return {
            'success': True,
            'checked': len(results),
            'installed': installed,
            'outdated': outdated,
            'missing': missing,
            'duration': duration,
            'results': results
        }
    
    def _get_packages_to_check(self) -> List[str]:
        """Get list of packages to check status for.
        
        Returns:
            List of package names to check
        """
        try:
            # Use parent's method to get packages from UI components
            return self._get_packages_to_process()
        except Exception as e:
            self.log(f"Gagal mendapatkan daftar paket: {str(e)}", 'error')
            return []
    
    def _check_packages_status(self, packages: List[str]) -> List[Dict[str, Any]]:
        """Check status of multiple packages in parallel with progress tracking.
        
        Args:
            packages: List of package names to check
            
        Returns:
            List of dictionaries with package status information
        """
        if not packages:
            return []
            
        total = len(packages)
        checked = 0
        results = []
        
        # Process packages in parallel with progress tracking
        processed_results = self._process_packages(
            packages,
            self._check_package_status,
            progress_message="Memeriksa status paket"
        )
        
        # Extract and return the results
        return [r for r in processed_results['details'] if r.get('status') != 'error']
    
    def _check_package_status(self, package: str) -> Dict[str, Any]:
        """Check status of a single package.
        
        Args:
            package: Package name to check
            
        Returns:
            Dictionary with package status information
        """
        try:
            # Check if package is installed
            result = self._execute_command(
                ['pip', 'show', package],
                progress_callback=lambda p, msg: self._update_progress(
                    message=f"Memeriksa {package}...",
                    current=p,
                    level_name='secondary'
                )
            )
            
            if not result['success'] or 'not found' in result['stdout'].lower():
                return {
                    'package': package,
                    'status': 'not_installed',
                    'message': 'Paket belum terpasang',
                    'success': True
                }
            
            # Parse package info
            info = {}
            for line in result['stdout'].split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    info[key.strip().lower()] = value.strip()
            
            # Get installed version
            installed_version = info.get('version', '')
            
            # Check for updates (non-blocking)
            update_result = self._execute_command(
                ['pip', 'index', 'versions', package],
                progress_callback=lambda p, msg: self._update_progress(
                    message=f"Memeriksa pembaruan untuk {package}...",
                    current=p,
                    level_name='secondary'
                )
            )
            
            latest_version = None
            if update_result['success']:
                # Parse the latest version from the output
                match = re.search(r'LATEST:\s+([\d.]+)', update_result['stdout'])
                if match:
                    latest_version = match.group(1)
            
            status = 'installed'
            message = f"Terpasang: {installed_version}"
            
            if latest_version and latest_version != installed_version:
                status = 'outdated'
                message = f"Pembaruan tersedia: {installed_version} → {latest_version}"
            
            return {
                'success': True,
                'package': package,
                'status': status,
                'version': installed_version,
                'latest_version': latest_version,
                'message': message,
                'location': info.get('location', '')
            }
            
        except Exception as e:
            error_msg = f"Gagal memeriksa status {package}: {str(e)}"
            self.log(error_msg, 'error')
            return {
                'success': False,
                'package': package,
                'status': 'error',
                'message': error_msg
            }
    
    def _update_package_statuses(self, results: List[Dict[str, Any]]) -> None:
        """Update UI with package status information.
        
        Args:
            results: List of package status dictionaries
        """
        try:
            # Update status panel if available
            status_panel = self.ui_components.get('status_panel')
            if status_panel and hasattr(status_panel, 'update_package_statuses'):
                # Call the method directly since it's now synchronous
                status_panel.update_package_statuses(results)
            
            # Also update the package list if available
            package_list = self.ui_components.get('package_list')
            if package_list and hasattr(package_list, 'update_package_statuses'):
                # Call the method directly since it's now synchronous
                package_list.update_package_statuses(results)
                
        except Exception as e:
            self.log(f"Gagal memperbarui UI status paket: {str(e)}", 'error')
    
    def _get_latest_version(self, package: str) -> Optional[str]:
        """Get latest version of a package from PyPI.
        
        Args:
            package: Package name
            
        Returns:
            Latest version string or None if not found
        """
        try:
            # Use pip index versions to get latest version
            result = self._execute_command(
                ['pip', 'index', 'versions', package],
                capture_output=True,
                text=True
            )
            
            if result['success'] and result.get('stdout'):
                # Extract version from output like: "Package 'package' 1.2.3 available"
                match = re.search(r"'(?:[^']+)'\s+([\d.]+)", result['stdout'])
                if match:
                    return match.group(1)
            
            return None
            
        except Exception:
            return None
    
    def _check_single_package_status(self, package: str) -> Dict[str, Any]:
        """Check status of a single package (alias for _check_package_status for compatibility).
        
        Args:
            package: Package name to check
            
        Returns:
            Dictionary with package status information
        """
        return self._check_package_status(package)
    
    def get_operations(self) -> Dict[str, Callable]:
        """Get available operations for this handler.
        
        Returns:
            Dictionary of operation name to callable mapping
        """
        return {
            'execute': self.execute_operation
        }
