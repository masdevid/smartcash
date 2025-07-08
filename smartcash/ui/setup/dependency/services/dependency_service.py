"""
File: smartcash/ui/setup/dependency/services/dependency_service.py
Deskripsi: Consolidated dependency service with package management functionality
"""

from typing import Dict, Any, Optional, List
import logging
from dataclasses import dataclass

@dataclass
class PackageStatus:
    """Package status information."""
    name: str
    installed: bool
    version: Optional[str] = None
    available_version: Optional[str] = None
    needs_update: bool = False

class DependencyService:
    """Consolidated service for dependency module with package management functionality."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize dependency service.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self._package_status_cache: Dict[str, PackageStatus] = {}
    
    def get_package_status(self, package_name: str) -> PackageStatus:
        """Get status of a specific package.
        
        Args:
            package_name: Name of the package to check
            
        Returns:
            PackageStatus object with package information
        """
        try:
            # Check cache first
            if package_name in self._package_status_cache:
                return self._package_status_cache[package_name]
            
            # Import and use the existing tracker
            from .package_status_tracker import PackageStatusTracker
            tracker = PackageStatusTracker({}, self.logger)
            
            # Get package info
            is_installed = tracker.is_package_installed(package_name)
            version = tracker.get_package_version(package_name) if is_installed else None
            
            # Create status object
            status = PackageStatus(
                name=package_name,
                installed=is_installed,
                version=version,
                needs_update=False  # TODO: Implement update checking
            )
            
            # Cache the result
            self._package_status_cache[package_name] = status
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting package status for {package_name}: {e}")
            return PackageStatus(
                name=package_name,
                installed=False,
                version=None
            )
    
    def get_multiple_package_status(self, package_names: List[str]) -> Dict[str, PackageStatus]:
        """Get status of multiple packages.
        
        Args:
            package_names: List of package names to check
            
        Returns:
            Dictionary mapping package names to their status
        """
        results = {}
        for package_name in package_names:
            results[package_name] = self.get_package_status(package_name)
        return results
    
    def clear_cache(self) -> None:
        """Clear the package status cache."""
        self._package_status_cache.clear()
        self.logger.info("Package status cache cleared")
    
    def refresh_package_status(self, package_name: str) -> PackageStatus:
        """Refresh status of a specific package (bypasses cache).
        
        Args:
            package_name: Name of the package to refresh
            
        Returns:
            Updated PackageStatus object
        """
        # Remove from cache if exists
        if package_name in self._package_status_cache:
            del self._package_status_cache[package_name]
        
        # Get fresh status
        return self.get_package_status(package_name)
    
    def get_installed_packages(self) -> List[str]:
        """Get list of all installed packages.
        
        Returns:
            List of installed package names
        """
        try:
            from .package_status_tracker import PackageStatusTracker
            tracker = PackageStatusTracker({}, self.logger)
            return tracker.get_installed_packages()
        except Exception as e:
            self.logger.error(f"Error getting installed packages: {e}")
            return []
    
    def validate_package_name(self, package_name: str) -> Dict[str, Any]:
        """Validate package name format.
        
        Args:
            package_name: Package name to validate
            
        Returns:
            Dictionary with validation results
        """
        try:
            # Basic validation
            if not package_name or not isinstance(package_name, str):
                return {
                    'valid': False,
                    'error': 'Package name must be a non-empty string'
                }
            
            # Check for invalid characters
            invalid_chars = [' ', '\t', '\n', '\r']
            if any(char in package_name for char in invalid_chars):
                return {
                    'valid': False,
                    'error': 'Package name contains invalid characters'
                }
            
            return {
                'valid': True,
                'error': None
            }
            
        except Exception as e:
            self.logger.error(f"Error validating package name {package_name}: {e}")
            return {
                'valid': False,
                'error': f'Validation error: {str(e)}'
            }
    
    def get_package_info(self, package_name: str) -> Dict[str, Any]:
        """Get detailed information about a package.
        
        Args:
            package_name: Name of the package
            
        Returns:
            Dictionary with package information
        """
        try:
            status = self.get_package_status(package_name)
            return {
                'name': status.name,
                'installed': status.installed,
                'version': status.version,
                'available_version': status.available_version,
                'needs_update': status.needs_update
            }
        except Exception as e:
            self.logger.error(f"Error getting package info for {package_name}: {e}")
            return {
                'name': package_name,
                'installed': False,
                'version': None,
                'available_version': None,
                'needs_update': False,
                'error': str(e)
            }