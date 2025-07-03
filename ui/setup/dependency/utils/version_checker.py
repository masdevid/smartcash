"""Version checking utilities with real-time progress tracking.

This module provides a VersionChecker class for checking package versions
and determining if updates are available.
"""

import subprocess
import sys
import re
import importlib.util
import importlib
from typing import Optional, Dict, List, Callable, Any

class VersionChecker:
    """Enhanced version checking with real-time progress tracking.
    
    This class provides methods for checking installed and available package versions
    with real-time progress tracking and callbacks for UI integration.
    """
    
    def __init__(self):
        """Initialize the VersionChecker with the current Python executable."""
        self.python_executable = sys.executable
    
    def get_installed_version(self, 
                             package_name: str, 
                             use_pip: bool = True) -> Optional[str]:
        """Get installed version of a package.
        
        Args:
            package_name: Name of the package to check
            use_pip: Whether to use pip to get version (more reliable) or importlib
            
        Returns:
            Version string or None if not installed
        """
        if use_pip:
            try:
                cmd = [self.python_executable, '-m', 'pip', 'show', package_name]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if line.startswith('Version:'):
                            return line.split(':', 1)[1].strip()
            except Exception:
                pass
        
        # Fall back to importlib if pip fails or use_pip is False
        try:
            spec = importlib.util.find_spec(package_name)
            if spec is None:
                return None
            
            module = importlib.import_module(package_name)
            return getattr(module, '__version__', None)
        except Exception:
            return None
            
    def check_package_status(self, 
                            package_name: str, 
                            progress_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """Check the installation status and version information for a package.
        
        Args:
            package_name: Name of the package to check
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Dictionary with package status information
        """
        result = {
            'name': package_name,
            'installed': False,
            'installed_version': None,
            'latest_version': None,
            'update_available': False
        }
        
        try:
            # Check if installed and get version
            if progress_callback:
                progress_callback(f"Checking if {package_name} is installed...")
                
            installed_version = self.get_installed_version(package_name)
            result['installed'] = installed_version is not None
            result['installed_version'] = installed_version
            
            if not result['installed']:
                if progress_callback:
                    progress_callback(f"{package_name} is not installed")
                return result
                
            # Get latest version
            if progress_callback:
                progress_callback(f"Checking for updates to {package_name}...")
                
            cmd = [self.python_executable, '-m', 'pip', 'index', 'versions', package_name]
            pip_result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if pip_result.returncode == 0:
                for line in pip_result.stdout.split('\n'):
                    if 'Available versions:' in line:
                        versions = line.split(':', 1)[1].strip()
                        if versions:
                            latest_version = versions.split(',')[0].strip()
                            result['latest_version'] = latest_version
                            
                            # Check if update is available
                            if self._is_newer_version(latest_version, installed_version):
                                result['update_available'] = True
                                if progress_callback:
                                    progress_callback(f"Update available for {package_name}: {installed_version} → {latest_version}")
                            else:
                                if progress_callback:
                                    progress_callback(f"{package_name} is up to date ({installed_version})")
                            break
        except Exception as e:
            if progress_callback:
                progress_callback(f"Error checking {package_name}: {str(e)}")
        
        return result
    
    def _is_newer_version(self, version1: str, version2: str) -> bool:
        """Compare two version strings and return True if version1 is newer than version2."""
        if not version1 or not version2 or version1 == "unknown" or version2 == "unknown":
            return False
            
        try:
            # Convert version strings to lists of integers
            v1_parts = [int(x) for x in re.findall(r'\d+', version1)]
            v2_parts = [int(x) for x in re.findall(r'\d+', version2)]
            
            # Pad with zeros to make equal length
            max_len = max(len(v1_parts), len(v2_parts))
            v1_parts.extend([0] * (max_len - len(v1_parts)))
            v2_parts.extend([0] * (max_len - len(v2_parts)))
            
            # Compare version parts
            return v1_parts > v2_parts
        except Exception:
            # Fall back to string comparison if parsing fails
            return version1 != version2