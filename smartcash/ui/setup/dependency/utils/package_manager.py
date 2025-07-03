"""Package management utilities with real-time progress tracking.

This module provides a PackageManager class for handling Python package operations
with real-time progress tracking and callbacks for UI integration.
"""

import subprocess
import sys
import time
import json
import re
from typing import List, Dict, Any, Optional, Callable, Tuple

class PackageManager:
    """Package management operations with real-time progress tracking.
    
    This class provides methods for installing, uninstalling, and checking packages
    with real-time progress tracking and callbacks for UI integration.
    """
    
    def __init__(self):
        """Initialize the PackageManager with the current Python executable."""
        self.python_executable = sys.executable
    
    def install_package(self, 
                       pip_name: str, 
                       progress_callback: Optional[Callable[[int, int, str], None]] = None,
                       timeout: int = 300) -> Tuple[bool, str]:
        """Install a package with real-time progress tracking.
        
        Args:
            pip_name: Package name and version specifier (e.g., 'numpy>=1.20.0')
            progress_callback: Function to call with progress updates
            timeout: Maximum time to wait for installation in seconds
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Start with 0% progress
            if progress_callback:
                progress_callback(0, 100, f"Starting installation of {pip_name}...")
            
            # Run pip in a way that we can capture real-time output
            cmd = [self.python_executable, '-m', 'pip', 'install', pip_name, '--verbose']
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Track progress through output parsing
            output_lines = []
            progress = 0
            start_time = time.time()
            
            while True:
                # Check if process has finished
                if process.poll() is not None:
                    break
                    
                # Check for timeout
                if time.time() - start_time > timeout:
                    process.kill()
                    if progress_callback:
                        progress_callback(100, 100, f"Installation timed out after {timeout} seconds")
                    return False, f"Installation of {pip_name} timed out"
                
                # Read output line
                line = process.stdout.readline()
                if not line:
                    time.sleep(0.1)
                    continue
                    
                output_lines.append(line.strip())
                
                # Update progress based on output parsing
                if 'Collecting' in line:
                    progress = 10
                    if progress_callback:
                        progress_callback(progress, 100, f"Collecting {pip_name}...")
                elif 'Downloading' in line:
                    progress = 30
                    if progress_callback:
                        progress_callback(progress, 100, f"Downloading {pip_name}...")
                elif 'Installing' in line:
                    progress = 70
                    if progress_callback:
                        progress_callback(progress, 100, f"Installing {pip_name}...")
                elif 'Successfully installed' in line:
                    progress = 100
                    if progress_callback:
                        progress_callback(progress, 100, f"Successfully installed {pip_name}")
            
            # Process complete, check return code
            success = process.returncode == 0
            message = "\n".join(output_lines[-5:])  # Last 5 lines of output
            
            # Final progress update
            if progress_callback:
                status = "Successfully installed" if success else "Failed to install"
                progress_callback(100, 100, f"{status} {pip_name}")
                
            return success, message
            
        except Exception as e:
            if progress_callback:
                progress_callback(100, 100, f"Error installing {pip_name}: {str(e)}")
            return False, str(e)
    
    def uninstall_package(self, 
                         package_name: str,
                         progress_callback: Optional[Callable[[int, int, str], None]] = None) -> Tuple[bool, str]:
        """Uninstall a package with real-time progress tracking.
        
        Args:
            package_name: Name of the package to uninstall
            progress_callback: Function to call with progress updates
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Start with 0% progress
            if progress_callback:
                progress_callback(0, 100, f"Starting uninstallation of {package_name}...")
            
            # Run pip in a way that we can capture real-time output
            cmd = [self.python_executable, '-m', 'pip', 'uninstall', '-y', package_name, '--verbose']
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Track progress through output parsing
            output_lines = []
            progress = 0
            
            while True:
                # Check if process has finished
                if process.poll() is not None:
                    break
                    
                # Read output line
                line = process.stdout.readline()
                if not line:
                    time.sleep(0.1)
                    continue
                    
                output_lines.append(line.strip())
                
                # Update progress based on output parsing
                if 'Found existing' in line:
                    progress = 20
                    if progress_callback:
                        progress_callback(progress, 100, f"Found {package_name}...")
                elif 'Uninstalling' in line:
                    progress = 50
                    if progress_callback:
                        progress_callback(progress, 100, f"Uninstalling {package_name}...")
                elif 'Successfully uninstalled' in line:
                    progress = 100
                    if progress_callback:
                        progress_callback(progress, 100, f"Successfully uninstalled {package_name}")
            
            # Process complete, check return code
            success = process.returncode == 0
            message = "\n".join(output_lines[-5:])  # Last 5 lines of output
            
            # Final progress update
            if progress_callback:
                status = "Successfully uninstalled" if success else "Failed to uninstall"
                progress_callback(100, 100, f"{status} {package_name}")
                
            return success, message
            
        except Exception as e:
            if progress_callback:
                progress_callback(100, 100, f"Error uninstalling {package_name}: {str(e)}")
            return False, str(e)
    
    def get_installed_packages(self) -> Dict[str, str]:
        """Get list of installed packages with versions.
        
        Returns:
            Dictionary mapping package names to versions
        """
        try:
            cmd = [self.python_executable, '-m', 'pip', 'list', '--format=json']
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                packages = json.loads(result.stdout)
                return {pkg['name']: pkg['version'] for pkg in packages}
        except Exception:
            pass
        return {}
        
    def check_for_updates(self, 
                         packages: List[str],
                         progress_callback: Optional[Callable[[int, int, str], None]] = None) -> Dict[str, str]:
        """Check for available updates for the given packages.
        
        Args:
            packages: List of package names to check
            progress_callback: Function to call with progress updates
            
        Returns:
            Dictionary mapping package names to available versions for packages with updates
        """
        updates_available = {}
        total_packages = len(packages)
        
        for i, package in enumerate(packages):
            try:
                if progress_callback:
                    progress_callback(i, total_packages, f"Checking {package} for updates...")
                
                # Get installed version
                installed_version = self.get_installed_packages().get(package)
                if not installed_version:
                    continue
                
                # Check for latest version
                cmd = [self.python_executable, '-m', 'pip', 'index', 'versions', package]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    # Parse output to find latest version
                    latest_version = None
                    for line in result.stdout.split('\n'):
                        if 'Available versions:' in line:
                            versions = line.split(':', 1)[1].strip()
                            if versions:
                                latest_version = versions.split(',')[0].strip()
                                break
                    
                    # Compare versions
                    if latest_version and self._is_newer_version(latest_version, installed_version):
                        updates_available[package] = latest_version
                        if progress_callback:
                            progress_callback(i+1, total_packages, 
                                             f"Update available for {package}: {installed_version} → {latest_version}")
                    else:
                        if progress_callback:
                            progress_callback(i+1, total_packages, f"{package} is up to date ({installed_version})")
                            
            except Exception as e:
                if progress_callback:
                    progress_callback(i+1, total_packages, f"Error checking {package}: {str(e)}")
        
        # Final progress update
        if progress_callback:
            progress_callback(total_packages, total_packages, 
                             f"Found {len(updates_available)} package(s) with updates available")
        
        return updates_available
    
    def _is_newer_version(self, version1: str, version2: str) -> bool:
        """Compare two version strings and return True if version1 is newer than version2."""
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