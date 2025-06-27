
# =============================================================================
# File: smartcash/ui/setup/dependency/utils/package_manager.py
# Deskripsi: Package management utilities
# =============================================================================

import subprocess
import sys
from typing import List, Dict, Any, Optional

class PackageManager:
    """Utility untuk package management operations"""
    
    def __init__(self):
        self.python_executable = sys.executable
    
    def install_package(self, pip_name: str, timeout: int = 300) -> bool:
        """Install single package"""
        try:
            cmd = [self.python_executable, '-m', 'pip', 'install', pip_name]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return result.returncode == 0
        except Exception:
            return False
    
    def uninstall_package(self, package_name: str) -> bool:
        """Uninstall single package"""
        try:
            cmd = [self.python_executable, '-m', 'pip', 'uninstall', '-y', package_name]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            return result.returncode == 0
        except Exception:
            return False
    
    def get_installed_packages(self) -> Dict[str, str]:
        """Get list of installed packages dengan versi"""
        try:
            cmd = [self.python_executable, '-m', 'pip', 'list', '--format=json']
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                import json
                packages = json.loads(result.stdout)
                return {pkg['name']: pkg['version'] for pkg in packages}
        except Exception:
            pass
        return {}