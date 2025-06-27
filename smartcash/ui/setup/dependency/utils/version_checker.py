
# =============================================================================
# File: smartcash/ui/setup/dependency/utils/version_checker.py  
# Deskripsi: Version comparison utilities
# =============================================================================

import subprocess
import sys
from typing import Optional, Tuple

class VersionChecker:
    """Utility untuk version checking dan comparison"""
    
    def __init__(self):
        self.python_executable = sys.executable
    
    def get_installed_version(self, package_name: str) -> Optional[str]:
        """Get installed version of package"""
        try:
            import importlib.util
            spec = importlib.util.find_spec(package_name)
            if spec is None:
                return None
            
            module = importlib.import_module(package_name)
            return getattr(module, '__version__', 'unknown')
        except Exception:
            return None
    
    def get_latest_version(self, pip_name: str) -> Optional[str]:
        """Get latest version dari PyPI"""
        try:
            package_name = pip_name.split('>=')[0].split('==')[0].strip()
            cmd = [self.python_executable, '-m', 'pip', 'index', 'versions', package_name]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'Available versions:' in line:
                        versions = line.split(':', 1)[1].strip()
                        if versions:
                            return versions.split(',')[0].strip()
        except Exception:
            pass
        return None
    
    def compare_versions(self, installed: str, latest: str) -> bool:
        """Compare versions - return True jika update available"""
        if not installed or not latest or installed == "unknown" or latest == "unknown":
            return False
        
        try:
            installed_parts = [int(x) for x in installed.split('.')]
            latest_parts = [int(x) for x in latest.split('.')]
            
            max_len = max(len(installed_parts), len(latest_parts))
            installed_parts.extend([0] * (max_len - len(installed_parts)))
            latest_parts.extend([0] * (max_len - len(latest_parts)))
            
            return latest_parts > installed_parts
        except Exception:
            return installed != latest