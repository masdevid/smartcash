"""
File: smartcash/ui/setup/dependency_installer_fallbacks.py
Deskripsi: Fallback mechanisms untuk instalasi dependencies
"""

import sys
import subprocess
from typing import List, Optional

class DependencyInstallerError(Exception):
    """Custom exception untuk dependency installer."""
    pass

def manual_pip_install(
    packages: List[str], 
    upgrade: bool = False, 
    user: bool = False, 
    verbose: bool = False
) -> Optional[str]:
    """
    Manual pip install dengan fallback mekanisme.
    
    Args:
        packages: Daftar package untuk diinstall
        upgrade: Flag untuk upgrade package
        user: Install untuk user saat ini
        verbose: Mode verbose untuk debugging
    
    Returns:
        Output instalasi atau None jika berhasil
    """
    try:
        cmd = [sys.executable, '-m', 'pip', 'install']
        
        if upgrade:
            cmd.append('--upgrade')
        if user:
            cmd.append('--user')
        if verbose:
            cmd.append('-v')
        
        cmd.extend(packages)
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise DependencyInstallerError(f"Instalasi gagal: {result.stderr}")
        
        return result.stdout if verbose else None
    
    except Exception as e:
        raise DependencyInstallerError(f"Error saat instalasi: {str(e)}")

def check_package_compatibility(
    package_name: str, 
    min_version: Optional[str] = None
) -> bool:
    """
    Periksa kompatibilitas package.
    
    Args:
        package_name: Nama package
        min_version: Versi minimum yang diperlukan
    
    Returns:
        Boolean menunjukkan kompatibilitas
    """
    try:
        module = __import__(package_name)
        version = getattr(module, '__version__', '0.0.0')
        
        if min_version:
            from packaging import version as version_parser
            return version_parser.parse(version) >= version_parser.parse(min_version)
        
        return True
    except ImportError:
        return False

def generate_requirements_file(
    packages: List[str], 
    output_path: str = 'requirements.txt'
):
    """
    Generate file requirements.
    
    Args:
        packages: Daftar package
        output_path: Path file output
    """
    try:
        with open(output_path, 'w') as f:
            for pkg in packages:
                f.write(f"{pkg}\n")
    except IOError as e:
        raise DependencyInstallerError(f"Gagal membuat requirements file: {str(e)}")