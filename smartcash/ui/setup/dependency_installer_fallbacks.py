"""
File: smartcash/ui/setup/dependency_installer_fallbacks.py
Deskripsi: Fallback mechanisms untuk instalasi dependencies
"""

import sys
import subprocess
from typing import List, Optional, Dict, Any
from pathlib import Path

class DependencyInstallerError(Exception):
    """Custom exception untuk dependency installer."""
    pass

def manual_pip_install(
    packages: List[str], 
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Manual pip install dengan konfigurasi lanjutan.
    
    Args:
        packages: Daftar package untuk diinstall
        options: Konfigurasi tambahan
    
    Returns:
        Dictionary berisi status instalasi
    """
    default_options = {
        'upgrade': False,
        'user': False,
        'verbose': False,
        'force': False
    }
    
    # Merge options
    options = {**default_options, **(options or {})}
    
    try:
        cmd = [sys.executable, '-m', 'pip', 'install']
        
        if options['upgrade']:
            cmd.append('--upgrade')
        if options['user']:
            cmd.append('--user')
        if options['verbose']:
            cmd.append('-v')
        if options['force']:
            cmd.append('--force-reinstall')
        
        cmd.extend(packages)
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def generate_requirements_file(
    packages: List[str], 
    output_path: Optional[str] = None
) -> Path:
    """
    Generate file requirements dengan fallback.
    
    Args:
        packages: Daftar package
        output_path: Path file output
    
    Returns:
        Path file requirements
    """
    # Default path jika tidak ditentukan
    if not output_path:
        output_path = Path.cwd() / 'requirements.txt'
    
    try:
        with open(output_path, 'w') as f:
            for pkg in packages:
                f.write(f"{pkg}\n")
        
        return Path(output_path)
    except IOError as e:
        raise DependencyInstallerError(f"Gagal membuat requirements file: {str(e)}")

def check_package_compatibility(
    package_name: str, 
    min_version: Optional[str] = None
) -> Dict[str, Any]:
    """
    Periksa kompatibilitas package secara mendalam.
    
    Args:
        package_name: Nama package
        min_version: Versi minimum yang diperlukan
    
    Returns:
        Dictionary status kompatibilitas
    """
    try:
        module = __import__(package_name)
        version = getattr(module, '__version__', 'Unknown')
        
        result = {
            'installed': True,
            'name': package_name,
            'version': version,
            'compatible': True
        }
        
        if min_version:
            from packaging import version as version_parser
            result['compatible'] = version_parser.parse(version) >= version_parser.parse(min_version)
        
        return result
    
    except ImportError:
        return {
            'installed': False,
            'name': package_name,
            'version': None,
            'compatible': False
        }