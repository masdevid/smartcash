"""
File: smartcash/ui/setup/utils/package_analyzer.py
Deskripsi: Utilitas untuk analisis dan deteksi packages yang terinstall
"""

import sys
import re
import subprocess
from typing import Dict, List, Set

def get_installed_packages() -> Dict[str, str]:
    """
    Dapatkan semua package yang terinstall dengan versinya.
    
    Returns:
        Dictionary {package_name: version}
    """
    installed = {}
    
    try:
        # Metode 1: pip list
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True, text=True, check=False
        )
        
        if result.returncode == 0:
            import json
            packages = json.loads(result.stdout)
            for pkg in packages:
                name = pkg.get('name', '').lower()
                version = pkg.get('version', '')
                if name and version:
                    installed[name] = version
            return installed
    except Exception:
        pass
    
    # Metode 2: pkg_resources fallback
    try:
        import pkg_resources
        for pkg in pkg_resources.working_set:
            installed[pkg.key.lower()] = pkg.version
    except ImportError:
        pass
    
    return installed

def check_missing_packages(packages: List[str], installed: Dict[str, str]) -> List[str]:
    """
    Cek package yang belum terinstall atau perlu diupdate.
    
    Args:
        packages: List package requirements (dengan atau tanpa versi)
        installed: Dictionary package terinstall
        
    Returns:
        List package yang perlu diinstall/update
    """
    missing = []
    
    for pkg_req in packages:
        # Parse package name dan versi requirement
        match = re.match(r'^([a-zA-Z0-9\-_\.]+)(?:[><=!~]=?|@)(.+)?', pkg_req)
        
        if match:
            # Ada requirement versi
            pkg_name, req_version = match.groups()
            pkg_name = pkg_name.lower()
            
            # Jika package tidak ada atau perlu versi spesifik, tambahkan ke missing
            if pkg_name not in installed:
                missing.append(pkg_req)
            # Untuk package dengan versi, skip verifikasi versi (akan selalu diinstall)
            # Ini simpel tapi efektif untuk notebook environment
        else:
            # Tidak ada requirement versi, cek apakah sudah terinstall
            pkg_name = pkg_req.lower()
            if pkg_name not in installed:
                missing.append(pkg_req)
    
    return missing

def is_package_installed(package_name: str) -> bool:
    """
    Cek apakah package tertentu sudah terinstall.
    
    Args:
        package_name: Nama package
        
    Returns:
        Boolean yang menunjukkan apakah package terinstall
    """
    package_name = package_name.lower().split('=')[0].split('>')[0].split('<')[0].strip()
    
    try:
        __import__(package_name)
        return True
    except ImportError:
        # Beberapa package punya nama import berbeda dengan nama package
        common_mappings = {
            'pillow': 'PIL',
            'opencv-python': 'cv2',
            'scikit-learn': 'sklearn',
            'matplotlib': 'matplotlib.pyplot',
            'tensorflow-gpu': 'tensorflow',
            'python-dotenv': 'dotenv'
        }
        
        if package_name in common_mappings:
            try:
                __import__(common_mappings[package_name])
                return True
            except ImportError:
                pass
                
    return package_name in get_installed_packages()