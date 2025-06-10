"""
File: smartcash/ui/setup/dependency/utils/package_utils.py
Deskripsi: Consolidated utilities untuk package operations dengan DRY pattern
"""

from typing import Dict, Any, List, Tuple, Optional
import subprocess
import sys
import json
import re
import pkg_resources

def get_installed_packages_dict() -> Dict[str, str]:
    """Get normalized dictionary of installed packages dengan versi - one-liner approach"""
    try:
        # Primary method: pip list JSON
        process = subprocess.run([sys.executable, "-m", "pip", "list", "--format=json"], 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        
        if process.returncode == 0:
            packages = json.loads(process.stdout)
            # One-liner normalization dengan multiple variants
            return {variant: pkg['version'] 
                   for pkg in packages 
                   for variant in _get_package_name_variants(pkg['name'].lower())}
    except Exception:
        pass
    
    # Fallback: pkg_resources dengan one-liner normalization
    try:
        return {variant: pkg.version 
               for pkg in pkg_resources.working_set 
               for variant in _get_package_name_variants(pkg.key.lower())}
    except Exception:
        return {}

def _get_package_name_variants(name: str) -> List[str]:
    """Get all variants dari package name untuk comprehensive matching - one-liner"""
    return [name, name.replace('-', '_'), name.replace('_', '-')]

def parse_package_requirement(requirement: str) -> Tuple[str, str]:
    """Parse pip requirement string menjadi (name, version_spec) - one-liner regex"""
    requirement = requirement.strip().split('#')[0].strip()  # Remove comments
    match = re.match(r'^([a-zA-Z0-9_\-\.]+)([<>=!~]+.+)?$', requirement)
    return (match.group(1).lower(), match.group(2) or "") if match else (requirement.lower(), "")

def check_version_compatibility(installed_version: str, version_spec: str) -> bool:
    """Check version compatibility dengan fallback - one-liner try/except"""
    if not version_spec:
        return True
    
    try:
        return pkg_resources.parse_version(installed_version) in pkg_resources.Requirement.parse(f"dummy{version_spec}").specifier
    except Exception:
        # Fallback: simple version comparison
        return _simple_version_check(installed_version, version_spec)

def _simple_version_check(installed: str, spec: str) -> bool:
    """Simple version check fallback - one-liner mapping"""
    comparisons = {
        '>=': lambda i, r: pkg_resources.parse_version(i) >= pkg_resources.parse_version(r),
        '==': lambda i, r: i == r,
        '>': lambda i, r: pkg_resources.parse_version(i) > pkg_resources.parse_version(r),
        '<=': lambda i, r: pkg_resources.parse_version(i) <= pkg_resources.parse_version(r),
        '<': lambda i, r: pkg_resources.parse_version(i) < pkg_resources.parse_version(r)
    }
    
    for op, func in comparisons.items():
        if spec.startswith(op):
            try:
                return func(installed, spec[len(op):])
            except Exception:
                return True
    return True

def check_package_installation_status(package_name: str, version_spec: str = "", 
                                    installed_packages: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Check comprehensive installation status untuk single package"""
    
    if installed_packages is None:
        installed_packages = get_installed_packages_dict()
    
    # Check all variants
    variants = _get_package_name_variants(package_name.lower())
    installed_version = next((installed_packages[variant] for variant in variants if variant in installed_packages), None)
    
    if not installed_version:
        return {'status': 'missing', 'installed': False, 'version': None, 'compatible': False}
    
    # Check compatibility
    compatible = check_version_compatibility(installed_version, version_spec)
    status = 'installed' if compatible else 'upgrade'
    
    return {
        'status': status,
        'installed': True,
        'version': installed_version,
        'compatible': compatible,
        'required_version': version_spec
    }

def filter_uninstalled_packages(selected_packages: List[str], logger_func=None) -> List[str]:
    """Filter packages yang belum terinstall untuk skip installed ones dengan logger function"""
    
    installed_packages = get_installed_packages_dict()
    packages_to_install = []
    
    for package in selected_packages:
        package_name, version_spec = parse_package_requirement(package)
        status_info = check_package_installation_status(package_name, version_spec, installed_packages)
        
        if status_info['installed']:
            version_info = status_info.get('version', 'unknown')
            logger_func and logger_func(f"⏭️ {package_name} sudah terinstall v{version_info}, dilewati")
        else:
            packages_to_install.append(package)
            logger_func and logger_func(f"📦 {package_name} akan diinstall")
    
    return packages_to_install

def install_single_package(package: str, timeout: int = 300) -> Tuple[bool, str]:
    """Install single package dan return (success, message) - one-liner result processing"""
    
    try:
        # Pastikan package adalah string
        if not isinstance(package, str):
            return False, f"Invalid package format: {package}. Expected string."
            
        # Bersihkan input package
        package = package.strip()
        if not package:
            return False, "Empty package name provided"
            
        # Log command yang akan dijalankan
        cmd = [sys.executable, "-m", "pip", "install", package, "--no-cache-dir"]
        
        try:
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True, 
                check=False, 
                timeout=timeout
            )
            
            if process.returncode == 0:
                return True, f"Successfully installed {package}"
            else:
                error_msg = process.stderr.strip() or "Unknown error"
                return False, f"Failed to install {package}: {error_msg}"
                
        except subprocess.TimeoutExpired:
            return False, f"Installation timeout ({timeout}s) for {package}"
            
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return False, f"Error installing {package}: {str(e)}\n{error_trace}"

def get_package_detailed_info(package_name: str) -> Dict[str, Any]:
    """Get detailed info untuk package menggunakan pip show - one-liner parsing"""
    
    try:
        process = subprocess.run([sys.executable, "-m", "pip", "show", package_name],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        
        if process.returncode == 0:
            # One-liner parsing pip show output
            details = {line.split(': ', 1)[0].lower(): line.split(': ', 1)[1] 
                      for line in process.stdout.strip().split('\n') 
                      if ': ' in line}
            
            # Special handling untuk requires field
            if 'requires' in details:
                details['requires'] = [pkg.strip() for pkg in details['requires'].split(',')] if details['requires'] else []
            
            return details
    except Exception:
        pass
    
    return {}

def batch_check_packages_status(package_list: List[str]) -> Dict[str, Dict[str, Any]]:
    """Batch check status untuk multiple packages - one-liner mapping"""
    
    installed_packages = get_installed_packages_dict()
    
    return {
        package: check_package_installation_status(
            *parse_package_requirement(package), 
            installed_packages
        )
        for package in package_list
    }

def normalize_package_name_for_matching(name: str) -> str:
    """Normalize package name untuk consistent matching - one-liner"""
    return name.lower().replace('-', '_').replace(' ', '_')

def extract_package_name_from_requirement(requirement: str) -> str:
    """Extract clean package name dari pip requirement - one-liner"""
    return parse_package_requirement(requirement)[0]

def is_package_installed(package_name: str, installed_packages: Optional[Dict[str, str]] = None) -> bool:
    """Simple check apakah package terinstall - one-liner"""
    if installed_packages is None:
        installed_packages = get_installed_packages_dict()
    
    return any(variant in installed_packages for variant in _get_package_name_variants(package_name.lower()))

def get_package_version(package_name: str, installed_packages: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Get installed version untuk package - one-liner dengan None fallback"""
    if installed_packages is None:
        installed_packages = get_installed_packages_dict()
    
    return next((installed_packages[variant] for variant in _get_package_name_variants(package_name.lower()) 
                if variant in installed_packages), None)