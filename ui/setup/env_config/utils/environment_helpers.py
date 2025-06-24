"""
File: smartcash/ui/setup/env_config/utils/environment_helpers.py
Deskripsi: Helper functions untuk environment dan auto-detection config templates
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from smartcash.ui.setup.env_config.constants import (
    CONFIG_SOURCE_PATH, CONFIG_EXTENSIONS, ESSENTIAL_CONFIG_PATTERNS
)

def discover_config_templates() -> List[str]:
    """🔍 Auto-detect semua config templates dari smartcash/configs"""
    config_path = Path(CONFIG_SOURCE_PATH)
    if not config_path.exists():
        return []
    
    discovered_configs = []
    for ext in CONFIG_EXTENSIONS:
        discovered_configs.extend([
            f.name for f in config_path.glob(f'*{ext}')
            if f.is_file() and not f.name.startswith('.')
        ])
    
    return sorted(discovered_configs)

def get_essential_configs() -> List[str]:
    """📋 Get essential configs berdasarkan pattern detection"""
    all_configs = discover_config_templates()
    essential_configs = []
    
    for pattern in ESSENTIAL_CONFIG_PATTERNS:
        matching_configs = [
            config for config in all_configs 
            if pattern in config.lower()
        ]
        if matching_configs:
            essential_configs.extend(matching_configs)
    
    return list(set(essential_configs))  # Remove duplicates

def validate_config_completeness() -> Dict[str, Any]:
    """✅ Validasi kelengkapan config templates"""
    discovered = discover_config_templates()
    essential = get_essential_configs()
    
    missing_essential = [
        pattern for pattern in ESSENTIAL_CONFIG_PATTERNS
        if not any(pattern in config.lower() for config in discovered)
    ]
    
    return {
        'total_configs': len(discovered),
        'discovered_configs': discovered,
        'essential_configs': essential,
        'missing_essential': missing_essential,
        'is_complete': len(missing_essential) == 0
    }

def get_config_summary() -> str:
    """📊 Generate config discovery summary"""
    validation = validate_config_completeness()
    
    summary_lines = [
        f"📋 Config Templates Ditemukan: {validation['total_configs']}",
        f"✅ Essential Configs: {len(validation['essential_configs'])}",
    ]
    
    if validation['missing_essential']:
        summary_lines.append(f"⚠️ Missing Essential: {', '.join(validation['missing_essential'])}")
    
    return "\n".join(summary_lines)

def create_directories_if_missing(directories: List[str]) -> Dict[str, bool]:
    """📁 Create directories jika belum ada"""
    results = {}
    
    for directory in directories:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            results[directory] = True
        except Exception:
            results[directory] = False
    
    return results

def check_symlink_validity(symlink_path: str) -> bool:
    """🔗 Check apakah symlink masih valid"""
    path = Path(symlink_path)
    return path.is_symlink() and path.exists()

def get_environment_info() -> Dict[str, Any]:
    """🔍 Get environment information summary"""
    try:
        import google.colab
        is_colab = True
        drive_mounted = Path('/content/drive').exists()
    except ImportError:
        is_colab = False
        drive_mounted = False
    
    config_validation = validate_config_completeness()
    
    return {
        'is_colab': is_colab,
        'drive_mounted': drive_mounted,
        'config_source_exists': Path(CONFIG_SOURCE_PATH).exists(),
        'total_configs_found': config_validation['total_configs'],
        'essential_configs_complete': config_validation['is_complete'],
        'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
        'working_directory': str(Path.cwd())
    }