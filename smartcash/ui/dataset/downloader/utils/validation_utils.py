"""
File: smartcash/ui/dataset/downloader/utils/validation_utils.py
Deskripsi: Dataset validation utilities dengan backend integration
"""

from typing import Dict, Any

def validate_download_config(ui_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate download configuration dengan comprehensive checks.
    
    Args:
        ui_config: UI configuration
        
    Returns:
        Dictionary dengan validation result
    """
    errors = []
    warnings = []
    
    # Extract roboflow config
    roboflow = ui_config.get('data', {}).get('roboflow', {})
    download = ui_config.get('download', {})
    
    # Required fields validation
    required_fields = {
        'workspace': roboflow.get('workspace', '').strip(),
        'project': roboflow.get('project', '').strip(),
        'version': roboflow.get('version', '').strip(),
        'api_key': roboflow.get('api_key', '').strip()
    }
    
    # Check missing fields
    missing_fields = [field for field, value in required_fields.items() if not value]
    if missing_fields:
        errors.extend([f"Field '{field}' wajib diisi" for field in missing_fields])
    
    # Format validation
    if required_fields['workspace'] and len(required_fields['workspace']) < 3:
        errors.append("Workspace minimal 3 karakter")
    
    if required_fields['project'] and len(required_fields['project']) < 3:
        errors.append("Project minimal 3 karakter")
    
    if required_fields['api_key'] and len(required_fields['api_key']) < 10:
        errors.append("API key terlalu pendek (minimal 10 karakter)")
    
    # Download options validation
    retry_count = download.get('retry_count', 3)
    if not isinstance(retry_count, int) or retry_count < 1 or retry_count > 10:
        warnings.append("Retry count sebaiknya antara 1-10")
    
    timeout = download.get('timeout', 30)
    if not isinstance(timeout, int) or timeout < 10 or timeout > 300:
        warnings.append("Timeout sebaiknya antara 10-300 detik")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'config_summary': {
            'target': f"{required_fields['workspace']}/{required_fields['project']}:v{required_fields['version']}",
            'api_key_masked': f"{required_fields['api_key'][:4]}{'*' * (len(required_fields['api_key']) - 8)}{required_fields['api_key'][-4:]}" if len(required_fields['api_key']) > 8 else '****',
            'uuid_rename': download.get('rename_files', True),
            'validation': download.get('validate_download', True),
            'backup': download.get('backup_existing', False)
        }
    }

def check_cleanup_feasibility(targets_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check apakah cleanup operation feasible.
    
    Args:
        targets_result: Result dari get_cleanup_targets
        
    Returns:
        Dictionary dengan feasibility analysis
    """
    if targets_result.get('status') != 'success':
        return {
            'feasible': False,
            'reason': 'Gagal mendapatkan cleanup targets',
            'recommendation': 'Periksa koneksi dan akses file'
        }
    
    summary = targets_result.get('summary', {})
    targets = targets_result.get('targets', {})
    
    total_files = summary.get('total_files', 0)
    total_size = summary.get('total_size', 0)
    
    if total_files == 0:
        return {
            'feasible': False,
            'reason': 'Tidak ada file untuk dibersihkan',
            'recommendation': 'Dataset sudah bersih atau belum ada data'
        }
    
    # Check jika cleanup terlalu besar (> 10GB warning)
    if total_size > 10 * 1024 * 1024 * 1024:  # 10GB
        return {
            'feasible': True,
            'warning': True,
            'reason': f'Cleanup akan menghapus {summary.get("size_formatted", "data besar")}',
            'recommendation': 'Pertimbangkan backup terlebih dahulu'
        }
    
    # Check target distribution
    high_impact_targets = []
    for target_name, target_info in targets.items():
        if target_info.get('file_count', 0) > 1000:
            high_impact_targets.append(f"{target_name} ({target_info.get('file_count', 0):,} files)")
    
    result = {
        'feasible': True,
        'summary': {
            'total_files': total_files,
            'total_size_formatted': summary.get('size_formatted', '0 B'),
            'target_count': len(targets)
        }
    }
    
    if high_impact_targets:
        result['high_impact'] = high_impact_targets
        result['recommendation'] = 'Targets dengan file count tinggi terdeteksi'
    
    return result

def format_validation_summary(validation: Dict[str, Any]) -> str:
    """
    Format validation summary untuk display.
    
    Args:
        validation: Validation result
        
    Returns:
        Formatted summary string
    """
    if not validation['valid']:
        error_lines = [
            "❌ Konfigurasi tidak valid:",
            ""
        ]
        error_lines.extend([f"  • {error}" for error in validation['errors']])
        return '\n'.join(error_lines)
    
    config_summary = validation.get('config_summary', {})
    summary_lines = [
        "✅ Konfigurasi valid:",
        f"🎯 Target: {config_summary.get('target', 'N/A')}",
        f"🔑 API Key: {config_summary.get('api_key_masked', '****')}",
        f"🔄 UUID Rename: {'✅' if config_summary.get('uuid_rename') else '❌'}",
        f"✅ Validasi: {'✅' if config_summary.get('validation') else '❌'}",
        f"💾 Backup: {'✅' if config_summary.get('backup') else '❌'}"
    ]
    
    if validation.get('warnings'):
        summary_lines.append("")
        summary_lines.append("⚠️ Peringatan:")
        summary_lines.extend([f"  • {warning}" for warning in validation['warnings']])
    
    return '\n'.join(summary_lines)