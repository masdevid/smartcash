"""
File: smartcash/ui/dataset/downloader/utils/validation_utils.py
Deskripsi: Centralized validation logic untuk downloader handlers dan backend integration
"""

from typing import Dict, Any, List, Tuple, Optional

def validate_roboflow_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validasi konfigurasi Roboflow dengan pemeriksaan komprehensif.
    
    Args:
        config: Dictionary konfigurasi yang akan divalidasi
        
    Returns:
        Dictionary berisi hasil validasi dengan key 'status' dan 'errors'
    """
    errors = []
    warnings = []
    
    # Extract roboflow config dengan safe access
    roboflow = config.get('data', {}).get('roboflow', {})
    
    # Required fields validation dengan strip untuk menghindari whitespace
    required_fields = {
        'workspace': roboflow.get('workspace', '').strip(),
        'project': roboflow.get('project', '').strip(),
        'version': roboflow.get('version', '').strip(),
        'api_key': roboflow.get('api_key', '').strip()
    }
    
    # Check missing required fields dengan list comprehension
    errors.extend([f"Field '{field}' wajib diisi" for field, value in required_fields.items() if not value])
    
    # Format validation dengan conditional checks
    if required_fields['workspace'] and len(required_fields['workspace']) < 3:
        errors.append("Workspace minimal 3 karakter")
    
    if required_fields['project'] and len(required_fields['project']) < 3:
        errors.append("Project minimal 3 karakter")
    
    if required_fields['api_key'] and len(required_fields['api_key']) < 10:
        errors.append("API key terlalu pendek (minimal 10 karakter)")
    
    # Format validation untuk version
    if required_fields['version'] and not required_fields['version'].isdigit():
        warnings.append("Version biasanya berupa angka")
    
    # Mask API key untuk keamanan
    api_key = required_fields['api_key']
    api_key_masked = f"{api_key[:4]}{'*' * (len(api_key) - 8)}{api_key[-4:]}" if len(api_key) > 8 else '****'
    
    # Return hasil validasi dengan format yang konsisten
    is_valid = len(errors) == 0
    return {
        'status': is_valid,  # Key 'status' untuk API consistency
        'valid': is_valid,   # Key 'valid' untuk backward compatibility
        'errors': errors,
        'warnings': warnings,
        'values': {
            'workspace': required_fields['workspace'],
            'project': required_fields['project'],
            'version': required_fields['version'],
            'api_key_masked': api_key_masked
        }
    }

def validate_download_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validasi konfigurasi download dengan pemeriksaan komprehensif.
    
    Args:
        config: Dictionary konfigurasi yang akan divalidasi
        
    Returns:
        Dictionary berisi hasil validasi dengan key 'status' dan 'errors'
    """
    errors = []
    warnings = []
    
    # Extract download config
    download = config.get('download', {})
    
    # Download config validation
    retry_count = download.get('retry_count', 3)
    if not isinstance(retry_count, int) or retry_count < 1 or retry_count > 10:
        warnings.append("Retry count sebaiknya antara 1-10")
    
    timeout = download.get('timeout', 30)
    if not isinstance(timeout, int) or timeout < 10 or timeout > 300:
        warnings.append("Timeout sebaiknya antara 10-300 detik")
    
    return {
        'status': len(errors) == 0,
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'config_summary': {
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

def validate_ui_inputs(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Validate UI inputs dan return validation result.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        
    Returns:
        Dictionary berisi hasil validasi dengan key 'status' dan 'errors'
    """
    # Extract values untuk validation
    workspace = getattr(ui_components.get('workspace_input'), 'value', '').strip()
    project = getattr(ui_components.get('project_input'), 'value', '').strip()
    version = getattr(ui_components.get('version_input'), 'value', '').strip()
    api_key = getattr(ui_components.get('api_key_input'), 'value', '').strip()
    
    # Create config structure for validation
    config = {
        'data': {
            'roboflow': {
                'workspace': workspace,
                'project': project,
                'version': version,
                'api_key': api_key
            }
        }
    }
    
    # Use centralized validation
    return validate_roboflow_config(config)

def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validasi konfigurasi lengkap dengan pemeriksaan komprehensif.
    
    Args:
        config: Dictionary konfigurasi yang akan divalidasi
        
    Returns:
        Dictionary berisi hasil validasi dengan key 'status' dan 'errors'
    """
    # Validate roboflow dan download config secara paralel
    roboflow_validation = validate_roboflow_config(config)
    download_validation = validate_download_config(config)
    
    # Combine results dengan list concatenation
    errors = roboflow_validation.get('errors', []) + download_validation.get('errors', [])
    warnings = roboflow_validation.get('warnings', []) + download_validation.get('warnings', [])
    
    # Extract values untuk summary
    values = roboflow_validation.get('values', {})
    config_summary = download_validation.get('config_summary', {})
    
    # Create target string dengan conditional expression
    target = (f"{values.get('workspace')}/{values.get('project')}:v{values.get('version')}" 
              if all(values.get(k) for k in ['workspace', 'project', 'version']) 
              else 'N/A')
    
    # Determine validity status
    is_valid = len(errors) == 0
    
    # Return hasil validasi dengan format yang konsisten
    return {
        'status': is_valid,
        'valid': is_valid,  # Backward compatibility
        'errors': errors,
        'warnings': warnings,
        'config_summary': {
            'target': target,
            'api_key_masked': values.get('api_key_masked', '****'),
            'uuid_rename': config_summary.get('uuid_rename', True),
            'validation': config_summary.get('validation', True),
            'backup': config_summary.get('backup', False)
        }
    }

def format_validation_summary(validation: Dict[str, Any], html_format: bool = False) -> str:
    """Format validation summary untuk display di UI.
    
    Args:
        validation: Validation result
        html_format: If True, return HTML formatted content for summary_container
        
    Returns:
        Formatted summary string untuk UI display
    """
    if html_format:
        # HTML format untuk summary_container
        if not validation.get('valid', False):
            # Format untuk invalid configuration
            html_lines = [
                "<div style='padding: 10px; background-color: #fff0f0; border-radius: 5px; border-left: 5px solid #ff6b6b;'>",
                "<h3>❌ Konfigurasi Tidak Valid</h3>",
                "<ul>"
            ]
            html_lines.extend([f"<li>{error}</li>" for error in validation.get('errors', [])])
            html_lines.append("</ul></div>")
            return "".join(html_lines)
        
        # Format untuk valid configuration
        config_summary = validation.get('config_summary', {})
        
        # Format status items dengan emoji indicators dan HTML styling
        html_lines = [
            "<div style='padding: 10px; background-color: #f0f8ff; border-radius: 5px; border-left: 5px solid #4682b4;'>",
            "<h3>✅ Konfigurasi Valid</h3>",
            "<table style='width: 100%;'>"
        ]
        
        # Add status items as table rows
        status_items = [
            ("🎯 Target", config_summary.get('target', 'N/A')),
            ("🔑 API Key", config_summary.get('api_key_masked', '****')),
            ("🔄 UUID Rename", "✅" if config_summary.get('uuid_rename') else "❌"),
            ("✅ Validasi", "✅" if config_summary.get('validation') else "❌"),
            ("💾 Backup", "✅" if config_summary.get('backup') else "❌")
        ]
        
        for label, value in status_items:
            html_lines.append(f"<tr><td><b>{label}</b></td><td>{value}</td></tr>")
        
        html_lines.append("</table>")
        
        # Add warnings jika ada
        warnings = validation.get('warnings', [])
        if warnings:
            html_lines.append("<h4>⚠️ Peringatan:</h4>")
            html_lines.append("<ul>")
            html_lines.extend([f"<li>{warning}</li>" for warning in warnings])
            html_lines.append("</ul>")
        
        html_lines.append("</div>")
        return "".join(html_lines)
    else:
        # Plain text format untuk log_output
        if not validation.get('valid', False):
            error_lines = ["❌ Konfigurasi tidak valid:", ""]
            error_lines.extend([f"  • {error}" for error in validation.get('errors', [])])
            return '\n'.join(error_lines)
        
        # Format untuk valid configuration
        config_summary = validation.get('config_summary', {})
        
        # Format status items dengan emoji indicators
        status_items = {
            'Target': f"🎯 {config_summary.get('target', 'N/A')}",
            'API Key': f"🔑 {config_summary.get('api_key_masked', '****')}",
            'UUID Rename': f"{'✅' if config_summary.get('uuid_rename') else '❌'}",
            'Validasi': f"{'✅' if config_summary.get('validation') else '❌'}",
            'Backup': f"{'✅' if config_summary.get('backup') else '❌'}"
        }
        
        # Build summary lines
        summary_lines = ["✅ Konfigurasi valid:"]
        summary_lines.extend([f"{key}: {value}" for key, value in status_items.items()])
        
        # Add warnings jika ada
        warnings = validation.get('warnings', [])  
        if warnings:
            summary_lines.extend(["", "⚠️ Peringatan:"])
            summary_lines.extend([f"  • {warning}" for warning in warnings])
        
        return '\n'.join(summary_lines)