"""
File: smartcash/ui/dataset/downloader/utils/operation_utils.py
Deskripsi: Fixed operation utilities dengan proper method calls dan safe one-liner patterns
"""

from typing import Dict, Any, Optional
from pathlib import Path
from smartcash.dataset.utils.path_validator import get_path_validator
from smartcash.common.environment import get_environment_manager
from smartcash.common.utils.one_liner_fixes import (
    safe_operation_or_none, fix_path_operation, 
    fix_directory_operation, safe_boolean_and_operation
)

class StreamlinedDownloadOperations:
    """Fixed download operations dengan proper method calls dan safe patterns"""
    
    def __init__(self):
        self.path_validator = get_path_validator()
        self.env_manager = get_environment_manager()
    
    def check_existing_dataset(self) -> bool:
        """Fixed existing dataset check dengan safe operation pattern"""
        def check_operation():
            paths = self.path_validator.get_dataset_paths()
            validation = self.path_validator.validate_dataset_structure(paths['data_root'])
            # Perbaikan: Pastikan validation.get('valid') dan validation.get('total_images', 0) > 0 adalah boolean
            # dan gunakan all() sebagai pengganti safe_boolean_and_operation
            return all([validation.get('valid', False), validation.get('total_images', 0) > 0])
        
        return bool(safe_operation_or_none(check_operation) or False)
    
    def get_dataset_summary(self) -> Dict[str, Any]:
        """Fixed dataset summary dengan safe validation pattern"""
        def summary_operation():
            paths = self.path_validator.get_dataset_paths()
            validation = self.path_validator.validate_dataset_structure(paths['data_root'])
            
            return {
                'exists': validation.get('valid', False),
                'total_images': validation.get('total_images', 0),
                'total_labels': validation.get('total_labels', 0),
                'issues_count': len(validation.get('issues', [])),
                'splits': {k: v for k, v in validation.get('splits', {}).items() if v.get('exists', False)},
                'data_root': paths['data_root'],
                'drive_storage': self.env_manager.is_drive_mounted
            }
        
        return safe_operation_or_none(summary_operation) or {
            'exists': False, 'total_images': 0, 'total_labels': 0, 
            'issues_count': 0, 'splits': {}, 'data_root': '', 'drive_storage': False
        }
    
    def validate_download_space(self, required_mb: float) -> Dict[str, Any]:
        """Fixed space validation dengan safe disk usage check"""
        def space_operation():
            paths = self.path_validator.get_dataset_paths()
            target_path = fix_path_operation(paths['data_root'], 'mkdir', parents=True, exist_ok=True)
            
            import shutil
            total, used, free = shutil.disk_usage(target_path.parent if target_path.parent.exists() else target_path)
            free_mb = free / (1024 * 1024)
            
            return {
                'sufficient': free_mb >= required_mb,
                'free_mb': free_mb,
                'required_mb': required_mb,
                'shortage_mb': max(0, required_mb - free_mb)
            }
        
        return safe_operation_or_none(space_operation) or {
            'sufficient': False, 'free_mb': 0, 'required_mb': required_mb, 'shortage_mb': required_mb
        }
    
    def get_download_paths(self, dataset_identifier: str) -> Dict[str, str]:
        """Fixed download paths dengan safe path operations"""
        def paths_operation():
            paths = self.path_validator.get_dataset_paths()
            downloads_base = paths.get('downloads', f"{paths['data_root']}/downloads")
            
            # Safe identifier cleaning
            clean_identifier = dataset_identifier.replace('/', '_').replace(':', '_v') if dataset_identifier else 'unknown'
            temp_download = f"{downloads_base}/{clean_identifier}"
            
            return {
                'data_root': paths['data_root'],
                'temp_download': temp_download,
                'downloads_base': downloads_base,
                'train': paths['train'],
                'valid': paths['valid'],
                'test': paths['test'],
                'backup': paths.get('backup', f"{paths['data_root']}/backup")
            }
        
        return safe_operation_or_none(paths_operation) or {'data_root': '', 'temp_download': '', 'downloads_base': '', 'train': '', 'valid': '', 'test': '', 'backup': ''}
    
    def create_backup_if_needed(self, backup_enabled: bool = True) -> Dict[str, Any]:
        """Fixed backup creation dengan safe operations"""
        def backup_operation():
            if not backup_enabled or not self.check_existing_dataset():
                return {'created': False, 'message': 'Backup not needed or no existing data'}
            
            import shutil
            from datetime import datetime
            
            paths = self.path_validator.get_dataset_paths()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = paths.get('backup', f"{paths['data_root']}/backup")
            backup_path = f"{backup_dir}/dataset_backup_{timestamp}"
            
            # Safe directory operations
            fix_directory_operation(backup_dir, 'create', parents=True, exist_ok=True)
            
            # Safe copy operation
            shutil.copytree(
                paths['data_root'], backup_path,
                ignore=shutil.ignore_patterns('backup', 'downloads', '*.tmp'),
                dirs_exist_ok=True
            )
            
            return {
                'created': True,
                'backup_path': backup_path,
                'message': f'Backup created: {backup_path}'
            }
        
        return safe_operation_or_none(backup_operation) or {'created': False, 'message': 'Backup operation failed'}
    
    def cleanup_temp_files(self, dataset_identifier: str = None) -> Dict[str, Any]:
        """Fixed temp cleanup dengan safe file operations"""
        def cleanup_operation():
            paths = self.path_validator.get_dataset_paths()
            downloads_path = fix_path_operation(paths.get('downloads', f"{paths['data_root']}/downloads"), 'exists')
            
            if not downloads_path or not downloads_path.exists():
                return {'cleaned': 0, 'message': 'No temp files to clean'}
            
            cleaned_count = 0
            
            # Safe file pattern matching
            if dataset_identifier:
                temp_pattern = dataset_identifier.replace('/', '_').replace(':', '_v')
                temp_files = list(downloads_path.glob(f"*{temp_pattern}*"))
            else:
                temp_files = list(downloads_path.rglob('*'))
            
            # Safe file removal
            for temp_file in temp_files:
                try:
                    if temp_file.is_file():
                        temp_file.unlink()
                        cleaned_count += 1
                    elif temp_file.is_dir():
                        fix_directory_operation(str(temp_file), 'remove', ignore_errors=True)
                        cleaned_count += 1
                except Exception:
                    continue
            
            return {'cleaned': cleaned_count, 'message': f'Cleaned {cleaned_count} temp files'}
        
        return safe_operation_or_none(cleanup_operation) or {'cleaned': 0, 'message': 'Cleanup operation failed'}
    
    def estimate_download_time(self, size_mb: float, connection_speed_mbps: float = 10.0) -> Dict[str, Any]:
        """Fixed download time estimation dengan safe calculations"""
        def estimate_operation():
            # Safe calculation with fallbacks
            if size_mb <= 0 or connection_speed_mbps <= 0:
                return {'estimated_seconds': 0, 'formatted_time': 'Unknown', 'size_mb': size_mb, 'speed_mbps': connection_speed_mbps}
            
            size_mb_bits = size_mb * 8
            time_seconds = size_mb_bits / connection_speed_mbps
            
            # Safe time formatting
            if time_seconds < 60:
                formatted_time = f"{time_seconds:.0f} detik"
            elif time_seconds < 3600:
                formatted_time = f"{time_seconds/60:.1f} menit"
            else:
                formatted_time = f"{time_seconds/3600:.1f} jam"
            
            return {
                'estimated_seconds': time_seconds,
                'formatted_time': formatted_time,
                'size_mb': size_mb,
                'speed_mbps': connection_speed_mbps
            }
        
        return safe_operation_or_none(estimate_operation) or {'estimated_seconds': 0, 'formatted_time': 'Unknown', 'size_mb': size_mb, 'speed_mbps': connection_speed_mbps}
    
    def format_dataset_info(self, metadata: Dict[str, Any]) -> str:
        """Fixed dataset info formatting dengan safe data extraction"""
        def format_operation():
            # Safe metadata extraction
            project_info = metadata.get('project', {}) if metadata else {}
            version_info = metadata.get('version', {}) if metadata else {}
            export_info = metadata.get('export', {}) if metadata else {}
            
            classes = project_info.get('classes', [])
            images = version_info.get('images', 0)
            size_mb = export_info.get('size', 0)
            
            # Safe class list formatting
            class_display = ', '.join(classes[:5]) if classes else 'Unknown'
            if len(classes) > 5:
                class_display += '...'
            
            info_lines = [
                "📊 **Dataset Information**",
                f"🏷️ Classes: {len(classes)} ({class_display})",
                f"🖼️ Images: {images:,}",
                f"💾 Size: {size_mb:.1f} MB",
                "📦 Format: YOLOv5 PyTorch"
            ]
            
            # Safe download estimate
            estimate = self.estimate_download_time(size_mb)
            if estimate.get('formatted_time') != 'Unknown':
                info_lines.append(f"⏱️ Est. Download: {estimate['formatted_time']}")
            
            return "\n".join(info_lines)
        
        return safe_operation_or_none(format_operation) or "❌ Error formatting dataset info"
    
    def get_operation_status(self) -> Dict[str, Any]:
        """Fixed operation status dengan safe dataset summary"""
        def status_operation():
            dataset_summary = self.get_dataset_summary()
            
            return {
                'ready_for_download': not dataset_summary.get('exists', False) or dataset_summary.get('issues_count', 0) > 0,
                'ready_for_check': True,
                'ready_for_cleanup': dataset_summary.get('exists', False),
                'environment': {
                    'is_colab': self.env_manager.is_colab,
                    'drive_mounted': self.env_manager.is_drive_mounted,
                    'storage_location': 'Google Drive' if self.env_manager.is_drive_mounted else 'Local'
                },
                'dataset_summary': dataset_summary
            }
        
        return safe_operation_or_none(status_operation) or {
            'ready_for_download': True, 'ready_for_check': True, 'ready_for_cleanup': False,
            'environment': {'is_colab': False, 'drive_mounted': False, 'storage_location': 'Unknown'},
            'dataset_summary': {'exists': False, 'total_images': 0}
        }

# Fixed singleton pattern
_download_operations = None

def get_streamlined_download_operations() -> StreamlinedDownloadOperations:
    """Fixed singleton factory dengan safe initialization"""
    global _download_operations
    if _download_operations is None:
        _download_operations = safe_operation_or_none(StreamlinedDownloadOperations) or StreamlinedDownloadOperations()
    return _download_operations

# Fixed utility functions dengan safe operations
def check_dataset_exists() -> bool:
    """Fixed dataset existence check"""
    operations = get_streamlined_download_operations()
    return safe_operation_or_none(operations.check_existing_dataset) or False

def get_dataset_info() -> Dict[str, Any]:
    """Fixed dataset info retrieval"""
    operations = get_streamlined_download_operations()
    return safe_operation_or_none(operations.get_dataset_summary) or {'exists': False, 'total_images': 0}

def validate_space(mb: float) -> Dict[str, Any]:
    """Fixed space validation"""
    operations = get_streamlined_download_operations()
    return safe_operation_or_none(lambda: operations.validate_download_space(mb)) or {'sufficient': False}

def cleanup_temps(dataset_id: str = None) -> Dict[str, Any]:
    """Fixed temp cleanup"""
    operations = get_streamlined_download_operations()
    return safe_operation_or_none(lambda: operations.cleanup_temp_files(dataset_id)) or {'cleaned': 0}

def estimate_time(size_mb: float) -> Dict[str, Any]:
    """Fixed time estimation"""
    operations = get_streamlined_download_operations()
    return safe_operation_or_none(lambda: operations.estimate_download_time(size_mb)) or {'estimated_seconds': 0}

def get_status() -> Dict[str, Any]:
    """Fixed status retrieval"""
    operations = get_streamlined_download_operations()
    return safe_operation_or_none(operations.get_operation_status) or {'ready_for_download': True}

def format_size(bytes_size: int) -> str:
    """Fixed size formatting dengan safe calculations"""
    def format_operation():
        if bytes_size <= 0:
            return "0 B"
        
        units = ['B', 'KB', 'MB', 'GB']
        size = float(bytes_size)
        unit_index = 0
        
        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1
        
        return f"{size:.1f} {units[unit_index]}"
    
    return safe_operation_or_none(format_operation) or f"{bytes_size} bytes"

def validate_dataset_identifier(workspace: str, project: str, version: str) -> Dict[str, Any]:
    """Fixed dataset identifier validation dengan safe string operations"""
    def validate_operation():
        # Safe string operations
        workspace_clean = workspace.strip() if workspace else ""
        project_clean = project.strip() if project else ""
        version_clean = version.strip() if version else ""
        
        validation_rules = [
            (bool(workspace_clean), "Workspace tidak boleh kosong"),
            (bool(project_clean), "Project tidak boleh kosong"),
            (bool(version_clean), "Version tidak boleh kosong")
        ]
        
        # Safe character validation
        if workspace_clean:
            workspace_valid = workspace_clean.replace('-', '').replace('_', '').isalnum()
            validation_rules.append((workspace_valid, "Workspace hanya boleh huruf, angka, dash, underscore"))
        
        if project_clean:
            project_valid = project_clean.replace('-', '').replace('_', '').isalnum()
            validation_rules.append((project_valid, "Project hanya boleh huruf, angka, dash, underscore"))
        
        if version_clean:
            version_valid = version_clean.replace('.', '').isalnum()
            validation_rules.append((version_valid, "Version hanya boleh huruf, angka, titik"))
        
        errors = [message for valid, message in validation_rules if not valid]
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'identifier': f"{workspace_clean}/{project_clean}:v{version_clean}" if not errors else None
        }
    
    return safe_operation_or_none(validate_operation) or {'valid': False, 'errors': ['Validation failed'], 'identifier': None}

# Fixed utility functions dengan safe patterns
def create_temp_path(base: str, identifier: str) -> str:
    """Fixed temp path creation"""
    if not base or not identifier:
        return ""
    clean_identifier = identifier.replace('/', '_').replace(':', '_v')
    return f"{base}/{clean_identifier}"

def get_downloads_path(data_root: str) -> str:
    """Fixed downloads path"""
    return f"{data_root}/downloads" if data_root else "downloads"

def get_backup_path(data_root: str) -> str:
    """Fixed backup path"""
    return f"{data_root}/backup" if data_root else "backup"

def format_download_summary(summary: Dict[str, Any]) -> str:
    """Fixed download summary formatting"""
    if not summary:
        return "No summary available"
    
    total_images = summary.get('total_images', 0)
    issues_count = summary.get('issues_count', 0)
    drive_storage = summary.get('drive_storage', False)
    
    return f"Images: {total_images:,} | Issues: {issues_count} | Drive: {'✅' if drive_storage else '❌'}"

# Backward compatibility alias
consolidate_download_operations = get_streamlined_download_operations