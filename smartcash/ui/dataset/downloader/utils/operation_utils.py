"""
File: smartcash/ui/dataset/downloader/utils/operation_utils.py
Deskripsi: Fixed operation utilities dengan proper method calls dan one-liner style
"""

from typing import Dict, Any, Optional
from pathlib import Path
from smartcash.dataset.utils.path_validator import get_path_validator
from smartcash.common.environment import get_environment_manager

class StreamlinedDownloadOperations:
    """Fixed download operations dengan proper method calls"""
    
    def __init__(self):
        self.path_validator = get_path_validator()
        self.env_manager = get_environment_manager()
    
    def check_existing_dataset(self) -> bool:
        """Fixed existing dataset check dengan proper method call"""
        try:
            paths = self.path_validator.get_dataset_paths()
            # Fixed: call validate_dataset_structure as method, not property
            validation = self.path_validator.validate_dataset_structure(paths['data_root'])
            return validation['valid'] and validation['total_images'] > 0
        except Exception:
            return False
    
    def get_dataset_summary(self) -> Dict[str, Any]:
        """Fixed dataset summary dengan proper validation call"""
        try:
            paths = self.path_validator.get_dataset_paths()
            # Fixed: proper method call
            validation = self.path_validator.validate_dataset_structure(paths['data_root'])
            
            return {
                'exists': validation['valid'], 'total_images': validation['total_images'],
                'total_labels': validation['total_labels'], 'issues_count': len(validation.get('issues', [])),
                'splits': {k: v for k, v in validation['splits'].items() if v.get('exists', False)},
                'data_root': paths['data_root'], 'drive_storage': self.env_manager.is_drive_mounted
            }
        except Exception as e:
            return {'exists': False, 'error': str(e), 'total_images': 0, 'total_labels': 0, 'splits': {}}
    
    def validate_download_space(self, required_mb: float) -> Dict[str, Any]:
        """Fixed space validation dengan proper error handling"""
        try:
            import shutil
            paths = self.path_validator.get_dataset_paths()
            target_path = Path(paths['data_root'])
            
            # Ensure parent exists
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            total, used, free = shutil.disk_usage(target_path.parent)
            free_mb = free / (1024 * 1024)
            
            return {
                'sufficient': free_mb >= required_mb, 'free_mb': free_mb,
                'required_mb': required_mb, 'shortage_mb': max(0, required_mb - free_mb)
            }
        except Exception as e:
            return {'sufficient': False, 'error': str(e), 'free_mb': 0, 'required_mb': required_mb}
    
    def get_download_paths(self, dataset_identifier: str) -> Dict[str, str]:
        """Fixed download paths dengan environment awareness"""
        paths = self.path_validator.get_dataset_paths()
        
        # Create temp download path
        downloads_base = paths.get('downloads', f"{paths['data_root']}/downloads")
        temp_download = f"{downloads_base}/{dataset_identifier.replace('/', '_').replace(':', '_v')}"
        
        return {
            'data_root': paths['data_root'], 'temp_download': temp_download,
            'downloads_base': downloads_base, 'train': paths['train'],
            'valid': paths['valid'], 'test': paths['test'],
            'backup': paths.get('backup', f"{paths['data_root']}/backup")
        }
    
    def create_backup_if_needed(self, backup_enabled: bool = True) -> Dict[str, Any]:
        """Fixed backup creation dengan proper dataset check"""
        if not backup_enabled or not self.check_existing_dataset():
            return {'created': False, 'message': 'Backup not needed or no existing data'}
        
        try:
            import shutil
            from datetime import datetime
            
            paths = self.path_validator.get_dataset_paths()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = paths.get('backup', f"{paths['data_root']}/backup")
            backup_path = f"{backup_dir}/dataset_backup_{timestamp}"
            
            # Copy existing data
            shutil.copytree(paths['data_root'], backup_path, 
                          ignore=shutil.ignore_patterns('backup', 'downloads', '*.tmp'))
            
            return {'created': True, 'backup_path': backup_path, 'message': f'Backup created: {backup_path}'}
            
        except Exception as e:
            return {'created': False, 'error': str(e), 'message': f'Backup failed: {str(e)}'}
    
    def cleanup_temp_files(self, dataset_identifier: str = None) -> Dict[str, Any]:
        """Fixed temp cleanup dengan proper path handling"""
        try:
            paths = self.path_validator.get_dataset_paths()
            downloads_path = Path(paths.get('downloads', f"{paths['data_root']}/downloads"))
            
            if not downloads_path.exists():
                return {'cleaned': 0, 'message': 'No temp files to clean'}
            
            cleaned_count = 0
            
            if dataset_identifier:
                # Cleanup specific dataset temp files
                temp_pattern = dataset_identifier.replace('/', '_').replace(':', '_v')
                temp_files = list(downloads_path.glob(f"*{temp_pattern}*"))
            else:
                # Cleanup all temp files
                temp_files = list(downloads_path.rglob('*'))
            
            for temp_file in temp_files:
                try:
                    if temp_file.is_file():
                        temp_file.unlink()
                        cleaned_count += 1
                    elif temp_file.is_dir():
                        import shutil
                        shutil.rmtree(temp_file, ignore_errors=True)
                        cleaned_count += 1
                except Exception:
                    continue
            
            return {'cleaned': cleaned_count, 'message': f'Cleaned {cleaned_count} temp files'}
            
        except Exception as e:
            return {'cleaned': 0, 'error': str(e), 'message': f'Cleanup failed: {str(e)}'}
    
    def estimate_download_time(self, size_mb: float, connection_speed_mbps: float = 10.0) -> Dict[str, Any]:
        """Fixed download time estimation"""
        try:
            # Convert MB to Mb dan calculate time
            size_mb_bits = size_mb * 8
            time_seconds = size_mb_bits / connection_speed_mbps
            
            # Format time
            formatted_time = (f"{time_seconds:.0f} detik" if time_seconds < 60 else
                            f"{time_seconds/60:.1f} menit" if time_seconds < 3600 else
                            f"{time_seconds/3600:.1f} jam")
            
            return {
                'estimated_seconds': time_seconds, 'formatted_time': formatted_time,
                'size_mb': size_mb, 'speed_mbps': connection_speed_mbps
            }
            
        except Exception as e:
            return {'estimated_seconds': 0, 'formatted_time': 'Unknown', 'error': str(e)}
    
    def format_dataset_info(self, metadata: Dict[str, Any]) -> str:
        """Fixed dataset info formatting"""
        try:
            project_info, version_info, export_info = metadata.get('project', {}), metadata.get('version', {}), metadata.get('export', {})
            classes, images, size_mb = project_info.get('classes', []), version_info.get('images', 0), export_info.get('size', 0)
            
            info_lines = [
                f"📊 **Dataset Information**",
                f"🏷️ Classes: {len(classes)} ({', '.join(classes[:5])}{'...' if len(classes) > 5 else ''})",
                f"🖼️ Images: {images:,}", f"💾 Size: {size_mb:.1f} MB",
                f"📦 Format: YOLOv5 PyTorch (hardcoded)"
            ]
            
            # Add download estimate
            estimate = self.estimate_download_time(size_mb)
            estimate.get('formatted_time') and info_lines.append(f"⏱️ Est. Download: {estimate['formatted_time']}")
            
            return "\n".join(info_lines)
            
        except Exception as e:
            return f"❌ Error formatting dataset info: {str(e)}"
    
    def get_operation_status(self) -> Dict[str, Any]:
        """Fixed operation status dengan proper dataset summary"""
        dataset_summary = self.get_dataset_summary()
        
        return {
            'ready_for_download': not dataset_summary['exists'] or dataset_summary['issues_count'] > 0,
            'ready_for_check': True, 'ready_for_cleanup': dataset_summary['exists'],
            'environment': {
                'is_colab': self.env_manager.is_colab, 'drive_mounted': self.env_manager.is_drive_mounted,
                'storage_location': 'Google Drive' if self.env_manager.is_drive_mounted else 'Local'
            },
            'dataset_summary': dataset_summary
        }

# Fixed singleton instance
_download_operations = None

def get_streamlined_download_operations() -> StreamlinedDownloadOperations:
    """Fixed singleton factory"""
    global _download_operations
    if _download_operations is None:
        _download_operations = StreamlinedDownloadOperations()
    return _download_operations

# Fixed utility functions
check_dataset_exists = lambda: get_streamlined_download_operations().check_existing_dataset()
get_dataset_info = lambda: get_streamlined_download_operations().get_dataset_summary()
validate_space = lambda mb: get_streamlined_download_operations().validate_download_space(mb)
cleanup_temps = lambda dataset_id=None: get_streamlined_download_operations().cleanup_temp_files(dataset_id)
estimate_time = lambda size_mb: get_streamlined_download_operations().estimate_download_time(size_mb)
get_status = lambda: get_streamlined_download_operations().get_operation_status()

def format_size(bytes_size: int) -> str:
    """Fixed size formatting"""
    units, size, unit_index = ['B', 'KB', 'MB', 'GB'], float(bytes_size), 0
    
    while size >= 1024 and unit_index < len(units) - 1:
        size, unit_index = size / 1024, unit_index + 1
    
    return f"{size:.1f} {units[unit_index]}"

def validate_dataset_identifier(workspace: str, project: str, version: str) -> Dict[str, Any]:
    """Fixed dataset identifier validation"""
    validation_rules = [
        (bool(workspace.strip()), "Workspace tidak boleh kosong"),
        (bool(project.strip()), "Project tidak boleh kosong"), 
        (bool(version.strip()), "Version tidak boleh kosong"),
        (workspace.replace('-', '').replace('_', '').isalnum(), "Workspace hanya boleh huruf, angka, dash, underscore"),
        (project.replace('-', '').replace('_', '').isalnum(), "Project hanya boleh huruf, angka, dash, underscore"),
        (version.replace('.', '').isalnum(), "Version hanya boleh huruf, angka, titik")
    ]
    
    errors = [message for valid, message in validation_rules if not valid]
    
    return {
        'valid': len(errors) == 0, 'errors': errors,
        'identifier': f"{workspace}/{project}:v{version}" if not errors else None
    }

# Fixed utility functions
create_temp_path = lambda base, identifier: f"{base}/{identifier.replace('/', '_').replace(':', '_v')}"
get_downloads_path = lambda data_root: f"{data_root}/downloads"
get_backup_path = lambda data_root: f"{data_root}/backup"
format_download_summary = lambda summary: f"Images: {summary.get('total_images', 0):,} | Issues: {summary.get('issues_count', 0)} | Drive: {'✅' if summary.get('drive_storage') else '❌'}"

# Backward compatibility alias
consolidate_download_operations = get_streamlined_download_operations