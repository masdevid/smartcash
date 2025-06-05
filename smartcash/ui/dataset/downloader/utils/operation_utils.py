"""
File: smartcash/ui/dataset/downloader/utils/operation_utils.py
Deskripsi: Consolidated utilities untuk download operations dengan one-liner style
"""

from typing import Dict, Any, Optional
from pathlib import Path
from smartcash.dataset.utils.path_validator import get_path_validator
from smartcash.common.environment import get_environment_manager

class DownloadOperations:
    """Consolidated operations untuk download module dengan reusable methods"""
    
    def __init__(self):
        self.path_validator = get_path_validator()
        self.env_manager = get_environment_manager()
    
    def check_existing_dataset(self) -> bool:
        """Check apakah ada existing dataset dengan one-liner"""
        try:
            paths = self.path_validator.get_dataset_paths()
            validation = self.path_validator.validate_dataset_structure(paths['data_root'])
            return validation['valid'] and validation['total_images'] > 0
        except Exception:
            return False
    
    def get_dataset_summary(self) -> Dict[str, Any]:
        """Get comprehensive dataset summary dengan one-liner info"""
        try:
            paths = self.path_validator.get_dataset_paths()
            validation = self.path_validator.validate_dataset_structure(paths['data_root'])
            
            return {
                'exists': validation['valid'],
                'total_images': validation['total_images'],
                'total_labels': validation['total_labels'],
                'splits': {k: v for k, v in validation['splits'].items() if v['exists']},
                'issues_count': len(validation.get('issues', [])),
                'data_root': paths['data_root'],
                'drive_storage': self.env_manager.is_drive_mounted
            }
        except Exception as e:
            return {'exists': False, 'error': str(e), 'total_images': 0, 'total_labels': 0, 'splits': {}}
    
    def validate_download_space(self, required_mb: float) -> Dict[str, Any]:
        """Validate disk space untuk download dengan one-liner check"""
        try:
            import shutil
            paths = self.path_validator.get_dataset_paths()
            target_path = Path(paths['data_root'])
            
            # Ensure parent exists untuk disk usage check
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            total, used, free = shutil.disk_usage(target_path.parent)
            free_mb = free / (1024 * 1024)
            
            return {
                'sufficient': free_mb >= required_mb,
                'free_mb': free_mb,
                'required_mb': required_mb,
                'shortage_mb': max(0, required_mb - free_mb)
            }
        except Exception as e:
            return {'sufficient': False, 'error': str(e), 'free_mb': 0, 'required_mb': required_mb}
    
    def get_download_paths(self, dataset_identifier: str) -> Dict[str, str]:
        """Get download paths untuk dataset dengan environment awareness"""
        paths = self.path_validator.get_dataset_paths()
        
        # Create temp download path
        downloads_base = paths.get('downloads', f"{paths['data_root']}/downloads")
        temp_download = f"{downloads_base}/{dataset_identifier.replace('/', '_').replace(':', '_v')}"
        
        return {
            'data_root': paths['data_root'],
            'temp_download': temp_download,
            'downloads_base': downloads_base,
            'train': paths['train'],
            'valid': paths['valid'],
            'test': paths['test'],
            'backup': paths.get('backup', f"{paths['data_root']}/backup")
        }
    
    def create_backup_if_needed(self, backup_enabled: bool = True) -> Dict[str, Any]:
        """Create backup dataset jika diperlukan dengan one-liner check"""
        if not backup_enabled or not self.check_existing_dataset():
            return {'created': False, 'message': 'Backup not needed or no existing data'}
        
        try:
            import shutil
            from datetime import datetime
            
            paths = self.path_validator.get_dataset_paths()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{paths.get('backup', f'{paths['data_root']}/backup')}/dataset_backup_{timestamp}"
            
            # Copy existing data
            shutil.copytree(paths['data_root'], backup_path, ignore=shutil.ignore_patterns('backup', 'downloads', '*.tmp'))
            
            return {
                'created': True,
                'backup_path': backup_path,
                'message': f'Backup created: {backup_path}'
            }
            
        except Exception as e:
            return {'created': False, 'error': str(e), 'message': f'Backup failed: {str(e)}'}
    
    def cleanup_temp_files(self, dataset_identifier: str = None) -> Dict[str, Any]:
        """Cleanup temporary download files dengan selective cleanup"""
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
            
            return {
                'cleaned': cleaned_count,
                'message': f'Cleaned {cleaned_count} temp files'
            }
            
        except Exception as e:
            return {'cleaned': 0, 'error': str(e), 'message': f'Cleanup failed: {str(e)}'}
    
    def estimate_download_time(self, size_mb: float, connection_speed_mbps: float = 10.0) -> Dict[str, Any]:
        """Estimate download time dengan one-liner calculation"""
        try:
            # Convert MB to Mb (megabits) dan calculate time
            size_mb_bits = size_mb * 8
            time_seconds = size_mb_bits / connection_speed_mbps
            
            # Format time
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
            
        except Exception as e:
            return {'estimated_seconds': 0, 'formatted_time': 'Unknown', 'error': str(e)}
    
    def format_dataset_info(self, metadata: Dict[str, Any]) -> str:
        """Format dataset info untuk display dengan emoji formatting"""
        try:
            project_info = metadata.get('project', {})
            version_info = metadata.get('version', {})
            export_info = metadata.get('export', {})
            
            classes = project_info.get('classes', [])
            images = version_info.get('images', 0)
            size_mb = export_info.get('size', 0)
            
            info_lines = [
                f"📊 **Dataset Information**",
                f"🏷️ Classes: {len(classes)} ({', '.join(classes[:5])}{'...' if len(classes) > 5 else ''})",
                f"🖼️ Images: {images:,}",
                f"💾 Size: {size_mb:.1f} MB"
            ]
            
            # Add download estimate
            estimate = self.estimate_download_time(size_mb)
            if estimate.get('formatted_time'):
                info_lines.append(f"⏱️ Est. Download: {estimate['formatted_time']}")
            
            return "\n".join(info_lines)
            
        except Exception as e:
            return f"❌ Error formatting dataset info: {str(e)}"
    
    def get_operation_status(self) -> Dict[str, Any]:
        """Get comprehensive operation status dengan one-liner summary"""
        dataset_summary = self.get_dataset_summary()
        
        return {
            'ready_for_download': not dataset_summary['exists'] or dataset_summary['issues_count'] > 0,
            'ready_for_check': True,  # Check selalu bisa dilakukan
            'ready_for_cleanup': dataset_summary['exists'],
            'environment': {
                'is_colab': self.env_manager.is_colab,
                'drive_mounted': self.env_manager.is_drive_mounted,
                'storage_location': 'Google Drive' if self.env_manager.is_drive_mounted else 'Local'
            },
            'dataset_summary': dataset_summary
        }

# Singleton instance
_download_operations = None

def consolidate_download_operations() -> DownloadOperations:
    """Get singleton instance dengan one-liner factory"""
    global _download_operations
    if _download_operations is None:
        _download_operations = DownloadOperations()
    return _download_operations

# One-liner utilities untuk common operations
check_dataset_exists = lambda: consolidate_download_operations().check_existing_dataset()
get_dataset_info = lambda: consolidate_download_operations().get_dataset_summary()
validate_space = lambda mb: consolidate_download_operations().validate_download_space(mb)
cleanup_temps = lambda dataset_id=None: consolidate_download_operations().cleanup_temp_files(dataset_id)
estimate_time = lambda size_mb: consolidate_download_operations().estimate_download_time(size_mb)
get_status = lambda: consolidate_download_operations().get_operation_status()

def format_size(bytes_size: int) -> str:
    """Format bytes ke human readable dengan one-liner"""
    units = ['B', 'KB', 'MB', 'GB']
    size = float(bytes_size)
    unit_index = 0
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    return f"{size:.1f} {units[unit_index]}"

def validate_dataset_identifier(workspace: str, project: str, version: str) -> Dict[str, Any]:
    """Validate dataset identifier components dengan one-liner checks"""
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
        'valid': len(errors) == 0,
        'errors': errors,
        'identifier': f"{workspace}/{project}:v{version}" if not errors else None
    }