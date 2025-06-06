"""
File: smartcash/dataset/downloader/validators.py
Deskripsi: Optimized validators dengan one-liner methods dan parallel validation
"""

import zipfile
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor
from smartcash.common.logger import get_logger

class DatasetValidator:
    """Optimized validator dengan one-liner methods dan parallel processing."""
    
    def __init__(self, logger=None, max_workers: int = 4):
        self.logger, self.max_workers = logger or get_logger(), max_workers
    
    def validate_zip_file(self, zip_path: Path) -> Dict[str, Any]:
        """One-liner optimized ZIP validation"""
        try:
            # One-liner existence dan format check
            not zip_path.exists() and self._return_error('ZIP file tidak ditemukan')
            not zipfile.is_zipfile(zip_path) and self._return_error('File bukan ZIP yang valid')
            
            # One-liner content validation
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = zip_ref.infolist()
                not file_list and self._return_error('ZIP file kosong')
                
                # One-liner structure analysis
                has_images = any('images' in f.filename.lower() or f.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')) for f in file_list)
                has_labels = any('labels' in f.filename.lower() or f.filename.lower().endswith('.txt') for f in file_list)
                
                return {
                    'valid': True, 'total_files': len(file_list), 'has_images': has_images, 'has_labels': has_labels,
                    'size_mb': zip_path.stat().st_size / 1048576, 'message': f'✅ ZIP valid: {len(file_list)} files'
                }
            
        except Exception as e:
            return {'valid': False, 'message': f'❌ Error validasi ZIP: {str(e)}'}
    
    def validate_extracted_dataset(self, dataset_dir: Path) -> Dict[str, Any]:
        """Optimized extracted dataset validation dengan parallel processing"""
        try:
            not dataset_dir.exists() and self._return_error('Dataset directory tidak ditemukan')
            
            validation_result = {'valid': True, 'splits': {}, 'total_images': 0, 'total_labels': 0, 'issues': [], 'structure_type': 'unknown'}
            
            # One-liner structure detection
            structure = self._detect_dataset_structure_optimized(dataset_dir)
            validation_result['structure_type'] = structure['type']
            
            if structure['type'] == 'split_based':
                # Parallel split validation
                validation_result.update(self._validate_splits_parallel(dataset_dir, structure['splits']))
            elif structure['type'] == 'flat':
                # One-liner flat validation
                validation_result.update(self._validate_flat_structure_optimized(dataset_dir))
            else:
                validation_result['valid'], validation_result['issues'] = False, ['Struktur dataset tidak dikenali']
            
            # One-liner overall validation
            validation_result['total_images'] == 0 and validation_result.update({'valid': False}) and validation_result['issues'].append('Tidak ada gambar ditemukan')
            
            return validation_result
            
        except Exception as e:
            return {'valid': False, 'message': f'❌ Error validasi dataset: {str(e)}'}
    
    def _detect_dataset_structure_optimized(self, dataset_dir: Path) -> Dict[str, Any]:
        """One-liner optimized structure detection"""
        # One-liner split detection
        splits = [split for split in ['train', 'valid', 'test', 'val'] 
                 if (dataset_dir / split).exists() and (dataset_dir / split / 'images').exists()]
        
        # One-liner fallback flat detection
        return ({'type': 'split_based', 'splits': splits} if splits else
                {'type': 'flat', 'splits': []} if (dataset_dir / 'images').exists() or any(dataset_dir.glob('*.jpg')) else
                {'type': 'unknown', 'splits': []})
    
    def _validate_splits_parallel(self, dataset_dir: Path, splits: List[str]) -> Dict[str, Any]:
        """Parallel split validation dengan optimized aggregation"""
        validation_result = {'splits': {}, 'total_images': 0, 'total_labels': 0, 'issues': []}
        
        # Parallel validation
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            split_futures = {split: executor.submit(self._validate_split_optimized, dataset_dir / split, split) for split in splits}
            
            # One-liner result aggregation
            [validation_result['splits'].update({split: result}) and
             validation_result.update({
                 'total_images': validation_result['total_images'] + result.get('images', 0),
                 'total_labels': validation_result['total_labels'] + result.get('labels', 0)
             }) and
             self._add_split_issues(validation_result['issues'], split, result)
             for split, future in split_futures.items() for result in [future.result()]]
        
        return validation_result
    
    def _add_split_issues(self, issues: List[str], split: str, result: Dict[str, Any]) -> None:
        """One-liner issue addition dengan validation checks"""
        result.get('images', 0) == 0 and issues.append(f"Split {split}: Tidak ada gambar")
        result.get('labels', 0) == 0 and issues.append(f"Split {split}: Tidak ada label")
        abs(result.get('images', 0) - result.get('labels', 0)) > result.get('images', 0) * 0.1 and issues.append(f"Split {split}: Mismatch gambar vs label")
    
    def _validate_split_optimized(self, split_dir: Path, split_name: str) -> Dict[str, Any]:
        """One-liner optimized split validation"""
        if not split_dir.exists():
            return {'exists': False, 'images': 0, 'labels': 0, 'issues': [f'Split {split_name} tidak ditemukan']}
        
        images_dir, labels_dir = split_dir / 'images', split_dir / 'labels'
        
        # One-liner file counting dengan extension filtering
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_count = len([f for f in images_dir.glob('*.*') if f.suffix.lower() in image_extensions]) if images_dir.exists() else 0
        label_count = len(list(labels_dir.glob('*.txt'))) if labels_dir.exists() else 0
        
        # One-liner issue detection
        issues = []
        not images_dir.exists() and issues.append(f'Folder images tidak ditemukan dalam {split_name}')
        not labels_dir.exists() and issues.append(f'Folder labels tidak ditemukan dalam {split_name}')
        
        return {'exists': True, 'images': image_count, 'labels': label_count, 'issues': issues}
    
    def _validate_flat_structure_optimized(self, dataset_dir: Path) -> Dict[str, Any]:
        """One-liner optimized flat structure validation"""
        images_dir, labels_dir = dataset_dir / 'images', dataset_dir / 'labels'
        
        # One-liner file counting
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        total_images = (len([f for f in images_dir.glob('*.*') if f.suffix.lower() in image_extensions]) if images_dir.exists() 
                       else len([f for f in dataset_dir.glob('*.*') if f.suffix.lower() in image_extensions]))
        total_labels = (len(list(labels_dir.glob('*.txt'))) if labels_dir.exists() 
                       else len(list(dataset_dir.glob('*.txt'))))
        
        return {
            'total_images': total_images, 'total_labels': total_labels,
            'splits': {'train': {'exists': True, 'images': total_images, 'labels': total_labels, 'issues': []}}
        }
    
    def validate_download_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """One-liner optimized parameter validation"""
        required_fields = ['workspace', 'project', 'version', 'api_key']
        issues = [f'Field {field} tidak boleh kosong' for field in required_fields if not params.get(field)]
        
        # One-liner API key validation
        api_key = params.get('api_key', '')
        api_key and len(api_key) < 10 and issues.append('API key terlalu pendek')
        
        # One-liner output directory validation
        output_dir = params.get('output_dir')
        output_dir and self._validate_output_directory(output_dir) and None
        
        return {'valid': not issues, 'issues': issues}
    
    def _validate_output_directory(self, output_dir: str) -> bool:
        """One-liner output directory validation"""
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            return True
        except Exception:
            return False
    
    def validate_metadata_response(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """One-liner optimized metadata validation"""
        issues = []
        
        # One-liner required fields check
        'export' not in metadata and issues.append('Metadata tidak mengandung export info')
        'export' in metadata and 'link' not in metadata['export'] and issues.append('Metadata tidak mengandung download link')
        'project' in metadata and 'classes' not in metadata['project'] and issues.append('Metadata tidak mengandung info kelas')
        'version' in metadata and 'images' not in metadata['version'] and issues.append('Metadata tidak mengandung info jumlah gambar')
        
        return {'valid': not issues, 'issues': issues}
    
    def batch_validate_files(self, file_paths: List[Path]) -> Dict[str, Any]:
        """Parallel batch file validation dengan optimized processing"""
        if not file_paths:
            return {'valid': True, 'total_files': 0, 'valid_files': 0, 'invalid_files': [], 'missing_files': []}
        
        # Parallel validation
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            validation_futures = {str(file_path): executor.submit(self._validate_single_file, file_path) for file_path in file_paths}
            
            # One-liner result aggregation
            results = {path: future.result() for path, future in validation_futures.items()}
            valid_files = sum(1 for result in results.values() if result['valid'])
            invalid_files = [path for path, result in results.items() if not result['valid'] and result['exists']]
            missing_files = [path for path, result in results.items() if not result['exists']]
        
        return {
            'valid': len(invalid_files) == 0 and len(missing_files) == 0,
            'total_files': len(file_paths), 'valid_files': valid_files,
            'invalid_files': invalid_files, 'missing_files': missing_files
        }
    
    def _validate_single_file(self, file_path: Path) -> Dict[str, Any]:
        """One-liner single file validation"""
        if not file_path.exists():
            return {'valid': False, 'exists': False, 'message': 'File tidak ditemukan'}
        
        # One-liner file type validation
        if file_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
            return self._validate_image_file_optimized(file_path)
        elif file_path.suffix.lower() == '.txt':
            return self._validate_label_file_optimized(file_path)
        elif file_path.suffix.lower() == '.zip':
            return self.validate_zip_file(file_path)
        else:
            return {'valid': True, 'exists': True, 'message': 'File type tidak divalidasi'}
    
    def _validate_image_file_optimized(self, file_path: Path) -> Dict[str, Any]:
        """One-liner optimized image file validation"""
        try:
            stat = file_path.stat()
            return {
                'valid': stat.st_size > 0, 'exists': True,
                'size_bytes': stat.st_size, 'message': 'Image valid' if stat.st_size > 0 else 'Image file kosong'
            }
        except Exception as e:
            return {'valid': False, 'exists': True, 'message': f'Error validasi image: {str(e)}'}
    
    def _validate_label_file_optimized(self, file_path: Path) -> Dict[str, Any]:
        """One-liner optimized YOLO label validation"""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # One-liner label format validation
            valid_lines = sum(1 for line in lines 
                            if len(parts := line.strip().split()) >= 5 
                            and self._is_valid_yolo_line(parts))
            
            return {
                'valid': valid_lines > 0, 'exists': True, 'total_lines': len(lines),
                'valid_lines': valid_lines, 'message': f'{valid_lines}/{len(lines)} baris valid'
            }
            
        except Exception as e:
            return {'valid': False, 'exists': True, 'message': f'Error validasi label: {str(e)}'}
    
    def _is_valid_yolo_line(self, parts: List[str]) -> bool:
        """One-liner YOLO line validation"""
        try:
            class_id, x, y, w, h = int(float(parts[0])), *map(float, parts[1:5])
            return 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1
        except (ValueError, IndexError):
            return False
    
    def _return_error(self, message: str) -> None:
        """One-liner error return"""
        raise ValueError(message)

class FileValidator:
    """Optimized file validator dengan one-liner methods."""
    
    def __init__(self, logger=None, max_workers: int = 4):
        self.logger, self.max_workers = logger or get_logger(), max_workers
    
    def validate_file_existence(self, file_paths: List[Path]) -> Dict[str, Any]:
        """One-liner parallel file existence validation"""
        existing_files = [str(fp) for fp in file_paths if fp.exists()]
        missing_files = [str(fp) for fp in file_paths if not fp.exists()]
        return {'valid': not missing_files, 'missing_files': missing_files, 'existing_files': existing_files}
    
    def validate_disk_space(self, required_mb: float, target_dir: Path) -> Dict[str, Any]:
        """One-liner disk space validation"""
        try:
            import shutil
            total, used, free = shutil.disk_usage(target_dir.parent if target_dir.parent.exists() else target_dir)
            free_mb = free / 1048576
            return {'valid': free_mb >= required_mb, 'free_mb': free_mb, 'required_mb': required_mb, 'sufficient': free_mb >= required_mb}
        except Exception as e:
            return {'valid': False, 'message': f'Error checking disk space: {str(e)}'}
    
    def validate_write_permissions(self, directory: Path) -> Dict[str, Any]:
        """One-liner write permission validation"""
        try:
            directory.mkdir(parents=True, exist_ok=True)
            test_file = directory / '.test_write'
            test_file.touch() and test_file.unlink()
            return {'valid': True, 'writable': True}
        except Exception as e:
            return {'valid': False, 'writable': False, 'message': f'Write permission error: {str(e)}'}
    
    def batch_validate_permissions(self, directories: List[Path]) -> Dict[str, Any]:
        """Parallel permission validation dengan optimized processing"""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            permission_futures = {str(dir_path): executor.submit(self.validate_write_permissions, dir_path) for dir_path in directories}
            results = {path: future.result() for path, future in permission_futures.items()}
        
        # One-liner result aggregation
        valid_dirs = [path for path, result in results.items() if result['valid']]
        invalid_dirs = [path for path, result in results.items() if not result['valid']]
        
        return {'valid': not invalid_dirs, 'valid_directories': valid_dirs, 'invalid_directories': invalid_dirs}

# One-liner factory functions
def create_dataset_validator(logger=None, max_workers: int = None) -> DatasetValidator:
    """Factory untuk optimized DatasetValidator"""
    import os
    optimal_workers = max_workers or min(4, (os.cpu_count() or 1) + 1)
    return DatasetValidator(logger, optimal_workers)

def create_file_validator(logger=None, max_workers: int = None) -> FileValidator:
    """Factory untuk optimized FileValidator"""
    import os
    optimal_workers = max_workers or min(4, (os.cpu_count() or 1) + 1)
    return FileValidator(logger, optimal_workers)

# One-liner utility functions
def validate_image_file_quick(file_path: Path) -> bool:
    """Quick one-liner image validation"""
    return file_path.exists() and file_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'} and file_path.stat().st_size > 0

def validate_label_file_quick(file_path: Path) -> bool:
    """Quick one-liner label validation"""
    try:
        return (file_path.exists() and file_path.suffix.lower() == '.txt' and 
                any(len(line.strip().split()) >= 5 for line in open(file_path, 'r').readlines()[:5]))  # Check first 5 lines
    except Exception:
        return False

def get_validation_summary_optimized(validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """One-liner optimized validation summary"""
    total_validations = len(validation_results)
    passed_validations = sum(1 for result in validation_results if result.get('valid', False))
    
    return {
        'overall_valid': passed_validations == total_validations,
        'total_checks': total_validations, 'passed_checks': passed_validations,
        'failed_checks': total_validations - passed_validations,
        'success_rate': (passed_validations / total_validations * 100) if total_validations > 0 else 0,
        'issues': [issue for result in validation_results for issue in result.get('issues', result.get('message', []) if isinstance(result.get('message'), list) else [result.get('message')] if result.get('message') else []) if not result.get('valid', True)]
    }

def validate_dataset_structure_quick(dataset_dir: Path) -> bool:
    """One-liner quick dataset structure validation"""
    return (dataset_dir.exists() and 
            (any((dataset_dir / split / 'images').exists() for split in ['train', 'valid', 'test']) or
             (dataset_dir / 'images').exists()))

# One-liner batch operations
batch_validate_images = lambda image_paths, validator=None: (validator or create_dataset_validator()).batch_validate_files(image_paths)
batch_validate_labels = lambda label_paths, validator=None: (validator or create_dataset_validator()).batch_validate_files(label_paths)
validate_complete_dataset = lambda dataset_dir, validator=None: (validator or create_dataset_validator()).validate_extracted_dataset(dataset_dir)