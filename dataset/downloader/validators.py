"""
File: smartcash/dataset/downloader/validators.py
Deskripsi: Complete validators untuk data validation dalam download process
"""

import zipfile
from pathlib import Path
from typing import Dict, Any, List, Optional
from smartcash.common.logger import get_logger

class DatasetValidator:
    """Validator untuk dataset structure dan integrity."""
    
    def __init__(self, logger=None):
        self.logger = logger or get_logger()
    
    def validate_zip_file(self, zip_path: Path) -> Dict[str, Any]:
        """Validate ZIP file integrity dan structure."""
        try:
            if not zip_path.exists():
                return {'valid': False, 'message': 'ZIP file tidak ditemukan'}
            
            if not zipfile.is_zipfile(zip_path):
                return {'valid': False, 'message': 'File bukan ZIP yang valid'}
            
            # Check ZIP content
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = zip_ref.infolist()
                
                if not file_list:
                    return {'valid': False, 'message': 'ZIP file kosong'}
                
                # Basic structure validation
                has_images = any('images' in f.filename.lower() or 
                               f.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')) 
                               for f in file_list)
                
                has_labels = any('labels' in f.filename.lower() or 
                               f.filename.lower().endswith('.txt') 
                               for f in file_list)
                
                return {
                    'valid': True,
                    'total_files': len(file_list),
                    'has_images': has_images,
                    'has_labels': has_labels,
                    'size_mb': zip_path.stat().st_size / (1024 * 1024),
                    'message': f'ZIP valid: {len(file_list)} files'
                }
            
        except Exception as e:
            return {'valid': False, 'message': f'Error validasi ZIP: {str(e)}'}
    
    def validate_extracted_dataset(self, dataset_dir: Path) -> Dict[str, Any]:
        """Validate extracted dataset structure."""
        try:
            if not dataset_dir.exists():
                return {'valid': False, 'message': 'Dataset directory tidak ditemukan'}
            
            validation_result = {
                'valid': True,
                'splits': {},
                'total_images': 0,
                'total_labels': 0,
                'issues': [],
                'structure_type': 'unknown'
            }
            
            # Detect structure type
            structure = self._detect_dataset_structure(dataset_dir)
            validation_result['structure_type'] = structure['type']
            
            if structure['type'] == 'split_based':
                # Validate split-based structure
                for split in structure['splits']:
                    split_validation = self._validate_split(dataset_dir / split, split)
                    validation_result['splits'][split] = split_validation
                    
                    validation_result['total_images'] += split_validation['images']
                    validation_result['total_labels'] += split_validation['labels']
                    
                    if split_validation['issues']:
                        validation_result['issues'].extend(split_validation['issues'])
            
            elif structure['type'] == 'flat':
                # Validate flat structure
                flat_validation = self._validate_flat_structure(dataset_dir)
                validation_result.update(flat_validation)
            
            else:
                validation_result['valid'] = False
                validation_result['issues'].append('Struktur dataset tidak dikenali')
            
            # Overall validation checks
            if validation_result['total_images'] == 0:
                validation_result['valid'] = False
                validation_result['issues'].append('Tidak ada gambar ditemukan')
            
            return validation_result
            
        except Exception as e:
            return {'valid': False, 'message': f'Error validasi dataset: {str(e)}'}
    
    def _detect_dataset_structure(self, dataset_dir: Path) -> Dict[str, Any]:
        """Detect tipe struktur dataset."""
        # Check for split-based structure
        splits = []
        for split_name in ['train', 'valid', 'test', 'val']:
            split_dir = dataset_dir / split_name
            if split_dir.exists() and (split_dir / 'images').exists():
                splits.append(split_name)
        
        if splits:
            return {'type': 'split_based', 'splits': splits}
        
        # Check for flat structure
        if (dataset_dir / 'images').exists() or any(dataset_dir.glob('*.jpg')):
            return {'type': 'flat', 'splits': []}
        
        return {'type': 'unknown', 'splits': []}
    
    def _validate_split(self, split_dir: Path, split_name: str) -> Dict[str, Any]:
        """Validate single split directory."""
        validation = {
            'exists': split_dir.exists(),
            'images': 0,
            'labels': 0,
            'issues': []
        }
        
        if not validation['exists']:
            validation['issues'].append(f'Split {split_name} tidak ditemukan')
            return validation
        
        # Check images directory
        images_dir = split_dir / 'images'
        if images_dir.exists():
            image_files = list(images_dir.glob('*.*'))
            validation['images'] = len([f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
        else:
            validation['issues'].append(f'Folder images tidak ditemukan dalam {split_name}')
        
        # Check labels directory
        labels_dir = split_dir / 'labels'
        if labels_dir.exists():
            label_files = list(labels_dir.glob('*.txt'))
            validation['labels'] = len(label_files)
        else:
            validation['issues'].append(f'Folder labels tidak ditemukan dalam {split_name}')
        
        # Check image-label pairing
        if validation['images'] > 0 and validation['labels'] > 0:
            mismatch = abs(validation['images'] - validation['labels'])
            if mismatch > 0:
                validation['issues'].append(f'Mismatch dalam {split_name}: {validation["images"]} gambar vs {validation["labels"]} label')
        
        return validation
    
    def _validate_flat_structure(self, dataset_dir: Path) -> Dict[str, Any]:
        """Validate flat dataset structure."""
        images_dir = dataset_dir / 'images'
        labels_dir = dataset_dir / 'labels'
        
        validation = {
            'valid': True,
            'total_images': 0,
            'total_labels': 0,
            'issues': []
        }
        
        # Count images
        if images_dir.exists():
            image_files = list(images_dir.glob('*.*'))
            validation['total_images'] = len([f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
        else:
            # Check for images in root directory
            image_files = list(dataset_dir.glob('*.jpg')) + list(dataset_dir.glob('*.png'))
            validation['total_images'] = len(image_files)
        
        # Count labels
        if labels_dir.exists():
            label_files = list(labels_dir.glob('*.txt'))
            validation['total_labels'] = len(label_files)
        else:
            # Check for labels in root directory
            label_files = list(dataset_dir.glob('*.txt'))
            validation['total_labels'] = len(label_files)
        
        # Add flat structure to splits for consistency
        validation['splits'] = {
            'train': {
                'exists': True,
                'images': validation['total_images'],
                'labels': validation['total_labels'],
                'issues': []
            }
        }
        
        return validation
    
    def validate_download_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate download parameters."""
        validation = {'valid': True, 'issues': []}
        
        # Required fields
        required_fields = ['workspace', 'project', 'version', 'api_key']
        for field in required_fields:
            if not params.get(field):
                validation['issues'].append(f'Field {field} tidak boleh kosong')
        
        # API key validation
        api_key = params.get('api_key', '')
        if api_key and len(api_key) < 10:
            validation['issues'].append('API key terlalu pendek')
        
        # Output directory validation
        output_dir = params.get('output_dir')
        if output_dir:
            try:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
            except Exception:
                validation['issues'].append('Output directory tidak dapat dibuat')
        
        validation['valid'] = len(validation['issues']) == 0
        return validation
    
    def validate_metadata_response(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate metadata response dari Roboflow API."""
        validation = {'valid': True, 'issues': []}
        
        # Check required fields
        if 'export' not in metadata:
            validation['issues'].append('Metadata tidak mengandung export info')
        elif 'link' not in metadata['export']:
            validation['issues'].append('Metadata tidak mengandung download link')
        
        # Check project info
        if 'project' in metadata:
            project_info = metadata['project']
            if 'classes' not in project_info:
                validation['issues'].append('Metadata tidak mengandung info kelas')
        
        # Check version info
        if 'version' in metadata:
            version_info = metadata['version']
            if 'images' not in version_info:
                validation['issues'].append('Metadata tidak mengandung info jumlah gambar')
        
        validation['valid'] = len(validation['issues']) == 0
        return validation

class FileValidator:
    """Validator untuk file operations."""
    
    def __init__(self, logger=None):
        self.logger = logger or get_logger()
    
    def validate_file_existence(self, file_paths: List[Path]) -> Dict[str, Any]:
        """Validate keberadaan multiple files."""
        result = {'valid': True, 'missing_files': [], 'existing_files': []}
        
        for file_path in file_paths:
            if file_path.exists():
                result['existing_files'].append(str(file_path))
            else:
                result['missing_files'].append(str(file_path))
        
        result['valid'] = len(result['missing_files']) == 0
        return result
    
    def validate_disk_space(self, required_mb: float, target_dir: Path) -> Dict[str, Any]:
        """Validate disk space availability."""
        try:
            import shutil
            total, used, free = shutil.disk_usage(target_dir.parent if target_dir.parent.exists() else target_dir)
            free_mb = free / (1024 * 1024)
            
            return {
                'valid': free_mb >= required_mb,
                'free_mb': free_mb,
                'required_mb': required_mb,
                'sufficient': free_mb >= required_mb
            }
            
        except Exception as e:
            return {'valid': False, 'message': f'Error checking disk space: {str(e)}'}
    
    def validate_write_permissions(self, directory: Path) -> Dict[str, Any]:
        """Validate write permissions untuk directory."""
        try:
            directory.mkdir(parents=True, exist_ok=True)
            
            # Test write permission
            test_file = directory / '.test_write'
            test_file.touch()
            test_file.unlink()
            
            return {'valid': True, 'writable': True}
            
        except Exception as e:
            return {'valid': False, 'writable': False, 'message': f'Write permission error: {str(e)}'}

# Factory functions
def create_dataset_validator(logger=None) -> DatasetValidator:
    """Factory untuk create DatasetValidator."""
    return DatasetValidator(logger)

def create_file_validator(logger=None) -> FileValidator:
    """Factory untuk create FileValidator."""
    return FileValidator(logger)

# Utility functions
def validate_image_file(file_path: Path) -> bool:
    """Quick validation untuk image file."""
    return file_path.exists() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp'] and file_path.stat().st_size > 0

def validate_label_file(file_path: Path) -> Dict[str, Any]:
    """Validate YOLO label file format."""
    try:
        if not file_path.exists() or file_path.suffix.lower() != '.txt':
            return {'valid': False, 'message': 'Label file tidak ditemukan atau bukan .txt'}
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        valid_lines = 0
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    class_id = int(float(parts[0]))
                    x, y, w, h = map(float, parts[1:5])
                    
                    # Validate YOLO format bounds
                    if 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1:
                        valid_lines += 1
                    
                except (ValueError, IndexError):
                    continue
        
        return {
            'valid': valid_lines > 0,
            'total_lines': len(lines),
            'valid_lines': valid_lines,
            'message': f'{valid_lines}/{len(lines)} baris valid'
        }
        
    except Exception as e:
        return {'valid': False, 'message': f'Error validasi label: {str(e)}'}

def get_validation_summary(validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate summary dari multiple validation results."""
    total_validations = len(validation_results)
    passed_validations = sum(1 for result in validation_results if result.get('valid', False))
    
    summary = {
        'overall_valid': passed_validations == total_validations,
        'total_checks': total_validations,
        'passed_checks': passed_validations,
        'failed_checks': total_validations - passed_validations,
        'success_rate': (passed_validations / total_validations * 100) if total_validations > 0 else 0
    }
    
    # Collect all issues
    all_issues = []
    for result in validation_results:
        if 'issues' in result:
            all_issues.extend(result['issues'])
        elif not result.get('valid', True) and 'message' in result:
            all_issues.append(result['message'])
    
    summary['issues'] = all_issues
    return summary