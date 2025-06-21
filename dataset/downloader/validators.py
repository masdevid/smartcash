"""
File: smartcash/dataset/downloader/validators.py
Deskripsi: Simplified validators menggunakan base components
"""

import zipfile
from pathlib import Path
from typing import Dict, Any, List
from smartcash.dataset.downloader.base import BaseDownloaderComponent, ValidationHelper


class DatasetValidator(BaseDownloaderComponent):
    """Simplified dataset validator"""
    
    def __init__(self, logger=None):
        super().__init__(logger)
    
    def validate_zip_file(self, zip_path: Path) -> Dict[str, Any]:
        """Validate ZIP file"""
        try:
            if not zip_path.exists():
                return self._create_error_result('ZIP file tidak ditemukan')
            
            if not zipfile.is_zipfile(zip_path):
                return self._create_error_result('File bukan ZIP yang valid')
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = zip_ref.infolist()
                
                if not file_list:
                    return self._create_error_result('ZIP file kosong')
                
                has_images = any('images' in f.filename.lower() or 
                               f.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')) 
                               for f in file_list)
                has_labels = any('labels' in f.filename.lower() or 
                               f.filename.lower().endswith('.txt') 
                               for f in file_list)
                
                return self._create_success_result(
                    total_files=len(file_list),
                    has_images=has_images,
                    has_labels=has_labels,
                    size_mb=zip_path.stat().st_size / 1048576
                )
            
        except Exception as e:
            return self._create_error_result(f'Error validasi ZIP: {str(e)}')
    
    def validate_extracted_dataset(self, dataset_dir: Path) -> Dict[str, Any]:
        """Validate extracted dataset"""
        try:
            if not dataset_dir.exists():
                return self._create_error_result('Dataset directory tidak ditemukan')
            
            # Use shared validation helper
            structure_result = ValidationHelper.validate_dataset_structure(dataset_dir)
            
            if not structure_result['valid']:
                return self._create_error_result(structure_result['message'])
            
            # Count files per split
            validation_result = {
                'valid': True,
                'splits': {},
                'total_images': 0,
                'total_labels': 0,
                'issues': []
            }
            
            for split in structure_result.get('splits', []):
                split_result = self._validate_split(dataset_dir / split)
                validation_result['splits'][split] = split_result
                validation_result['total_images'] += split_result.get('images', 0)
                validation_result['total_labels'] += split_result.get('labels', 0)
                
                # Check for issues
                if split_result.get('images', 0) == 0:
                    validation_result['issues'].append(f"Split {split}: Tidak ada gambar")
                if split_result.get('labels', 0) == 0:
                    validation_result['issues'].append(f"Split {split}: Tidak ada label")
            
            if validation_result['total_images'] == 0:
                validation_result['valid'] = False
                validation_result['issues'].append('Tidak ada gambar ditemukan')
            
            return validation_result
            
        except Exception as e:
            return self._create_error_result(f'Error validasi dataset: {str(e)}')
    
    def _validate_split(self, split_dir: Path) -> Dict[str, Any]:
        """Validate single split"""
        if not split_dir.exists():
            return {'exists': False, 'images': 0, 'labels': 0}
        
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_count = len([f for f in images_dir.glob('*.*') 
                          if f.suffix.lower() in image_extensions]) if images_dir.exists() else 0
        label_count = len(list(labels_dir.glob('*.txt'))) if labels_dir.exists() else 0
        
        return {
            'exists': True,
            'images': image_count,
            'labels': label_count
        }
    
    def validate_download_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate download parameters"""
        required_fields = ['workspace', 'project', 'version', 'api_key']
        return ValidationHelper.validate_config(params, required_fields)


def create_dataset_validator(logger=None, max_workers: int = None) -> DatasetValidator:
    """Factory untuk DatasetValidator"""
    return DatasetValidator(logger)