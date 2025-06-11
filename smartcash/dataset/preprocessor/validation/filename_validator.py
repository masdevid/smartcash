"""
File: smartcash/dataset/preprocessor/validation/filename_validator.py
Deskripsi: Simplified filename validator menggunakan FileNamingManager patterns
"""

from pathlib import Path
from typing import Dict, Any, List, Optional

from smartcash.common.logger import get_logger
from smartcash.common.utils.file_naming_manager import create_file_naming_manager

class FilenameValidator:
    """📝 Simplified filename validator menggunakan FileNamingManager patterns"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = get_logger(__name__)
        self.naming_manager = create_file_naming_manager(config)
    
    def validate_filename(self, filename: str) -> Dict[str, Any]:
        """✅ Validate filename menggunakan FileNamingManager patterns"""
        validation = self.naming_manager.validate_filename_format(filename)
        
        if validation['valid']:
            return {
                'is_valid': True,
                'pattern': validation['format'],
                'components': validation['parsed'],
                'needs_rename': False,
                'naming_info': validation
            }
        
        return {
            'is_valid': False,
            'pattern': 'unknown',
            'components': None,
            'needs_rename': True,
            'suggested_name': self._generate_research_filename(filename)
        }
    
    def batch_validate(self, filenames: List[str]) -> Dict[str, Dict[str, Any]]:
        """📦 Batch validate menggunakan naming manager"""
        return {filename: self.validate_filename(filename) for filename in filenames}
    
    def rename_invalid_files(self, file_paths: List[Path], 
                           progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """🔄 Rename files menggunakan FileNamingManager"""
        stats = {'renamed': 0, 'skipped': 0, 'errors': 0}
        rename_map = {}
        
        for i, file_path in enumerate(file_paths):
            try:
                validation = self.validate_filename(file_path.name)
                
                if validation['needs_rename']:
                    primary_class = self._extract_class_from_filename(file_path)
                    file_info = self.naming_manager.generate_file_info(
                        file_path.name, primary_class, 'raw'
                    )
                    new_name = file_info.get_filename()
                    new_path = file_path.parent / new_name
                    
                    # Handle duplicates
                    counter = 1
                    original_new_path = new_path
                    while new_path.exists():
                        stem = original_new_path.stem
                        ext = original_new_path.suffix
                        new_path = original_new_path.parent / f"{stem}_{counter}{ext}"
                        counter += 1
                    
                    file_path.rename(new_path)
                    rename_map[str(file_path)] = str(new_path)
                    stats['renamed'] += 1
                    
                    self.logger.info(f"🔄 Renamed: {file_path.name} → {new_path.name}")
                else:
                    stats['skipped'] += 1
                
                if progress_callback and i % max(1, len(file_paths) // 20) == 0:
                    progress_callback('current', i + 1, len(file_paths), f"Processing {file_path.name}")
                    
            except Exception as e:
                self.logger.error(f"❌ Rename error {file_path.name}: {str(e)}")
                stats['errors'] += 1
        
        return {
            'stats': stats,
            'rename_map': rename_map,
            'success': stats['errors'] == 0,
            'naming_stats': self.naming_manager.get_nominal_statistics()
        }
    
    def _generate_research_filename(self, original_filename: str) -> str:
        """🔧 Generate research filename menggunakan FileNamingManager"""
        primary_class = self._extract_class_from_filename(Path(original_filename))
        file_info = self.naming_manager.generate_file_info(
            original_filename, primary_class, 'raw'
        )
        return file_info.get_filename()
    
    def _extract_class_from_filename(self, file_path: Path) -> Optional[str]:
        """💰 Extract class dari corresponding label"""
        label_path = file_path.with_suffix('.txt')
        
        if not label_path.exists() and file_path.parent.name == 'images':
            labels_dir = file_path.parent.parent / 'labels'
            label_path = labels_dir / f"{file_path.stem}.txt"
        
        if label_path.exists():
            return self.naming_manager.extract_primary_class_from_label(label_path)
        
        return self._detect_class_from_filename_pattern(file_path.name)
    
    def _detect_class_from_filename_pattern(self, filename: str) -> Optional[str]:
        """🔍 Detect class dari filename pattern"""
        filename_lower = filename.lower()
        patterns = [
            (r'100000|100k|rp100000', '6'), (r'50000|50k|rp50000', '5'),
            (r'20000|20k|rp20000', '4'), (r'10000|10k|rp10000', '3'),
            (r'5000|5k|rp5000', '2'), (r'2000|2k|rp2000', '1'), (r'1000|1k|rp1000', '0')
        ]
        
        import re
        for pattern, class_id in patterns:
            if re.search(pattern, filename_lower):
                return class_id
        return None
    
    def get_rename_preview(self, file_paths: List[Path]) -> List[Dict[str, str]]:
        """👀 Preview rename operations"""
        previews = []
        
        for file_path in file_paths:
            validation = self.validate_filename(file_path.name)
            
            if validation['needs_rename']:
                primary_class = self._extract_class_from_filename(file_path)
                file_info = self.naming_manager.generate_file_info(
                    file_path.name, primary_class, 'raw'
                )
                suggested_name = file_info.get_filename()
                
                previews.append({
                    'original': file_path.name,
                    'suggested': suggested_name,
                    'path': str(file_path),
                    'reason': 'Pattern tidak sesuai research format',
                    'detected_class': primary_class,
                    'nominal': file_info.nominal,
                    'description': self.naming_manager.NOMINAL_TO_DESCRIPTION.get(file_info.nominal, 'Unknown')
                })
        
        return previews
    
    def validate_directory_files(self, directory: Path, auto_rename: bool = False) -> Dict[str, Any]:
        """📁 Validate directory dengan FileNamingManager"""
        if not directory.exists():
            return {'success': False, 'message': f"❌ Directory tidak ditemukan: {directory}"}
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in directory.iterdir() 
                      if f.is_file() and f.suffix.lower() in image_extensions]
        
        if not image_files:
            return {
                'success': True,
                'message': f"✅ Tidak ada image files di {directory}",
                'stats': {'total': 0, 'valid': 0, 'invalid': 0, 'renamed': 0}
            }
        
        validations = {f.name: self.validate_filename(f.name) for f in image_files}
        valid_count = sum(1 for v in validations.values() if v['is_valid'])
        invalid_count = len(image_files) - valid_count
        
        stats = {'total': len(image_files), 'valid': valid_count, 'invalid': invalid_count, 'renamed': 0}
        
        if auto_rename and invalid_count > 0:
            invalid_files = [f for f in image_files if not self.validate_filename(f.name)['is_valid']]
            rename_result = self.rename_invalid_files(invalid_files)
            stats['renamed'] = rename_result['stats']['renamed']
            stats['naming_stats'] = rename_result['naming_stats']
        
        message = f"✅ Validation complete: {valid_count}/{len(image_files)} valid"
        if auto_rename and stats['renamed'] > 0:
            message += f", {stats['renamed']} files renamed dengan research format"
        elif invalid_count > 0:
            message += f", {invalid_count} files perlu rename ke research format"
        
        return {
            'success': True,
            'message': message,
            'stats': stats,
            'validations': validations if not auto_rename else None,
            'naming_manager_stats': self.naming_manager.get_nominal_statistics()
        }
    
    def get_filename_patterns_info(self) -> Dict[str, Any]:
        """📋 Get supported patterns info"""
        return {
            'supported_patterns': {
                file_type: f"{prefix}{{nominal}}_{{uuid}}.ext"
                for file_type, prefix in self.naming_manager.PATTERN_PREFIXES.items()
            },
            'nominal_mapping': self.naming_manager.NOMINAL_TO_DESCRIPTION,
            'class_to_nominal': self.naming_manager.CLASS_TO_NOMINAL,
            'supported_types': self.naming_manager.get_supported_types()
        }