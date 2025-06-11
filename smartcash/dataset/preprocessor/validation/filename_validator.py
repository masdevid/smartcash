"""
File: smartcash/dataset/preprocessor/validation/filename_validator.py
Deskripsi: Filename pattern validation dan auto-rename untuk research format
"""

import re
import uuid
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from smartcash.common.logger import get_logger

class FilenameValidator:
    """📝 Filename pattern validator dengan auto-rename capability"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = get_logger(__name__)
        
        # Research filename pattern: rp_001000_uuid.jpg
        self.raw_pattern = re.compile(r'rp_(\d{6})_([a-f0-9-]{36})\.(\w+)')
        self.preprocessed_pattern = re.compile(r'pre_rp_(\d{6})_([a-f0-9-]{36})_(\d+)\.(\w+)')
        self.augmented_pattern = re.compile(r'aug_rp_(\d{6})_([a-f0-9-]{36})_(\d+)\.(\w+)')
        
        # Denomination mapping untuk auto-detection
        self.denomination_patterns = [
            (r'100000|rp100000|100k', '100000'),
            (r'50000|rp50000|50k', '050000'),
            (r'20000|rp20000|20k', '020000'),
            (r'10000|rp10000|10k', '010000'),
            (r'5000|rp5000|5k', '005000'),
            (r'2000|rp2000|2k', '002000'),
            (r'1000|rp1000|1k', '001000')
        ]
    
    def validate_filename(self, filename: str) -> Dict[str, Any]:
        """✅ Validate filename pattern"""
        # Check research patterns
        for pattern_name, pattern in [
            ('raw', self.raw_pattern),
            ('preprocessed', self.preprocessed_pattern),
            ('augmented', self.augmented_pattern)
        ]:
            match = pattern.match(filename)
            if match:
                return {
                    'is_valid': True,
                    'pattern': pattern_name,
                    'components': self._extract_components(match, pattern_name),
                    'needs_rename': False
                }
        
        # Invalid pattern - needs rename
        return {
            'is_valid': False,
            'pattern': 'unknown',
            'components': None,
            'needs_rename': True,
            'suggested_name': self._generate_research_filename(filename)
        }
    
    def batch_validate(self, filenames: List[str]) -> Dict[str, Dict[str, Any]]:
        """📦 Batch validate filenames"""
        return {filename: self.validate_filename(filename) for filename in filenames}
    
    def rename_invalid_files(self, file_paths: List[Path], 
                           progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """🔄 Rename files dengan invalid patterns"""
        stats = {'renamed': 0, 'skipped': 0, 'errors': 0}
        rename_map = {}
        
        for i, file_path in enumerate(file_paths):
            try:
                validation = self.validate_filename(file_path.name)
                
                if validation['needs_rename']:
                    new_name = validation['suggested_name']
                    new_path = file_path.parent / new_name
                    
                    # Rename file
                    file_path.rename(new_path)
                    rename_map[str(file_path)] = str(new_path)
                    stats['renamed'] += 1
                    
                    self.logger.info(f"🔄 Renamed: {file_path.name} → {new_name}")
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
            'success': stats['errors'] == 0
        }
    
    def _extract_components(self, match, pattern_name: str) -> Dict[str, Any]:
        """📋 Extract filename components dari regex match"""
        if pattern_name == 'raw':
            nominal, uuid_str, sequence, extension = match.groups()
            return {
                'nominal': nominal,
                'uuid': uuid_str,
                # 'sequence': int(sequence),
                'extension': extension
            }
        elif pattern_name in ['preprocessed', 'augmented']:
            nominal, uuid_str, sequence, variance, extension = match.groups()
            return {
                'nominal': nominal,
                'uuid': uuid_str,
                # 'sequence': int(sequence),
                'variance': int(variance),
                'extension': extension
            }
        return {}
    
    def _generate_research_filename(self, original_filename: str) -> str:
        """🔧 Generate research format filename dari original"""
        path = Path(original_filename)
        stem = path.stem.lower()
        extension = path.suffix.lower().lstrip('.')
        
        # Detect denomination
        nominal = self._detect_nominal(stem)
        
        # Generate UUID
        uuid_str = str(uuid.uuid4())
        
        # Default sequence
        sequence = 1
        
        return f"rp_{nominal}_{uuid_str}.{extension}"
    
    def _detect_nominal(self, filename: str) -> str:
        """💰 Detect denomination dari filename"""
        for pattern, nominal in self.denomination_patterns:
            if re.search(pattern, filename):
                return nominal
        return '000000'  # Unknown denomination
    
    def get_rename_preview(self, file_paths: List[Path]) -> List[Dict[str, str]]:
        """👀 Preview rename operations tanpa execute"""
        previews = []
        
        for file_path in file_paths:
            validation = self.validate_filename(file_path.name)
            
            if validation['needs_rename']:
                previews.append({
                    'original': file_path.name,
                    'suggested': validation['suggested_name'],
                    'path': str(file_path),
                    'reason': 'Invalid pattern'
                })
        
        return previews