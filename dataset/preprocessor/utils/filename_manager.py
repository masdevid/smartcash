"""
File: smartcash/dataset/preprocessor/utils/filename_manager.py
Deskripsi: Updated filename manager dengan pattern baru sesuai augmentor
"""
import re
import uuid
from pathlib import Path
from typing import Dict, Optional, Tuple

class FilenameManager:
    """📝 Updated manager untuk filename parsing dan generation dengan pattern baru"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._uuid_cache = {}
        
        # Updated patterns sesuai augmentor
        # Pattern untuk raw files: rp_001000_uuid_increment.ext
        self.raw_pattern = re.compile(r'rp_(\d{6})_([a-f0-9-]{36})_(\d+)\.(\w+)')
        
        # Pattern untuk preprocessed: pre_rp_001000_uuid_increment_variance.ext  
        self.prep_pattern = re.compile(r'pre_rp_(\d{6})_([a-f0-9-]{36})_(\d+)_(\d+)\.(\w+)')
        
        # Pattern untuk augmented: aug_rp_001000_uuid_increment_variance.ext
        self.aug_pattern = re.compile(r'aug_rp_(\d{6})_([a-f0-9-]{36})_(\d+)_(\d+)\.(\w+)')
        
        # Legacy patterns untuk backward compatibility
        self.legacy_patterns = [
            re.compile(r'(\w+)_(\d+)_(\w+)\.(\w+)'),  # old format
            re.compile(r'(\w+)\.(\w+)')  # simple format
        ]
    
    def parse_filename(self, filename: str) -> Optional[Dict[str, any]]:
        """📋 Parse filename dan extract metadata dengan pattern baru"""
        # Try raw pattern first
        match = self.raw_pattern.match(filename)
        if match:
            return {
                'type': 'raw',
                'nominal': match.group(1),
                'uuid': match.group(2), 
                'increment': int(match.group(3)),
                'extension': match.group(4),
                'pattern': 'raw'
            }
        
        # Try preprocessed pattern
        match = self.prep_pattern.match(filename)
        if match:
            return {
                'type': 'preprocessed',
                'nominal': match.group(1),
                'uuid': match.group(2),
                'increment': int(match.group(3)),
                'variance': int(match.group(4)),
                'extension': match.group(5),
                'pattern': 'preprocessed'
            }
        
        # Try augmented pattern
        match = self.aug_pattern.match(filename)
        if match:
            return {
                'type': 'augmented',
                'nominal': match.group(1),
                'uuid': match.group(2),
                'increment': int(match.group(3)),
                'variance': int(match.group(4)),
                'extension': match.group(5),
                'pattern': 'augmented'
            }
        
        # Try legacy patterns untuk backward compatibility
        for pattern in self.legacy_patterns:
            match = pattern.match(filename)
            if match:
                return {
                    'type': 'legacy',
                    'stem': Path(filename).stem,
                    'extension': Path(filename).suffix[1:],
                    'pattern': 'legacy'
                }
        
        return None
    
    def generate_preprocessed_filename(self, source_filename: str, nominal: str = None, 
                                     increment: int = 1, variance: int = 1) -> str:
        """🔧 Generate preprocessed filename dari source dengan pattern baru"""
        parsed = self.parse_filename(source_filename)
        
        if parsed and parsed['pattern'] in ['raw', 'preprocessed']:
            # Use existing metadata
            file_nominal = parsed.get('nominal', '000000')
            file_uuid = parsed.get('uuid', self._generate_uuid())
            file_increment = parsed.get('increment', increment)
            
            # For preprocessed files, increment variance
            if parsed['pattern'] == 'preprocessed':
                file_variance = parsed.get('variance', 0) + 1
            else:
                file_variance = variance
                
        else:
            # Generate new metadata untuk legacy atau unknown files
            file_nominal = nominal or self._extract_nominal_from_filename(source_filename)
            file_uuid = self._get_or_generate_uuid(source_filename)
            file_increment = increment
            file_variance = variance
        
        return f"pre_rp_{file_nominal}_{file_uuid}_{file_increment:03d}_{file_variance:02d}"
    
    def create_matching_label_filename(self, image_filename: str) -> str:
        """📋 Create matching label filename untuk image file"""
        stem = Path(image_filename).stem
        return f"{stem}.txt"
    
    def _extract_nominal_from_filename(self, filename: str) -> str:
        """💰 Extract nominal dari filename atau class mapping"""
        # Try to extract from filename patterns
        nominal_patterns = [
            (r'1000|rp1000', '001000'),
            (r'2000|rp2000', '002000'), 
            (r'5000|rp5000', '005000'),
            (r'10000|rp10000', '010000'),
            (r'20000|rp20000', '020000'),
            (r'50000|rp50000', '050000'),
            (r'100000|rp100000', '100000')
        ]
        
        filename_lower = filename.lower()
        for pattern, nominal in nominal_patterns:
            if re.search(pattern, filename_lower):
                return nominal
        
        # Default fallback
        return '000000'
    
    def _get_or_generate_uuid(self, filename: str) -> str:
        """🆔 Get cached UUID atau generate baru untuk filename"""
        if filename not in self._uuid_cache:
            self._uuid_cache[filename] = str(uuid.uuid4())
        return self._uuid_cache[filename]
    
    def _generate_uuid(self) -> str:
        """🆔 Generate UUID baru"""
        return str(uuid.uuid4())
    
    def get_file_metadata(self, filename: str) -> Dict[str, any]:
        """📊 Get comprehensive metadata dari filename"""
        parsed = self.parse_filename(filename)
        
        if not parsed:
            return {
                'valid': False,
                'filename': filename,
                'error': 'Unable to parse filename'
            }
        
        metadata = {
            'valid': True,
            'filename': filename,
            'parsed': parsed,
            'denomination': self._get_denomination_info(parsed.get('nominal', '000000')),
            'is_research_format': parsed['pattern'] in ['raw', 'preprocessed', 'augmented']
        }
        
        return metadata
    
    def _get_denomination_info(self, nominal: str) -> Dict[str, any]:
        """💰 Get denomination information dari nominal"""
        denomination_map = {
            '001000': {'value': 1000, 'display': 'Rp1.000'},
            '002000': {'value': 2000, 'display': 'Rp2.000'},
            '005000': {'value': 5000, 'display': 'Rp5.000'},
            '010000': {'value': 10000, 'display': 'Rp10.000'},
            '020000': {'value': 20000, 'display': 'Rp20.000'},
            '050000': {'value': 50000, 'display': 'Rp50.000'},
            '100000': {'value': 100000, 'display': 'Rp100.000'},
            '000000': {'value': 0, 'display': 'Unknown'}
        }
        
        return denomination_map.get(nominal, {'value': 0, 'display': 'Invalid'})
    
    def validate_filename_consistency(self, image_file: str, label_file: str) -> Dict[str, any]:
        """🔍 Validate consistency antara image dan label filename"""
        img_parsed = self.parse_filename(image_file)
        lbl_parsed = self.parse_filename(label_file)
        
        if not img_parsed or not lbl_parsed:
            return {
                'valid': False,
                'error': 'Cannot parse one or both filenames',
                'image_parsed': bool(img_parsed),
                'label_parsed': bool(lbl_parsed)
            }
        
        # Check consistency
        consistency_checks = {
            'nominal_match': img_parsed.get('nominal') == lbl_parsed.get('nominal'),
            'uuid_match': img_parsed.get('uuid') == lbl_parsed.get('uuid'),
            'increment_match': img_parsed.get('increment') == lbl_parsed.get('increment'),
            'type_match': img_parsed.get('type') == lbl_parsed.get('type')
        }
        
        all_consistent = all(consistency_checks.values())
        
        return {
            'valid': all_consistent,
            'consistency_checks': consistency_checks,
            'image_metadata': img_parsed,
            'label_metadata': lbl_parsed
        }

# Factory function
def create_filename_manager(config: Dict = None) -> FilenameManager:
    """🏭 Factory untuk create filename manager"""
    return FilenameManager(config)