"""
File: smartcash/dataset/preprocessor/utils/metadata_manager.py
Deskripsi: Konsolidasi metadata dan filename management dengan pattern recognition
"""

import re
import uuid
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime

from smartcash.common.logger import get_logger

@dataclass
class FileMetadata:
    """📊 Unified file metadata structure"""
    filename: str
    file_type: str  # 'raw', 'preprocessed', 'augmented', 'legacy'
    nominal: str
    uuid_str: str
    sequence: int
    variance: Optional[int] = None
    extension: str = ''
    is_valid: bool = True
    pattern_matched: str = ''
    denomination_info: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class MetadataManager:
    """📝 Konsolidasi metadata dan filename management dengan enhanced pattern support"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = get_logger(__name__)
        self._uuid_cache = {}
        self._metadata_cache = {}
        
        # Enhanced patterns dengan consistency
        self.patterns = {
            'raw': re.compile(r'rp_(\d{6})_([a-f0-9-]{36})_(\d+)\.(\w+)'),
            'preprocessed': re.compile(r'pre_rp_(\d{6})_([a-f0-9-]{36})_(\d+)_(\d+)\.(\w+)'),
            'augmented': re.compile(r'aug_rp_(\d{6})_([a-f0-9-]{36})_(\d+)_(\d+)\.(\w+)'),
            'legacy': re.compile(r'(.+)\.(\w+)')
        }
        
        # Denomination mapping
        self.denomination_map = {
            '001000': {'value': 1000, 'display': 'Rp1.000', 'class_id': 0},
            '002000': {'value': 2000, 'display': 'Rp2.000', 'class_id': 1},
            '005000': {'value': 5000, 'display': 'Rp5.000', 'class_id': 2},
            '010000': {'value': 10000, 'display': 'Rp10.000', 'class_id': 3},
            '020000': {'value': 20000, 'display': 'Rp20.000', 'class_id': 4},
            '050000': {'value': 50000, 'display': 'Rp50.000', 'class_id': 5},
            '100000': {'value': 100000, 'display': 'Rp100.000', 'class_id': 6},
            '000000': {'value': 0, 'display': 'Unknown', 'class_id': -1}
        }
    
    # === FILENAME PARSING ===
    
    def parse_filename(self, filename: str) -> Optional[FileMetadata]:
        """📋 Enhanced filename parsing dengan comprehensive pattern matching"""
        try:
            # Try research patterns first
            for pattern_name, pattern in self.patterns.items():
                if pattern_name == 'legacy':
                    continue
                    
                match = pattern.match(filename)
                if match:
                    return self._create_metadata_from_match(filename, pattern_name, match)
            
            # Fallback ke legacy pattern
            return self._parse_legacy_filename(filename)
            
        except Exception as e:
            self.logger.debug(f"⚠️ Parse error {filename}: {str(e)}")
            return None
    
    def _create_metadata_from_match(self, filename: str, pattern_type: str, match) -> FileMetadata:
        """🔧 Create metadata dari regex match"""
        if pattern_type == 'raw':
            nominal, uuid_str, sequence, extension = match.groups()
            return FileMetadata(
                filename=filename,
                file_type='raw',
                nominal=nominal,
                uuid_str=uuid_str,
                sequence=int(sequence),
                extension=extension,
                pattern_matched=pattern_type,
                denomination_info=self.denomination_map.get(nominal, self.denomination_map['000000'])
            )
        
        elif pattern_type in ['preprocessed', 'augmented']:
            nominal, uuid_str, sequence, variance, extension = match.groups()
            return FileMetadata(
                filename=filename,
                file_type=pattern_type,
                nominal=nominal,
                uuid_str=uuid_str,
                sequence=int(sequence),
                variance=int(variance),
                extension=extension,
                pattern_matched=pattern_type,
                denomination_info=self.denomination_map.get(nominal, self.denomination_map['000000'])
            )
        
        return None
    
    def _parse_legacy_filename(self, filename: str) -> FileMetadata:
        """📜 Parse legacy filename dengan heuristic detection"""
        stem = Path(filename).stem
        extension = Path(filename).suffix[1:]
        
        # Try extract nominal dari filename
        nominal = self._extract_nominal_from_text(filename)
        uuid_str = self._get_or_generate_uuid(filename)
        
        return FileMetadata(
            filename=filename,
            file_type='legacy',
            nominal=nominal,
            uuid_str=uuid_str,
            sequence=1,
            extension=extension,
            pattern_matched='legacy',
            denomination_info=self.denomination_map.get(nominal, self.denomination_map['000000'])
        )
    
    # === FILENAME GENERATION ===
    
    def generate_preprocessed_filename(self, source_metadata: Union[FileMetadata, str], 
                                     variance: int = 1, extension: str = 'npy') -> str:
        """🔧 Generate preprocessed filename dari source metadata"""
        if isinstance(source_metadata, str):
            source_metadata = self.parse_filename(source_metadata)
        
        if not source_metadata:
            # Fallback generation
            return f"pre_rp_000000_{str(uuid.uuid4())}_{1:03d}_{variance:02d}.{extension}"
        
        return f"pre_rp_{source_metadata.nominal}_{source_metadata.uuid_str}_{source_metadata.sequence:03d}_{variance:02d}.{extension}"
    
    def generate_augmented_filename(self, source_metadata: Union[FileMetadata, str], 
                                   variance: int = 1, extension: str = 'jpg') -> str:
        """🔧 Generate augmented filename dari source metadata"""
        if isinstance(source_metadata, str):
            source_metadata = self.parse_filename(source_metadata)
        
        if not source_metadata:
            return f"aug_rp_000000_{str(uuid.uuid4())}_{1:03d}_{variance:02d}.{extension}"
        
        return f"aug_rp_{source_metadata.nominal}_{source_metadata.uuid_str}_{source_metadata.sequence:03d}_{variance:02d}.{extension}"
    
    def generate_raw_filename(self, nominal: str, sequence: int = 1, extension: str = 'jpg') -> str:
        """🔧 Generate raw filename dengan new UUID"""
        uuid_str = str(uuid.uuid4())
        return f"rp_{nominal}_{uuid_str}_{sequence:03d}.{extension}"
    
    def create_matching_label_filename(self, image_filename: str) -> str:
        """📋 Create matching label filename untuk image"""
        stem = Path(image_filename).stem
        return f"{stem}.txt"
    
    # === METADATA OPERATIONS ===
    
    def get_file_metadata(self, filename: str, include_stats: bool = False) -> Dict[str, Any]:
        """📊 Get comprehensive file metadata"""
        if filename in self._metadata_cache:
            return self._metadata_cache[filename]
        
        parsed = self.parse_filename(filename)
        
        metadata = {
            'filename': filename,
            'is_valid': parsed is not None,
            'parsed': parsed.to_dict() if parsed else None,
            'is_research_format': parsed.file_type in ['raw', 'preprocessed', 'augmented'] if parsed else False
        }
        
        if parsed:
            metadata.update({
                'denomination': parsed.denomination_info,
                'file_type': parsed.file_type,
                'pattern': parsed.pattern_matched,
                'can_process': parsed.file_type in ['raw', 'legacy']
            })
        
        if include_stats:
            metadata['stats'] = self._calculate_filename_stats(filename)
        
        # Cache hasil
        self._metadata_cache[filename] = metadata
        return metadata
    
    def validate_filename_consistency(self, image_filename: str, label_filename: str) -> Dict[str, Any]:
        """🔍 Validate consistency antara image dan label filenames"""
        img_meta = self.parse_filename(image_filename)
        lbl_meta = self.parse_filename(label_filename)
        
        if not img_meta or not lbl_meta:
            return {
                'is_consistent': False,
                'error': 'Cannot parse filenames',
                'image_valid': img_meta is not None,
                'label_valid': lbl_meta is not None
            }
        
        # Consistency checks
        checks = {
            'nominal_match': img_meta.nominal == lbl_meta.nominal,
            'uuid_match': img_meta.uuid_str == lbl_meta.uuid_str,
            'sequence_match': img_meta.sequence == lbl_meta.sequence,
            'type_match': img_meta.file_type == lbl_meta.file_type
        }
        
        if img_meta.variance is not None and lbl_meta.variance is not None:
            checks['variance_match'] = img_meta.variance == lbl_meta.variance
        
        return {
            'is_consistent': all(checks.values()),
            'checks': checks,
            'image_metadata': img_meta.to_dict(),
            'label_metadata': lbl_meta.to_dict()
        }
    
    def extract_dataset_statistics(self, filenames: List[str]) -> Dict[str, Any]:
        """📈 Extract statistics dari dataset filenames"""
        stats = {
            'total_files': len(filenames),
            'by_type': {'raw': 0, 'preprocessed': 0, 'augmented': 0, 'legacy': 0, 'invalid': 0},
            'by_denomination': {k: 0 for k in self.denomination_map.keys()},
            'patterns_found': set(),
            'uuid_count': set(),
            'max_sequence': 0,
            'max_variance': 0
        }
        
        for filename in filenames:
            metadata = self.parse_filename(filename)
            
            if metadata:
                stats['by_type'][metadata.file_type] += 1
                stats['by_denomination'][metadata.nominal] += 1
                stats['patterns_found'].add(metadata.pattern_matched)
                stats['uuid_count'].add(metadata.uuid_str)
                stats['max_sequence'] = max(stats['max_sequence'], metadata.sequence)
                
                if metadata.variance:
                    stats['max_variance'] = max(stats['max_variance'], metadata.variance)
            else:
                stats['by_type']['invalid'] += 1
        
        # Convert set ke count
        stats['unique_uuids'] = len(stats['uuid_count'])
        stats['patterns_found'] = list(stats['patterns_found'])
        del stats['uuid_count']  # Remove set for JSON serialization
        
        return stats
    
    # === UTILITY METHODS ===
    
    def _extract_nominal_from_text(self, text: str) -> str:
        """💰 Extract nominal dari text menggunakan pattern matching"""
        text_lower = text.lower()
        
        # Pattern mapping untuk nominal extraction
        patterns = [
            (r'100000|rp100000|100k', '100000'),
            (r'50000|rp50000|50k', '050000'),
            (r'20000|rp20000|20k', '020000'),
            (r'10000|rp10000|10k', '010000'),
            (r'5000|rp5000|5k', '005000'),
            (r'2000|rp2000|2k', '002000'),
            (r'1000|rp1000|1k', '001000')
        ]
        
        for pattern, nominal in patterns:
            if re.search(pattern, text_lower):
                return nominal
        
        return '000000'  # Default unknown
    
    def _get_or_generate_uuid(self, filename: str) -> str:
        """🆔 Get cached UUID atau generate baru"""
        if filename not in self._uuid_cache:
            self._uuid_cache[filename] = str(uuid.uuid4())
        return self._uuid_cache[filename]
    
    def _calculate_filename_stats(self, filename: str) -> Dict[str, Any]:
        """📊 Calculate additional filename statistics"""
        return {
            'length': len(filename),
            'has_uuid': bool(re.search(r'[a-f0-9-]{36}', filename)),
            'has_sequence': bool(re.search(r'_\d{3}', filename)),
            'estimated_type': self._estimate_file_type(filename)
        }
    
    def _estimate_file_type(self, filename: str) -> str:
        """🔍 Estimate file type dari filename patterns"""
        filename_lower = filename.lower()
        
        if filename_lower.startswith('pre_'):
            return 'preprocessed'
        elif filename_lower.startswith('aug_'):
            return 'augmented'
        elif filename_lower.startswith('rp_'):
            return 'raw'
        else:
            return 'legacy'
    
    # === BATCH OPERATIONS ===
    
    def batch_parse_filenames(self, filenames: List[str]) -> Dict[str, Optional[FileMetadata]]:
        """📦 Batch parse filenames dengan caching"""
        results = {}
        for filename in filenames:
            results[filename] = self.parse_filename(filename)
        return results
    
    def batch_generate_preprocessed(self, source_filenames: List[str], 
                                   start_variance: int = 1) -> List[str]:
        """📦 Batch generate preprocessed filenames"""
        results = []
        for i, filename in enumerate(source_filenames):
            variance = start_variance + i
            results.append(self.generate_preprocessed_filename(filename, variance))
        return results
    
    def save_metadata_cache(self, cache_path: Union[str, Path]) -> bool:
        """💾 Save metadata cache ke file"""
        try:
            cache_data = {
                'uuid_cache': self._uuid_cache,
                'metadata_cache': {k: v for k, v in self._metadata_cache.items()},
                'timestamp': datetime.now().isoformat()
            }
            
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2, default=str)
            
            return True
        except Exception as e:
            self.logger.error(f"❌ Error saving cache: {str(e)}")
            return False
    
    def load_metadata_cache(self, cache_path: Union[str, Path]) -> bool:
        """📂 Load metadata cache dari file"""
        try:
            if not Path(cache_path).exists():
                return False
            
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
            
            self._uuid_cache.update(cache_data.get('uuid_cache', {}))
            self._metadata_cache.update(cache_data.get('metadata_cache', {}))
            
            return True
        except Exception as e:
            self.logger.error(f"❌ Error loading cache: {str(e)}")
            return False

# === FACTORY FUNCTIONS ===

def create_metadata_manager(config: Dict[str, Any] = None) -> MetadataManager:
    """🏭 Factory untuk create MetadataManager"""
    return MetadataManager(config)

# === CONVENIENCE FUNCTIONS ===

def parse_filename_safe(filename: str) -> Optional[FileMetadata]:
    """📋 One-liner safe filename parsing"""
    return create_metadata_manager().parse_filename(filename)

def generate_preprocessed_safe(source_filename: str, variance: int = 1) -> str:
    """🔧 One-liner safe preprocessed filename generation"""
    return create_metadata_manager().generate_preprocessed_filename(source_filename, variance)

def extract_nominal_safe(filename: str) -> str:
    """💰 One-liner safe nominal extraction"""
    metadata = parse_filename_safe(filename)
    return metadata.nominal if metadata else '000000'

def validate_pairs_safe(image_file: str, label_file: str) -> bool:
    """🔍 One-liner safe pair validation"""
    result = create_metadata_manager().validate_filename_consistency(image_file, label_file)
    return result.get('is_consistent', False)