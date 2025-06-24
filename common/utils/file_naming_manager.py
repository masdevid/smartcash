"""
File: smartcash/common/utils/file_naming_manager.py
Deskripsi: Enhanced FileNamingManager dengan pattern constants dan augmentation support
"""

import uuid
import re
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from pathlib import Path
from smartcash.common.logger import get_logger

@dataclass
class FileNamingInfo:
    """Container untuk informasi penamaan file dengan variance support"""
    uuid: str
    nominal: str
    class_id: str
    original_name: str
    source_type: str = 'raw'
    variance: int = 1  # untuk augmented files
    
    def get_filename(self, extension: str = None) -> str:
        """Generate filename berdasarkan source_type dengan variance support"""
        ext = extension or Path(self.original_name).suffix
        
        if self.source_type == 'raw':
            return f"rp_{self.nominal}_{self.uuid}{ext}"
        elif self.source_type == 'preprocessed':
            return f"pre_{self.nominal}_{self.uuid}{ext}"
        elif self.source_type == 'augmented':
            return f"aug_{self.nominal}_{self.uuid}_{self.variance:03d}{ext}"
        elif self.source_type == 'sample':
            return f"sample_pre_{self.nominal}_{self.uuid}{ext}"
        elif self.source_type == 'augmented_sample':
            return f"sample_aug_{self.nominal}_{self.uuid}_{self.variance:03d}{ext}"
        else:
            return f"rp_{self.nominal}_{self.uuid}{ext}"

class FileNamingManager:
    """Enhanced manager dengan pattern constants dan augmentation support"""
    
    # Pattern constants untuk semua file types dengan variance untuk augmented
    FILENAME_PATTERNS = {
        'raw': re.compile(r'rp_(\d{6})_([a-f0-9-]{36})\.(\w+)'),
        'preprocessed': re.compile(r'pre_(\d{6})_([a-f0-9-]{36})\.(\w+)'),
        'augmented': re.compile(r'aug_(\d{6})_([a-f0-9-]{36})_(\d{3})\.(\w+)'),
        'sample': re.compile(r'sample_pre_(\d{6})_([a-f0-9-]{36})\.(\w+)'),
        'augmented_sample': re.compile(r'sample_aug_(\d{6})_([a-f0-9-]{36})_(\d{3})\.(\w+)')
    }
    
    # Mapping class ID ke nominal
    CLASS_TO_NOMINAL = {
        0: '001000', 1: '002000', 2: '005000', 3: '010000', 4: '020000', 5: '050000', 6: '100000',
        7: '001000', 8: '002000', 9: '005000', 10: '010000', 11: '020000', 12: '050000', 13: '100000',
        14: '000000', 15: '000000', 16: '000000'
    }
    
    # Reverse mapping untuk lookup
    NOMINAL_TO_DESCRIPTION = {
        '001000': 'Rp1000', '002000': 'Rp2000', '005000': 'Rp5000', '010000': 'Rp10000',
        '020000': 'Rp20000', '050000': 'Rp50000', '100000': 'Rp100000', '000000': 'Undetermined'
    }
    
    # Pattern prefixes untuk file scanning
    PATTERN_PREFIXES = {
        'raw': 'rp_',
        'preprocessed': 'pre_',
        'augmented': 'aug_',
        'sample': 'sample_pre_',
        'augmented_sample': 'sample_aug_'
    }
    
    def __init__(self, config: Dict[str, Any] = None, logger=None):
        self.config = config or {}
        self.logger = logger or get_logger('file_naming')
        self.uuid_registry: Dict[str, str] = {}
        
    def generate_file_info(self, original_filename: str, primary_class_id: Optional[str] = None, 
                          source_type: str = 'raw', variance: int = 1) -> FileNamingInfo:
        """Generate file naming info dengan UUID consistency dan variance support"""
        original_stem = Path(original_filename).stem
        
        # UUID consistency
        file_uuid = self.uuid_registry.get(original_stem, str(uuid.uuid4()))
        self.uuid_registry[original_stem] = file_uuid
        
        nominal = self._extract_nominal_from_class_id(primary_class_id)
        class_id = str(primary_class_id) if primary_class_id is not None else '0'
        
        return FileNamingInfo(
            uuid=file_uuid, nominal=nominal, class_id=class_id,
            original_name=original_filename, source_type=source_type, variance=variance
        )
    
    def parse_filename(self, filename: str) -> Optional[Dict[str, str]]:
        """Parse filename menggunakan pattern constants dengan variance support"""
        for file_type, pattern in self.FILENAME_PATTERNS.items():
            match = pattern.match(filename)
            if match:
                if file_type in ['augmented', 'augmented_sample']:
                    # Handle augmented patterns dengan variance
                    nominal, uuid_str, variance, extension = match.groups()
                    return {
                        'type': file_type,
                        'nominal': nominal,
                        'uuid': uuid_str,
                        'variance': variance,
                        'extension': extension,
                        'description': self.NOMINAL_TO_DESCRIPTION.get(nominal, 'Unknown'),
                        'prefix': self.PATTERN_PREFIXES[file_type]
                    }
                else:
                    # Handle non-augmented patterns
                    nominal, uuid_str, extension = match.groups()
                    return {
                        'type': file_type,
                        'nominal': nominal,
                        'uuid': uuid_str,
                        'extension': extension,
                        'description': self.NOMINAL_TO_DESCRIPTION.get(nominal, 'Unknown'),
                        'prefix': self.PATTERN_PREFIXES[file_type]
                    }
        return None
    
    def validate_filename_format(self, filename: str) -> Dict[str, Any]:
        """Enhanced validation menggunakan pattern constants"""
        parsed = self.parse_filename(filename)
        
        if parsed:
            return {
                'valid': True, 
                'format': parsed['type'], 
                'parsed': parsed, 
                'message': f'Valid {parsed["type"]} format'
            }
        
        # Check legacy formats
        if any(pattern in filename.lower() for pattern in ['aug_', 'rp_', 'rupiah', 'pre_', 'sample_']):
            return {'valid': False, 'format': 'legacy', 'message': 'Format lama, perlu di-rename'}
        
        return {'valid': False, 'format': 'unknown', 'message': 'Format tidak dikenali'}
    
    def get_pattern(self, file_type: str) -> Optional[re.Pattern]:
        """Get pattern untuk specific file type"""
        return self.FILENAME_PATTERNS.get(file_type)
    
    def get_prefix(self, file_type: str) -> Optional[str]:
        """Get prefix untuk specific file type"""
        return self.PATTERN_PREFIXES.get(file_type)
    
    def is_valid_format(self, filename: str, file_type: str = None) -> bool:
        """Check jika filename valid untuk specific type atau any type"""
        if file_type:
            pattern = self.FILENAME_PATTERNS.get(file_type)
            return pattern.match(filename) is not None if pattern else False
        
        return any(pattern.match(filename) for pattern in self.FILENAME_PATTERNS.values())
    
    def generate_corresponding_filename(self, filename: str, target_type: str, 
                                      target_extension: str = None, variance: int = 1) -> str:
        """Generate corresponding filename dengan variance support"""
        parsed = self.parse_filename(filename)
        if not parsed:
            return filename
        
        file_info = FileNamingInfo(
            uuid=parsed['uuid'],
            nominal=parsed['nominal'],
            class_id='0',
            original_name=filename,
            source_type=target_type,
            variance=variance
        )
        
        return file_info.get_filename(target_extension)
    
    def _extract_nominal_from_class_id(self, class_id: Optional[str]) -> str:
        """Extract nominal dari class ID dengan priority mapping"""
        if class_id is None:
            return '000000'
        
        try:
            class_int = int(float(class_id))
            return self.CLASS_TO_NOMINAL.get(class_int, '000000')
        except (ValueError, TypeError):
            return '000000'
    
    def extract_primary_class_from_label(self, label_path: Path) -> Optional[str]:
        """Extract primary class dengan layer priority"""
        if not label_path.exists():
            return None
        
        try:
            class_counts = {}
            layer_priority = {0: 10, 1: 10, 2: 10, 3: 10, 4: 10, 5: 10, 6: 10,  # Layer 1
                            7: 9, 8: 9, 9: 9, 10: 9, 11: 9, 12: 9, 13: 9}       # Layer 2
            
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            class_id = int(float(parts[0]))
                            priority = layer_priority.get(class_id, 1)
                            class_counts[class_id] = class_counts.get(class_id, 0) + priority
                        except (ValueError, IndexError):
                            continue
            
            return str(max(class_counts.items(), key=lambda x: x[1])[0]) if class_counts else None
            
        except Exception as e:
            self.logger.debug(f"🔍 Error extracting class dari {label_path}: {str(e)}")
            return None
    
    def get_supported_types(self) -> List[str]:
        """Get list of supported file types"""
        return list(self.FILENAME_PATTERNS.keys())
    
    def get_nominal_statistics(self) -> Dict[str, Any]:
        """Get statistics dari processed files"""
        return {
            'total_files': len(self.uuid_registry), 
            'unique_uuids': len(set(self.uuid_registry.values())),
            'supported_types': len(self.FILENAME_PATTERNS),
            'nominal_coverage': f"{len([n for n in self.CLASS_TO_NOMINAL.values() if n != '000000'])}/7 denominations"
        }
    
    def cleanup_registry(self, keep_recent: int = 1000) -> None:
        """Cleanup registry untuk memory management"""
        if len(self.uuid_registry) > keep_recent:
            recent_items = list(self.uuid_registry.items())[-keep_recent:]
            self.uuid_registry = dict(recent_items)
            self.logger.info(f"🧹 Registry cleaned, kept {len(self.uuid_registry)} recent entries")

# Factory functions dan utilities
def create_file_naming_manager(config: Dict[str, Any] = None) -> FileNamingManager:
    """Factory untuk FileNamingManager"""
    return FileNamingManager(config or {})

def get_pattern(file_type: str) -> Optional[re.Pattern]:
    """One-liner pattern getter"""
    return FileNamingManager.FILENAME_PATTERNS.get(file_type)

def get_prefix(file_type: str) -> Optional[str]:
    """One-liner prefix getter"""
    return FileNamingManager.PATTERN_PREFIXES.get(file_type)

def parse_any_filename(filename: str) -> Optional[Dict[str, str]]:
    """One-liner filename parsing untuk any supported type"""
    return create_file_naming_manager().parse_filename(filename)

def is_research_format(filename: str) -> bool:
    """One-liner check untuk research format"""
    return create_file_naming_manager().is_valid_format(filename)

def generate_filename(original_name: str, class_id: str = None, 
                     source_type: str = 'raw', variance: int = 1) -> str:
    """One-liner filename generation dengan variance support"""
    manager = create_file_naming_manager()
    file_info = manager.generate_file_info(original_name, class_id, source_type, variance)
    return file_info.get_filename()

def generate_augmented_filename(original_name: str, variance: int = 1, class_id: str = None) -> str:
    """One-liner augmented filename generation"""
    return generate_filename(original_name, class_id, 'augmented', variance)

def generate_augmented_sample_filename(original_name: str, variance: int = 1, class_id: str = None) -> str:
    """One-liner augmented sample filename generation"""
    return generate_filename(original_name, class_id, 'augmented_sample', variance)

# Convenience functions dengan variance support
extract_variance_from_filename = lambda filename: (int(parsed['variance']) if (parsed := parse_any_filename(filename)) and 'variance' in parsed else 1)

# Convenience functions
extract_nominal_from_filename = lambda filename: (parsed['nominal'] if (parsed := parse_any_filename(filename)) else '000000')
get_nominal_from_class = lambda class_id: FileNamingManager.CLASS_TO_NOMINAL.get(int(class_id) if isinstance(class_id, str) and class_id.isdigit() else class_id, '000000')
get_description_from_nominal = lambda nominal: FileNamingManager.NOMINAL_TO_DESCRIPTION.get(nominal, 'Unknown')
extract_uuid_from_filename = lambda filename: (parsed['uuid'] if (parsed := parse_any_filename(filename)) else None)