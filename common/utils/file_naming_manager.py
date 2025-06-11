"""
File: smartcash/common/utils/file_naming_manager.py
Deskripsi: Manager untuk format penamaan file dengan UUID consistency dan mapping nominal
"""

import uuid
import re
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from pathlib import Path
from smartcash.common.logger import get_logger

@dataclass
class FileNamingInfo:
    """Container untuk informasi penamaan file dengan one-liner methods"""
    uuid: str
    nominal: str
    class_id: str
    original_name: str
    source_type: str = 'raw'  # raw, augmented, preprocessed
    
    def get_filename(self, extension: str = None) -> str:
        """Generate filename dengan format: rp_{nominal}_{uuid}"""
        ext = extension or Path(self.original_name).suffix
        return f"rp_{self.nominal}_{self.uuid}{ext}"
    
    def _extract_sequence_from_original(self) -> int:
        """Extract sequence number dari original filename dengan one-liner regex"""
        matches = re.findall(r'(\d+)', Path(self.original_name).stem)
        return int(matches[-1]) if matches else 1

class FileNamingManager:
    """Manager untuk consistency penamaan file dengan UUID dan mapping nominal yang optimized"""
    
    # Mapping class ID ke nominal dengan one-liner initialization
    CLASS_TO_NOMINAL = {
        0: '001000', 1: '002000', 2: '005000', 3: '010000', 4: '020000', 5: '050000', 6: '100000',
        7: '001000', 8: '002000', 9: '005000', 10: '010000', 11: '020000', 12: '050000', 13: '100000',
        14: '000000', 15: '000000', 16: '000000'  # Layer 3 atau undetermined
    }
    
    # Reverse mapping untuk lookup
    NOMINAL_TO_DESCRIPTION = {
        '001000': 'Rp1000', '002000': 'Rp2000', '005000': 'Rp5000', '010000': 'Rp10000',
        '020000': 'Rp20000', '050000': 'Rp50000', '100000': 'Rp100000', '000000': 'Undetermined'
    }
    
    def __init__(self, config: Dict[str, Any] = None, logger=None):
        self.config = config or {}
        self.logger = logger or get_logger('file_naming')
        self.uuid_registry: Dict[str, str] = {}  # original_stem -> uuid mapping
        
    def generate_file_info(self, original_filename: str, primary_class_id: Optional[str] = None, 
                          source_type: str = 'raw') -> FileNamingInfo:
        """Generate file naming info dengan UUID consistency dan nominal extraction"""
        original_stem = Path(original_filename).stem
        
        # UUID consistency - reuse jika sudah ada
        file_uuid = self.uuid_registry.get(original_stem, str(uuid.uuid4()))
        self.uuid_registry[original_stem] = file_uuid
        
        # Extract nominal dari class ID dengan priority layer 1-2
        nominal = self._extract_nominal_from_class_id(primary_class_id)
        class_id = str(primary_class_id) if primary_class_id is not None else '0'
        
        return FileNamingInfo(
            uuid=file_uuid, nominal=nominal, class_id=class_id,
            original_name=original_filename, source_type=source_type
        )
    
    def _extract_nominal_from_class_id(self, class_id: Optional[str]) -> str:
        """Extract nominal dari class ID dengan priority mapping dan one-liner fallback"""
        if class_id is None:
            return '000000'
        
        try:
            class_int = int(float(class_id))  # Support float string format
            return self.CLASS_TO_NOMINAL.get(class_int, '000000')
        except (ValueError, TypeError):
            return '000000'
    
    def extract_primary_class_from_label(self, label_path: Path) -> Optional[str]:
        """Extract primary class dari YOLO label dengan priority layer 1-2"""
        if not label_path.exists():
            return None
        
        try:
            class_counts = {}
            layer_priority = {0: 10, 1: 10, 2: 10, 3: 10, 4: 10, 5: 10, 6: 10,  # Layer 1 - highest priority
                            7: 9, 8: 9, 9: 9, 10: 9, 11: 9, 12: 9, 13: 9}       # Layer 2 - second priority
            
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            class_id = int(float(parts[0]))
                            priority = layer_priority.get(class_id, 1)  # Layer 3 atau unknown = lowest
                            class_counts[class_id] = class_counts.get(class_id, 0) + priority
                        except (ValueError, IndexError):
                            continue
            
            # Return class dengan highest priority+count
            return str(max(class_counts.items(), key=lambda x: x[1])[0]) if class_counts else None
            
        except Exception as e:
            self.logger.debug(f"🔍 Error extracting class dari {label_path}: {str(e)}")
            return None
    
    def parse_existing_filename(self, filename: str) -> Optional[Dict[str, str]]:
        """Parse existing filename dengan format rp_{nominal}_{uuid}_{sequence}"""
        pattern = r'^rp_(\d{6})_([0-9a-f-]{36})'
        match = re.match(pattern, Path(filename).stem)
        
        if match:
            return {
                'nominal': match.group(1), 'uuid': match.group(2),
                'description': self.NOMINAL_TO_DESCRIPTION.get(match.group(1), 'Unknown')
            }
        return None
    
    def validate_filename_format(self, filename: str) -> Dict[str, Any]:
        """Validate filename format dengan detailed feedback"""
        parsed = self.parse_existing_filename(filename)
        
        if parsed:
            return {'valid': True, 'format': 'rp_format', 'parsed': parsed, 'message': 'Format valid'}
        
        # Check legacy formats atau other patterns
        if any(pattern in filename.lower() for pattern in ['aug_', 'rp_', 'rupiah']):
            return {'valid': False, 'format': 'legacy', 'message': 'Format lama, perlu di-rename'}
        
        return {'valid': False, 'format': 'unknown', 'message': 'Format tidak dikenali'}
    
    def get_nominal_statistics(self) -> Dict[str, Any]:
        """Get statistics dari file yang sudah di-process dengan one-liner aggregation"""
        nominal_counts = {}
        for original_stem, file_uuid in self.uuid_registry.items():
            # Assume UUID sudah mapped, hitung berdasarkan registry
            nominal_counts['total_files'] = nominal_counts.get('total_files', 0) + 1
        
        return {
            'total_files': len(self.uuid_registry), 'unique_uuids': len(set(self.uuid_registry.values())),
            'nominal_mapping_coverage': f"{len([n for n in self.CLASS_TO_NOMINAL.values() if n != '000000'])}/7 denominations"
        }
    
    def cleanup_registry(self, keep_recent: int = 1000) -> None:
        """Cleanup registry untuk memory management dengan one-liner recent keep"""
        if len(self.uuid_registry) > keep_recent:
            # Keep most recent entries (approximate by dict order in Python 3.7+)
            recent_items = list(self.uuid_registry.items())[-keep_recent:]
            self.uuid_registry = dict(recent_items)
            self.logger.info(f"🧹 Registry cleaned, kept {len(self.uuid_registry)} recent entries")

# Factory functions dan utilities dengan one-liner style
def create_file_naming_manager(config: Dict[str, Any] = None) -> FileNamingManager:
    """Factory untuk FileNamingManager dengan default config"""
    return FileNamingManager(config or {})

def extract_nominal_from_filename(filename: str) -> str:
    """One-liner nominal extraction dari filename"""
    manager = create_file_naming_manager()
    parsed = manager.parse_existing_filename(filename)
    return parsed['nominal'] if parsed else '000000'

def generate_uuid_filename(original_name: str, class_id: str = None, source_type: str = 'raw') -> str:
    """One-liner UUID filename generation"""
    manager = create_file_naming_manager()
    file_info = manager.generate_file_info(original_name, class_id, source_type)
    return file_info.get_filename()

def validate_rupiah_filename(filename: str) -> bool:
    """One-liner validation untuk rupiah filename format"""
    return create_file_naming_manager().validate_filename_format(filename)['valid']

# Convenience functions untuk common operations
get_nominal_from_class = lambda class_id: FileNamingManager.CLASS_TO_NOMINAL.get(int(class_id) if isinstance(class_id, str) and class_id.isdigit() else class_id, '000000')
get_description_from_nominal = lambda nominal: FileNamingManager.NOMINAL_TO_DESCRIPTION.get(nominal, 'Unknown')
is_valid_rp_format = lambda filename: bool(re.match(r'^rp_\d{6}_[0-9a-f-]{36}', Path(filename).stem))
extract_uuid_from_filename = lambda filename: (match.group(2) if (match := re.match(r'^rp_\d{6}_([0-9a-f-]{36})', Path(filename).stem)) else None)