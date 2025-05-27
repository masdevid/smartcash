"""
File: smartcash/common/utils/file_naming_manager.py
Deskripsi: Manager untuk penamaan file konsisten dengan UUID dan format penelitian SmartCash
"""

import uuid
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class FileNamingInfo:
    """Info lengkap tentang file naming untuk SmartCash"""
    uuid: str
    nominal: str
    stage: str  # 'raw', 'preprocessed', 'augmented', 'normalized'
    variant: Optional[str] = None  # untuk augmentasi
    extension: str = '.jpg'
    
    def get_filename(self) -> str:
        """Generate filename berdasarkan stage dan info"""
        if self.stage == 'raw':
            return f"rp_{self.nominal}_{self.uuid}{self.extension}"
        elif self.stage == 'preprocessed':
            return f"pre_rp_{self.nominal}_{self.uuid}{self.extension}"
        elif self.stage == 'augmented':
            variant_suffix = f"_{self.variant}" if self.variant else ""
            return f"aug_rp_{self.nominal}_{self.uuid}{variant_suffix}{self.extension}"
        elif self.stage == 'normalized':
            variant_suffix = f"_{self.variant}" if self.variant else ""
            return f"aug_rp_{self.nominal}_{self.uuid}{variant_suffix}{self.extension}"
        return f"{self.uuid}{self.extension}"

class FileNamingManager:
    """Manager untuk konsistensi penamaan file dengan UUID tracking"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.uuid_registry: Dict[str, str] = {}  # original_name -> uuid mapping
        
        # Nominal mapping untuk konsistensi
        self.nominal_map = {
            '0': '001000', '1': '002000', '2': '005000', '3': '010000',
            '4': '020000', '5': '050000', '6': '100000',
            '7': '001000', '8': '002000', '9': '005000', '10': '010000',
            '11': '020000', '12': '050000', '13': '100000'
        }
    
    def parse_existing_filename(self, filename: str) -> Optional[FileNamingInfo]:
        """Parse filename existing untuk extract info"""
        stem = Path(filename).stem
        ext = Path(filename).suffix or '.jpg'
        
        # Pattern untuk berbagai stage
        patterns = {
            'raw': r'^rp_(\d{6})_([a-f0-9\-]{36})$',
            'preprocessed': r'^pre_rp_(\d{6})_([a-f0-9\-]{36})$',
            'augmented': r'^aug_rp_(\d{6})_([a-f0-9\-]{36})(?:_(\d{3}))?$',
            'normalized': r'^aug_rp_(\d{6})_([a-f0-9\-]{36})(?:_(\d{3}))?$'
        }
        
        for stage, pattern in patterns.items():
            match = re.match(pattern, stem)
            if match:
                nominal = match.group(1)
                file_uuid = match.group(2)
                variant = match.group(3) if len(match.groups()) > 2 else None
                
                return FileNamingInfo(
                    uuid=file_uuid, nominal=nominal, stage=stage,
                    variant=variant, extension=ext
                )
        
        return None
    
    def generate_file_info(self, original_filename: str, class_id: Optional[str] = None, stage: str = 'raw') -> FileNamingInfo:
        """Generate file info baru atau dari existing UUID"""
        # Cari existing UUID
        base_name = Path(original_filename).stem
        existing_uuid = self.uuid_registry.get(base_name)
        
        if not existing_uuid:
            existing_uuid = str(uuid.uuid4())
            self.uuid_registry[base_name] = existing_uuid
        
        # Determine nominal dari class_id atau filename
        nominal = self._determine_nominal(original_filename, class_id)
        ext = Path(original_filename).suffix or '.jpg'
        
        return FileNamingInfo(
            uuid=existing_uuid, nominal=nominal, stage=stage, extension=ext
        )
    
    def create_variant_info(self, base_info: FileNamingInfo, variant_num: int, stage: str = 'augmented') -> FileNamingInfo:
        """Create variant info untuk augmentasi"""
        return FileNamingInfo(
            uuid=base_info.uuid, nominal=base_info.nominal, stage=stage,
            variant=f"{variant_num:03d}", extension=base_info.extension
        )
    
    def get_consistent_uuid(self, filename: str) -> str:
        """Get atau create UUID konsisten untuk file"""
        parsed = self.parse_existing_filename(filename)
        if parsed:
            return parsed.uuid
        
        base_name = Path(filename).stem
        if base_name not in self.uuid_registry:
            self.uuid_registry[base_name] = str(uuid.uuid4())
        
        return self.uuid_registry[base_name]
    
    def _determine_nominal(self, filename: str, class_id: Optional[str] = None) -> str:
        """Determine nominal dari class_id atau filename analysis"""
        if class_id and class_id in self.nominal_map:
            return self.nominal_map[class_id]
        
        # Extract dari existing filename jika ada pattern
        parsed = self.parse_existing_filename(filename)
        if parsed:
            return parsed.nominal
        
        # Default fallback
        return '001000'
    
    def batch_rename_files(self, file_paths: list, target_stage: str = 'preprocessed') -> Dict[str, str]:
        """Batch rename files dengan konsistensi UUID"""
        rename_map = {}
        
        for file_path in file_paths:
            original_path = Path(file_path)
            
            # Generate atau parse info
            if target_stage == 'preprocessed':
                base_info = self.generate_file_info(original_path.name, stage='raw')
                new_info = FileNamingInfo(
                    uuid=base_info.uuid, nominal=base_info.nominal,
                    stage='preprocessed', extension=base_info.extension
                )
            else:
                new_info = self.generate_file_info(original_path.name, stage=target_stage)
            
            new_filename = new_info.get_filename()
            new_path = original_path.parent / new_filename
            
            rename_map[str(original_path)] = str(new_path)
        
        return rename_map

# One-liner utilities untuk common operations
get_uuid_from_filename = lambda filename: FileNamingManager().get_consistent_uuid(filename)
parse_filename = lambda filename: FileNamingManager().parse_existing_filename(filename)
generate_preprocessed_name = lambda original_name, class_id=None: FileNamingManager().generate_file_info(original_name, class_id, 'preprocessed').get_filename()
generate_augmented_name = lambda original_name, variant_num, class_id=None: FileNamingManager().create_variant_info(FileNamingManager().generate_file_info(original_name, class_id, 'raw'), variant_num, 'augmented').get_filename()

def create_naming_manager(config: Dict[str, Any] = None) -> FileNamingManager:
    """Factory untuk FileNamingManager"""
    return FileNamingManager(config)