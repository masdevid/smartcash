"""
File: smartcash/dataset/augmentor/utils/filename_manager.py  
Deskripsi: Manager untuk filename parsing dan generation
"""

import re
from typing import Optional, Dict, Any

class FilenameManager:
    """📝 Manager untuk filename parsing dan generation sesuai format penelitian"""
    
    def __init__(self):
        # Pattern untuk raw files: rp_001000_uuid.ext
        self.raw_pattern = re.compile(r'rp_(\d{6})_([a-f0-9-]{36})\.(\w+)')
        
        # Pattern untuk preprocessed: pre_rp_001000_uuid_variance.ext  
        self.prep_pattern = re.compile(r'pre_rp_(\d{6})_([a-f0-9-]{36})_(\d+)\.(\w+)')
        
        # Pattern untuk augmented: aug_rp_001000_uuid_variance.ext
        self.aug_pattern = re.compile(r'aug_rp_(\d{6})_([a-f0-9-]{36})_(\d+)\.(\w+)')
    
    def parse_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """Parse filename dan extract metadata"""
        # Try raw pattern first
        match = self.raw_pattern.match(filename)
        if match:
            return {
                'type': 'raw',
                'nominal': match.group(1),
                'uuid': match.group(2), 
                'extension': match.group(3)
            }
        
        # Try preprocessed pattern
        match = self.prep_pattern.match(filename)
        if match:
            return {
                'type': 'preprocessed',
                'nominal': match.group(1),
                'uuid': match.group(2),
                'variance': match.group(3),
                'extension': match.group(4)
            }
        
        # Try augmented pattern
        match = self.aug_pattern.match(filename)
        if match:
            return {
                'type': 'augmented',
                'nominal': match.group(1),
                'uuid': match.group(2),
                'variance': match.group(3),
                'extension': match.group(4)
            }
        
        return None
    
    def create_augmented_filename(self, parsed_info: Dict[str, Any], variance: int) -> str:
        """Create augmented filename dari parsed info"""
        if parsed_info['type'] == 'raw':
            return f"aug_rp_{parsed_info['nominal']}_{parsed_info['uuid']}_{variance:02d}"
        elif parsed_info['type'] == 'preprocessed':
            # Increment variance dari preprocessed
            new_variance = int(parsed_info['variance']) + variance
            return f"aug_rp_{parsed_info['nominal']}_{parsed_info['uuid']}_{new_variance:02d}"
        else:
            # Fallback
            return f"aug_{parsed_info.get('nominal', '000000')}_{variance:02d}"