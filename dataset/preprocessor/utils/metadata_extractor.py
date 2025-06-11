"""
File: smartcash/dataset/preprocessor/utils/metadata_extractor.py
Deskripsi: Metadata extraction dari filenames dan file content
"""

import re
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union

from smartcash.common.logger import get_logger
from ..config.defaults import MAIN_BANKNOTE_CLASSES, LAYER_CLASSES

class MetadataExtractor:
    """📋 Extract metadata dari files dan filenames"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # Patterns untuk research format
        self.patterns = {
            'raw': re.compile(r'rp_(\d{6})_([a-f0-9-]{36})_(\d+)\.(\w+)'),
            'preprocessed': re.compile(r'pre_rp_(\d{6})_([a-f0-9-]{36})_(\d+)_(\d+)\.(\w+)'),
            'augmented': re.compile(r'aug_rp_(\d{6})_([a-f0-9-]{36})_(\d+)_(\d+)\.(\w+)'),
            'sample': re.compile(r'sample_rp_(\d{6})_([a-f0-9-]{36})_(\d+)\.(\w+)')
        }
    
    def extract_file_metadata(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """📊 Extract comprehensive file metadata"""
        path = Path(file_path)
        
        metadata = {
            'filename': path.name,
            'file_type': 'unknown',
            'exists': path.exists(),
            'size_bytes': 0,
            'size_mb': 0,
            'research_format': False,
            'components': {}
        }
        
        if not path.exists():
            return metadata
        
        # Basic file info
        stat = path.stat()
        metadata.update({
            'size_bytes': stat.st_size,
            'size_mb': round(stat.st_size / (1024 * 1024), 2),
            'extension': path.suffix.lower()
        })
        
        # Parse filename
        filename_data = self.parse_research_filename(path.name)
        if filename_data:
            metadata.update({
                'research_format': True,
                'file_type': filename_data['type'],
                'components': filename_data
            })
        
        # Load additional metadata untuk .npy files
        if path.suffix.lower() == '.npy':
            meta_file = path.with_suffix('.meta.json')
            if meta_file.exists():
                try:
                    with open(meta_file, 'r') as f:
                        additional_meta = json.load(f)
                    metadata['normalization_info'] = additional_meta
                except Exception:
                    pass
        
        return metadata
    
    def parse_research_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """📝 Parse research format filename"""
        for file_type, pattern in self.patterns.items():
            match = pattern.match(filename)
            if match:
                if file_type == 'raw' or file_type == 'sample':
                    nominal, uuid_str, sequence, extension = match.groups()
                    return {
                        'type': file_type,
                        'nominal': nominal,
                        'uuid': uuid_str,
                        'sequence': int(sequence),
                        'extension': extension,
                        'class_info': self._get_class_info_from_nominal(nominal)
                    }
                else:  # preprocessed, augmented
                    nominal, uuid_str, sequence, variance, extension = match.groups()
                    return {
                        'type': file_type,
                        'nominal': nominal,
                        'uuid': uuid_str,
                        'sequence': int(sequence),
                        'variance': int(variance),
                        'extension': extension,
                        'class_info': self._get_class_info_from_nominal(nominal)
                    }
        
        return None
    
    def _get_class_info_from_nominal(self, nominal: str) -> Dict[str, Any]:
        """💰 Get class info dari nominal string"""
        # Map nominal ke class_id (main banknotes only)
        nominal_to_class = {
            '001000': 0, '002000': 1, '005000': 2, '010000': 3,
            '020000': 4, '050000': 5, '100000': 6
        }
        
        class_id = nominal_to_class.get(nominal, -1)
        
        if class_id in MAIN_BANKNOTE_CLASSES:
            class_info = MAIN_BANKNOTE_CLASSES[class_id].copy()
            class_info['class_id'] = class_id
            class_info['layer'] = 'l1_main'
            return class_info
        
        return {
            'class_id': -1,
            'nominal': nominal,
            'display': f'Unknown {nominal}',
            'layer': 'unknown'
        }
    
    def get_class_ids_from_labels(self, label_path: Union[str, Path]) -> list:
        """🏷️ Extract class IDs dari YOLO label file"""
        try:
            path = Path(label_path)
            if not path.exists():
                return []
            
            class_ids = []
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        try:
                            parts = line.split()
                            if len(parts) >= 1:
                                class_id = int(float(parts[0]))
                                class_ids.append(class_id)
                        except (ValueError, IndexError):
                            continue
            
            return list(set(class_ids))  # Unique class IDs
            
        except Exception as e:
            self.logger.debug(f"⚠️ Error reading labels {label_path}: {str(e)}")
            return []
    
    def get_class_names_from_ids(self, class_ids: list) -> Dict[int, str]:
        """🏷️ Map class IDs ke human-readable names"""
        class_names = {}
        
        for class_id in class_ids:
            if class_id in MAIN_BANKNOTE_CLASSES:
                class_names[class_id] = MAIN_BANKNOTE_CLASSES[class_id]['display']
            else:
                # Check layer classes
                for layer_name, class_range in LAYER_CLASSES.items():
                    if class_id in class_range:
                        class_names[class_id] = f"{layer_name}_{class_id:02d}"
                        break
                else:
                    class_names[class_id] = f"class_{class_id}"
        
        return class_names
    
    def extract_sample_metadata(self, image_path: Union[str, Path], 
                              label_path: Union[str, Path] = None) -> Dict[str, Any]:
        """🎲 Extract metadata untuk sample generation"""
        img_path = Path(image_path)
        
        # Base metadata
        sample_meta = self.extract_file_metadata(img_path)
        
        # Auto-detect label path jika tidak provided
        if label_path is None:
            label_path = img_path.with_suffix('.txt')
            # Try labels directory structure
            if not label_path.exists() and img_path.parent.name == 'images':
                labels_dir = img_path.parent.parent / 'labels'
                label_path = labels_dir / f"{img_path.stem}.txt"
        
        # Extract class information dari labels
        if Path(label_path).exists():
            class_ids = self.get_class_ids_from_labels(label_path)
            
            # Filter hanya main banknote classes untuk samples
            main_class_ids = [cid for cid in class_ids if cid in MAIN_BANKNOTE_CLASSES]
            
            sample_meta.update({
                'has_labels': True,
                'all_class_ids': class_ids,
                'main_class_ids': main_class_ids,
                'class_names': self.get_class_names_from_ids(main_class_ids),
                'is_main_banknote': len(main_class_ids) > 0
            })
        else:
            sample_meta.update({
                'has_labels': False,
                'all_class_ids': [],
                'main_class_ids': [],
                'class_names': {},
                'is_main_banknote': False
            })
        
        return sample_meta

# === Factory functions ===
def extract_file_metadata(file_path: Union[str, Path]) -> Dict[str, Any]:
    """📊 One-liner file metadata extraction"""
    return MetadataExtractor().extract_file_metadata(file_path)

def parse_research_filename(filename: str) -> Optional[Dict[str, Any]]:
    """📝 One-liner filename parsing"""
    return MetadataExtractor().parse_research_filename(filename)