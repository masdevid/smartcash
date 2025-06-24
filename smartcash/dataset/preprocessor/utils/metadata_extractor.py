"""
File: smartcash/dataset/preprocessor/utils/metadata_extractor.py
Deskripsi: Simplified metadata extractor menggunakan FileNamingManager patterns
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Union

from smartcash.common.logger import get_logger
from smartcash.common.utils.file_naming_manager import create_file_naming_manager
from ..config.defaults import MAIN_BANKNOTE_CLASSES, LAYER_CLASSES

class MetadataExtractor:
    """📋 Simplified metadata extractor menggunakan FileNamingManager patterns"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.naming_manager = create_file_naming_manager()
    
    def extract_file_metadata(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """📊 Extract metadata menggunakan FileNamingManager"""
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
        
        # Parse filename menggunakan FileNamingManager
        filename_data = self.naming_manager.parse_filename(path.name)
        if filename_data:
            metadata.update({
                'research_format': True,
                'file_type': filename_data['type'],
                'components': {
                    **filename_data,
                    'class_info': self._get_class_info_from_nominal(filename_data['nominal'])
                }
            })
        
        # Load .npy metadata
        if path.suffix.lower() == '.npy':
            meta_file = path.with_suffix('.meta.json')
            if meta_file.exists():
                try:
                    with open(meta_file, 'r') as f:
                        metadata['normalization_info'] = json.load(f)
                except Exception:
                    pass
        
        return metadata
    
    def parse_research_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """📝 Parse research filename menggunakan FileNamingManager"""
        parsed = self.naming_manager.parse_filename(filename)
        if parsed:
            parsed['class_info'] = self._get_class_info_from_nominal(parsed['nominal'])
        return parsed
    
    def _get_class_info_from_nominal(self, nominal: str) -> Dict[str, Any]:
        """💰 Get class info dari nominal"""
        class_id = None
        for cid, nom in self.naming_manager.CLASS_TO_NOMINAL.items():
            if nom == nominal:
                class_id = cid
                break
        
        if class_id is not None and class_id in MAIN_BANKNOTE_CLASSES:
            class_info = MAIN_BANKNOTE_CLASSES[class_id].copy()
            class_info.update({'class_id': class_id, 'layer': 'l1_main'})
            return class_info
        
        return {
            'class_id': -1,
            'nominal': nominal,
            'display': self.naming_manager.NOMINAL_TO_DESCRIPTION.get(nominal, f'Unknown {nominal}'),
            'layer': 'unknown'
        }
    
    def get_class_ids_from_labels(self, label_path: Union[str, Path]) -> list:
        """🏷️ Extract class IDs dari YOLO label"""
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
                                class_ids.append(int(float(parts[0])))
                        except (ValueError, IndexError):
                            continue
            
            return list(set(class_ids))
            
        except Exception as e:
            self.logger.debug(f"⚠️ Error reading labels {label_path}: {str(e)}")
            return []
    
    def get_class_names_from_ids(self, class_ids: list) -> Dict[int, str]:
        """🏷️ Map class IDs ke names"""
        class_names = {}
        
        for class_id in class_ids:
            if class_id in MAIN_BANKNOTE_CLASSES:
                class_names[class_id] = MAIN_BANKNOTE_CLASSES[class_id]['display']
            else:
                for layer_name, class_range in LAYER_CLASSES.items():
                    if class_id in class_range:
                        class_names[class_id] = f"{layer_name}_{class_id:02d}"
                        break
                else:
                    class_names[class_id] = f"class_{class_id}"
        
        return class_names
    
    def extract_sample_metadata(self, image_path: Union[str, Path], 
                              label_path: Union[str, Path] = None) -> Dict[str, Any]:
        """🎲 Extract sample metadata"""
        img_path = Path(image_path)
        sample_meta = self.extract_file_metadata(img_path)
        
        if label_path is None:
            label_path = self._find_corresponding_label(img_path)
        
        if label_path and Path(label_path).exists():
            class_ids = self.get_class_ids_from_labels(label_path)
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
    
    def _find_corresponding_label(self, image_path: Path) -> Optional[Path]:
        """🔍 Find corresponding label menggunakan naming consistency"""
        parsed = self.naming_manager.parse_filename(image_path.name)
        
        if parsed:
            label_name = self.naming_manager.generate_corresponding_filename(
                image_path.name, parsed['type'], '.txt'
            )
        else:
            label_name = f"{image_path.stem}.txt"
        
        # Check same directory first
        same_dir_label = image_path.parent / label_name
        if same_dir_label.exists():
            return same_dir_label
        
        # Check labels directory structure
        if image_path.parent.name == 'images':
            labels_dir = image_path.parent.parent / 'labels'
            labels_dir_file = labels_dir / label_name
            if labels_dir_file.exists():
                return labels_dir_file
        
        return None
    
    def generate_preprocessed_filename(self, original_filename: str) -> str:
        """🔧 Generate preprocessed filename"""
        return self.naming_manager.generate_corresponding_filename(
            original_filename, 'preprocessed', '.npy'
        )
    
    def generate_sample_filename(self, preprocessed_filename: str) -> str:
        """🎲 Generate sample filename"""
        return self.naming_manager.generate_corresponding_filename(
            preprocessed_filename, 'sample', '.jpg'
        )
    
    def generate_augmented_filename(self, original_filename: str, variance: int = 1) -> str:
        """🔄 Generate augmented filename dengan variance"""
        parsed = self.naming_manager.parse_filename(original_filename)
        if parsed:
            return f"aug_{parsed['nominal']}_{parsed['uuid']}_{variance:03d}.{parsed['extension']}"
        
        # Fallback generation
        file_info = self.naming_manager.generate_file_info(original_filename, source_type='augmented')
        return f"aug_{file_info.nominal}_{file_info.uuid}_{variance:03d}.jpg"
    
    def validate_filename_consistency(self, image_path: Path, label_path: Path) -> Dict[str, Any]:
        """🔍 Validate consistency"""
        img_parsed = self.naming_manager.parse_filename(image_path.name)
        label_parsed = self.naming_manager.parse_filename(label_path.name)
        
        if not img_parsed or not label_parsed:
            return {
                'consistent': False,
                'reason': 'Invalid filename pattern',
                'img_valid': img_parsed is not None,
                'label_valid': label_parsed is not None
            }
        
        if img_parsed['nominal'] != label_parsed['nominal'] or img_parsed['uuid'] != label_parsed['uuid']:
            return {
                'consistent': False,
                'reason': 'Nominal/UUID mismatch',
                'img_nominal': img_parsed['nominal'],
                'label_nominal': label_parsed['nominal']
            }
        
        return {
            'consistent': True,
            'nominal': img_parsed['nominal'],
            'uuid': img_parsed['uuid'],
            'description': img_parsed['description']
        }
    
    def get_nominal_from_filename(self, filename: str) -> str:
        """💰 Extract nominal"""
        parsed = self.naming_manager.parse_filename(filename)
        return parsed['nominal'] if parsed else '000000'
    
    def is_research_format(self, filename: str) -> bool:
        """✅ Check research format"""
        return self.naming_manager.is_valid_format(filename)

# Factory functions
def extract_file_metadata(file_path: Union[str, Path]) -> Dict[str, Any]:
    """📊 One-liner metadata extraction"""
    return MetadataExtractor().extract_file_metadata(file_path)

def parse_research_filename(filename: str) -> Optional[Dict[str, Any]]:
    """📝 One-liner filename parsing"""
    return MetadataExtractor().parse_research_filename(filename)