"""
File: smartcash/dataset/preprocessor/utils/sample_generator.py
Deskripsi: Updated sample generator menggunakan FileNamingManager patterns
"""

import cv2
import numpy as np
import random
from pathlib import Path
from typing import Dict, Any, List, Union, Optional

from smartcash.common.logger import get_logger
from smartcash.common.utils.file_naming_manager import create_file_naming_manager
from ..config.defaults import MAIN_BANKNOTE_CLASSES

class SampleGenerator:
    """🎲 Updated sample generator dengan FileNamingManager integration"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = get_logger(__name__)
        self.naming_manager = create_file_naming_manager()
        
        self.max_samples = self.config.get('max_samples', 50)
        self.sample_size = self.config.get('sample_size', (640, 640))
        self.quality = self.config.get('jpg_quality', 85)
    
    def generate_samples(self, data_dir: Union[str, Path], 
                        output_dir: Union[str, Path],
                        splits: List[str] = None,
                        max_per_class: int = 5) -> Dict[str, Any]:
        """🎲 Generate samples dengan naming manager patterns"""
        data_path = Path(data_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        splits = splits or ['train', 'valid']
        results = {'total_generated': 0, 'by_split': {}, 'by_class': {}, 'samples': []}
        
        for split in splits:
            split_results = self._generate_split_samples(
                data_path / split, output_path / split, max_per_class
            )
            results['by_split'][split] = split_results
            results['total_generated'] += split_results['count']
            
            for class_id, count in split_results['by_class'].items():
                results['by_class'][class_id] = results['by_class'].get(class_id, 0) + count
            
            results['samples'].extend(split_results['samples'])
        
        return results
    
    def _generate_split_samples(self, split_dir: Path, output_dir: Path, 
                              max_per_class: int) -> Dict[str, Any]:
        """🎯 Generate samples untuk single split"""
        from ..core.file_processor import FileProcessor
        
        file_processor = FileProcessor()
        
        # Scan preprocessed files menggunakan naming manager
        npy_files = file_processor.scan_files(split_dir / 'images', 
                                            self.naming_manager.get_prefix('preprocessed'), 
                                            {'.npy'})
        
        if not npy_files:
            return {'count': 0, 'by_class': {}, 'samples': []}
        
        # Group by class menggunakan naming manager
        class_groups = {}
        for npy_file in npy_files:
            parsed = self.naming_manager.parse_filename(npy_file.name)
            if parsed and parsed['type'] == 'preprocessed':
                class_info = self._get_class_info_from_nominal(parsed['nominal'])
                class_id = class_info.get('class_id', -1)
                
                if class_id in MAIN_BANKNOTE_CLASSES:
                    if class_id not in class_groups:
                        class_groups[class_id] = []
                    class_groups[class_id].append(npy_file)
        
        # Sample dari each class
        output_dir.mkdir(parents=True, exist_ok=True)
        samples = []
        by_class = {}
        
        for class_id, files in class_groups.items():
            sample_count = min(max_per_class, len(files))
            sampled_files = random.sample(files, sample_count)
            by_class[class_id] = sample_count
            
            for npy_file in sampled_files:
                sample_info = self._create_sample_from_npy(npy_file, output_dir, class_id)
                if sample_info:
                    samples.append(sample_info)
        
        return {'count': len(samples), 'by_class': by_class, 'samples': samples}
    
    def _create_sample_from_npy(self, npy_file: Path, output_dir: Path, 
                              class_id: int) -> Optional[Dict[str, Any]]:
        """🖼️ Create sample dengan naming manager"""
        try:
            from ..core.file_processor import FileProcessor
            from ..core.normalizer import YOLONormalizer
            
            file_processor = FileProcessor()
            
            # Load normalized array
            normalized_array, metadata = file_processor.load_normalized_array(npy_file)
            if normalized_array is None:
                return None
            
            # Denormalize
            normalizer = YOLONormalizer()
            if metadata:
                denormalized = normalizer.denormalize(normalized_array, metadata)
            else:
                denormalized = (normalized_array * 255).astype(np.uint8)
                if len(denormalized.shape) == 3:
                    denormalized = denormalized[:, :, :3]
            
            # Generate sample filename menggunakan naming manager
            sample_name = self.naming_manager.generate_corresponding_filename(
                npy_file.name, 'sample', '.jpg'
            )
            sample_path = output_dir / sample_name
            
            # Save image
            success = cv2.imwrite(str(sample_path), cv2.cvtColor(denormalized, cv2.COLOR_RGB2BGR))
            
            if success:
                file_info = file_processor.get_file_info(sample_path)
                label_path = file_processor.find_corresponding_label(npy_file)
                
                class_ids = []
                class_names = {}
                if label_path and label_path.exists():
                    from .metadata_extractor import MetadataExtractor
                    metadata_extractor = MetadataExtractor()
                    class_ids = metadata_extractor.get_class_ids_from_labels(label_path)
                    class_names = metadata_extractor.get_class_names_from_ids(class_ids)
                
                # Parse filename untuk metadata
                parsed = self.naming_manager.parse_filename(npy_file.name)
                uuid_str = parsed['uuid'] if parsed else 'unknown'
                nominal = parsed['nominal'] if parsed else '000000'
                
                return {
                    'npy_path': str(npy_file),
                    'sample_path': str(sample_path),
                    'filename': sample_name,
                    'file_size_mb': file_info.get('size_mb', 0),
                    'class_ids': class_ids,
                    'class_names': class_names,
                    'main_class_id': class_id,
                    'nominal': nominal,
                    'display_name': MAIN_BANKNOTE_CLASSES[class_id]['display'],
                    'uuid': uuid_str
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Sample creation error {npy_file.name}: {str(e)}")
            return None
    
    def get_random_samples(self, data_dir: Union[str, Path], 
                          count: int = 10, 
                          split: str = 'train') -> List[Dict[str, Any]]:
        """🎲 Get random samples menggunakan naming manager"""
        from ..core.file_processor import FileProcessor
        
        data_path = Path(data_dir)
        split_path = data_path / split
        
        if not split_path.exists():
            return []
        
        file_processor = FileProcessor()
        
        # Scan preprocessed files
        npy_files = file_processor.scan_files(split_path / 'images', 
                                            self.naming_manager.get_prefix('preprocessed'), 
                                            {'.npy'})
        
        if not npy_files:
            return []
        
        # Filter main banknote classes
        main_banknote_files = []
        for npy_file in npy_files:
            parsed = self.naming_manager.parse_filename(npy_file.name)
            if parsed and parsed['type'] == 'preprocessed':
                class_info = self._get_class_info_from_nominal(parsed['nominal'])
                if class_info.get('class_id', -1) in MAIN_BANKNOTE_CLASSES:
                    main_banknote_files.append(npy_file)
        
        # Random sample
        sample_count = min(count, len(main_banknote_files))
        sampled_files = random.sample(main_banknote_files, sample_count)
        
        # Get metadata
        samples = []
        for npy_file in sampled_files:
            parsed = self.naming_manager.parse_filename(npy_file.name)
            if parsed:
                label_path = file_processor.find_corresponding_label(npy_file)
                
                sample_info = {
                    'npy_path': str(npy_file),
                    'filename': npy_file.name,
                    'file_size_mb': file_processor.get_file_info(npy_file).get('size_mb', 0),
                    'uuid': parsed['uuid'],
                    'nominal': parsed['nominal'],
                    'potential_sample_path': self.naming_manager.generate_corresponding_filename(
                        npy_file.name, 'sample', '.jpg'
                    )
                }
                
                # Add class info
                class_info = self._get_class_info_from_nominal(parsed['nominal'])
                if class_info.get('class_id', -1) in MAIN_BANKNOTE_CLASSES:
                    sample_info['primary_class'] = {
                        'class_id': class_info['class_id'],
                        'display': class_info['display'],
                        'nominal': class_info['nominal'],
                        'value': class_info['value']
                    }
                
                samples.append(sample_info)
        
        return samples
    
    def _get_class_info_from_nominal(self, nominal: str) -> Dict[str, Any]:
        """💰 Get class info dari nominal"""
        class_id = None
        for cid, nom in self.naming_manager.CLASS_TO_NOMINAL.items():
            if nom == nominal:
                class_id = cid
                break
        
        if class_id is not None and class_id in MAIN_BANKNOTE_CLASSES:
            class_info = MAIN_BANKNOTE_CLASSES[class_id].copy()
            class_info['class_id'] = class_id
            return class_info
        
        return {'class_id': -1, 'nominal': nominal, 'display': f'Unknown {nominal}'}
    
    def validate_sample_consistency(self, data_dir: Union[str, Path], 
                                  samples_dir: Union[str, Path],
                                  split: str = 'train') -> Dict[str, Any]:
        """✅ Validate consistency menggunakan naming manager"""
        from ..core.file_processor import FileProcessor
        
        data_path = Path(data_dir)
        samples_path = Path(samples_dir)
        
        file_processor = FileProcessor()
        
        # Scan files
        npy_files = file_processor.scan_files(data_path / split / 'images', 
                                            self.naming_manager.get_prefix('preprocessed'), 
                                            {'.npy'})
        sample_files = file_processor.scan_files(samples_path / split, 
                                                self.naming_manager.get_prefix('sample'))
        
        # Create mapping berdasarkan nominal dan uuid
        npy_mapping = {}
        for npy_file in npy_files:
            parsed = self.naming_manager.parse_filename(npy_file.name)
            if parsed:
                key = f"{parsed['nominal']}_{parsed['uuid']}"
                npy_mapping[key] = npy_file
        
        sample_mapping = {}
        for sample_file in sample_files:
            parsed = self.naming_manager.parse_filename(sample_file.name)
            if parsed:
                key = f"{parsed['nominal']}_{parsed['uuid']}"
                sample_mapping[key] = sample_file
        
        # Find discrepancies
        missing_samples = set(npy_mapping.keys()) - set(sample_mapping.keys())
        orphaned_samples = set(sample_mapping.keys()) - set(npy_mapping.keys())
        
        return {
            'consistent': len(missing_samples) == 0 and len(orphaned_samples) == 0,
            'total_npy_files': len(npy_files),
            'total_sample_files': len(sample_files),
            'missing_samples': len(missing_samples),
            'orphaned_samples': len(orphaned_samples),
            'missing_list': list(missing_samples)[:10],
            'orphaned_list': list(orphaned_samples)[:10]
        }