"""
File: smartcash/dataset/preprocessor/utils/sample_generator.py
Deskripsi: Sample generation untuk visualization dan preview
"""

import cv2
import numpy as np
import random
from pathlib import Path
from typing import Dict, Any, List, Union, Optional

from smartcash.common.logger import get_logger
from ..config.defaults import MAIN_BANKNOTE_CLASSES

class SampleGenerator:
    """🎲 Generate samples untuk visualization dan analysis"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = get_logger(__name__)
        
        # Sample settings
        self.max_samples = self.config.get('max_samples', 50)
        self.sample_size = self.config.get('sample_size', (640, 640))
        self.quality = self.config.get('jpg_quality', 85)
    
    def generate_samples(self, data_dir: Union[str, Path], 
                        output_dir: Union[str, Path],
                        splits: List[str] = None,
                        max_per_class: int = 5) -> Dict[str, Any]:
        """🎲 Generate samples dari preprocessed data"""
        data_path = Path(data_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        splits = splits or ['train', 'valid']
        
        results = {
            'total_generated': 0,
            'by_split': {},
            'by_class': {},
            'samples': []
        }
        
        for split in splits:
            split_results = self._generate_split_samples(
                data_path / split, output_path / split, max_per_class
            )
            
            results['by_split'][split] = split_results
            results['total_generated'] += split_results['count']
            
            # Aggregate by class
            for class_id, count in split_results['by_class'].items():
                results['by_class'][class_id] = results['by_class'].get(class_id, 0) + count
            
            results['samples'].extend(split_results['samples'])
        
        return results
    
    def _generate_split_samples(self, split_dir: Path, output_dir: Path, 
                              max_per_class: int) -> Dict[str, Any]:
        """🎯 Generate samples untuk single split"""
        from .metadata_extractor import MetadataExtractor
        from ..core.file_processor import FileProcessor
        
        metadata_extractor = MetadataExtractor()
        file_processor = FileProcessor()
        
        # Scan preprocessed .npy files
        npy_files = file_processor.scan_files(split_dir / 'images', 'pre_', {'.npy'})
        
        if not npy_files:
            return {'count': 0, 'by_class': {}, 'samples': []}
        
        # Group by class
        class_groups = {}
        for npy_file in npy_files:
            metadata = metadata_extractor.extract_file_metadata(npy_file)
            
            if metadata.get('research_format') and metadata['components'].get('class_info'):
                class_id = metadata['components']['class_info'].get('class_id', -1)
                
                # Only main banknote classes
                if class_id in MAIN_BANKNOTE_CLASSES:
                    if class_id not in class_groups:
                        class_groups[class_id] = []
                    class_groups[class_id].append(npy_file)
        
        # Sample dari each class
        output_dir.mkdir(parents=True, exist_ok=True)
        samples = []
        by_class = {}
        
        for class_id, files in class_groups.items():
            # Random sample
            sample_count = min(max_per_class, len(files))
            sampled_files = random.sample(files, sample_count)
            by_class[class_id] = sample_count
            
            for npy_file in sampled_files:
                sample_info = self._create_sample_from_npy(npy_file, output_dir, class_id)
                if sample_info:
                    samples.append(sample_info)
        
        return {
            'count': len(samples),
            'by_class': by_class,
            'samples': samples
        }
    
    def _create_sample_from_npy(self, npy_file: Path, output_dir: Path, 
                              class_id: int) -> Optional[Dict[str, Any]]:
        """🖼️ Create denormalized sample dari .npy file"""
        try:
            from ..core.file_processor import FileProcessor
            from ..core.normalizer import YOLONormalizer
            
            file_processor = FileProcessor()
            
            # Load normalized array dan metadata
            normalized_array, metadata = file_processor.load_normalized_array(npy_file)
            if normalized_array is None:
                return None
            
            # Denormalize untuk visualization
            normalizer = YOLONormalizer()
            if metadata:
                denormalized = normalizer.denormalize(normalized_array, metadata)
            else:
                # Fallback denormalization
                denormalized = (normalized_array * 255).astype(np.uint8)
                if len(denormalized.shape) == 3:
                    denormalized = denormalized[:, :, :3]  # Remove alpha channel jika ada
            
            # Generate sample filename
            sample_name = f"sample_{MAIN_BANKNOTE_CLASSES[class_id]['nominal']}_{npy_file.stem.split('_')[-2]}.jpg"
            sample_path = output_dir / sample_name
            
            # Save denormalized image
            success = cv2.imwrite(str(sample_path), cv2.cvtColor(denormalized, cv2.COLOR_RGB2BGR))
            
            if success:
                # Get file info
                file_info = file_processor.get_file_info(sample_path)
                
                # Find corresponding label
                label_path = npy_file.parent.parent / 'labels' / f"{npy_file.stem}.txt"
                class_ids = []
                class_names = {}
                
                if label_path.exists():
                    from .metadata_extractor import MetadataExtractor
                    metadata_extractor = MetadataExtractor()
                    class_ids = metadata_extractor.get_class_ids_from_labels(label_path)
                    class_names = metadata_extractor.get_class_names_from_ids(class_ids)
                
                return {
                    'npy_path': str(npy_file),
                    'sample_path': str(sample_path),
                    'filename': sample_name,
                    'file_size_mb': file_info.get('size_mb', 0),
                    'class_ids': class_ids,
                    'class_names': class_names,
                    'main_class_id': class_id,
                    'nominal': MAIN_BANKNOTE_CLASSES[class_id]['nominal'],
                    'display_name': MAIN_BANKNOTE_CLASSES[class_id]['display'],
                    'uuid': npy_file.stem.split('_')[2] if '_' in npy_file.stem else 'unknown'
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Sample creation error {npy_file.name}: {str(e)}")
            return None
    
    def get_random_samples(self, data_dir: Union[str, Path], 
                          count: int = 10, 
                          split: str = 'train') -> List[Dict[str, Any]]:
        """🎲 Get random samples tanpa generate files"""
        from .metadata_extractor import MetadataExtractor
        from ..core.file_processor import FileProcessor
        
        data_path = Path(data_dir)
        split_path = data_path / split
        
        if not split_path.exists():
            return []
        
        metadata_extractor = MetadataExtractor()
        file_processor = FileProcessor()
        
        # Scan preprocessed files
        npy_files = file_processor.scan_files(split_path / 'images', 'pre_', {'.npy'})
        
        if not npy_files:
            return []
        
        # Filter main banknote classes only
        main_banknote_files = []
        for npy_file in npy_files:
            metadata = metadata_extractor.extract_file_metadata(npy_file)
            if (metadata.get('research_format') and 
                metadata['components'].get('class_info', {}).get('class_id', -1) in MAIN_BANKNOTE_CLASSES):
                main_banknote_files.append(npy_file)
        
        # Random sample
        sample_count = min(count, len(main_banknote_files))
        sampled_files = random.sample(main_banknote_files, sample_count)
        
        # Get metadata untuk each sample
        samples = []
        for npy_file in sampled_files:
            metadata = metadata_extractor.extract_sample_metadata(npy_file)
            if metadata.get('is_main_banknote'):
                # Get label info
                label_path = npy_file.parent.parent / 'labels' / f"{npy_file.stem}.txt"
                
                sample_info = {
                    'npy_path': str(npy_file),
                    'filename': npy_file.name,
                    'file_size_mb': metadata.get('size_mb', 0),
                    'class_ids': metadata.get('main_class_ids', []),
                    'class_names': metadata.get('class_names', {}),
                    'uuid': metadata.get('components', {}).get('uuid', 'unknown')
                }
                
                # Potential denormalized path
                if metadata.get('components'):
                    nominal = metadata['components'].get('nominal', '000000')
                    uuid_part = metadata['components'].get('uuid', 'unknown')[:8]
                    sample_info['potential_sample_path'] = f"sample_{nominal}_{uuid_part}.jpg"
                
                samples.append(sample_info)
        
        return samples