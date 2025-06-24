"""
File: smartcash/dataset/preprocessor/api/samples_api.py
Deskripsi: Updated samples API menggunakan FileNamingManager
"""

from typing import Dict, Any, List, Union, Optional, Callable
from pathlib import Path

from smartcash.common.logger import get_logger
from smartcash.common.utils.file_naming_manager import create_file_naming_manager
from ..config.defaults import MAIN_BANKNOTE_CLASSES
from ..utils.sample_generator import SampleGenerator
from ..utils.metadata_extractor import MetadataExtractor

def get_samples(data_dir: Union[str, Path],
               split: str = 'train',
               max_samples: int = 50,
               class_filter: List[int] = None) -> Dict[str, Any]:
    """🎲 Get samples menggunakan naming manager"""
    try:
        logger = get_logger(__name__)
        data_path = Path(data_dir)
        split_path = data_path / split
        
        if not split_path.exists():
            return {'success': False, 'message': f"❌ Split directory not found: {split_path}", 'samples': []}
        
        target_classes = class_filter or list(MAIN_BANKNOTE_CLASSES.keys())
        target_classes = [cid for cid in target_classes if cid in MAIN_BANKNOTE_CLASSES]
        
        if not target_classes:
            return {'success': False, 'message': "❌ No valid main banknote classes", 'samples': []}
        
        sample_generator = SampleGenerator()
        raw_samples = sample_generator.get_random_samples(data_dir, max_samples * len(target_classes), split)
        
        enhanced_samples = []
        class_counts = {cid: 0 for cid in target_classes}
        
        for sample in raw_samples:
            main_class_ids = [cid for cid in sample.get('class_ids', []) if cid in target_classes]
            
            if main_class_ids and class_counts[main_class_ids[0]] < max_samples:
                primary_class = main_class_ids[0]
                enhanced_sample = _enhance_sample_info(sample, primary_class, split_path)
                if enhanced_sample:
                    enhanced_samples.append(enhanced_sample)
                    class_counts[primary_class] += 1
        
        return {
            'success': True,
            'message': f"✅ Retrieved {len(enhanced_samples)} samples from {split}",
            'split': split,
            'total_samples': len(enhanced_samples),
            'by_class': class_counts,
            'samples': enhanced_samples
        }
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"❌ Get samples error: {str(e)}")
        return {'success': False, 'message': f"❌ Error: {str(e)}", 'samples': []}

def generate_sample_previews(data_dir: Union[str, Path],
                           output_dir: Union[str, Path],
                           splits: List[str] = None,
                           max_per_class: int = 5) -> Dict[str, Any]:
    """🖼️ Generate sample previews"""
    try:
        logger = get_logger(__name__)
        splits = splits or ['train', 'valid']
        
        sample_generator = SampleGenerator({'max_samples': max_per_class * len(MAIN_BANKNOTE_CLASSES)})
        results = sample_generator.generate_samples(data_dir, output_dir, splits, max_per_class)
        
        enhanced_results = {
            'success': True,
            'message': f"✅ Generated {results['total_generated']} sample previews",
            'output_dir': str(output_dir),
            'total_generated': results['total_generated'],
            'by_split': results['by_split'],
            'by_class': {},
            'samples': []
        }
        
        # Add class information
        for class_id, count in results['by_class'].items():
            if class_id in MAIN_BANKNOTE_CLASSES:
                enhanced_results['by_class'][str(class_id)] = {
                    'count': count,
                    'class_info': MAIN_BANKNOTE_CLASSES[class_id]
                }
        
        # Process sample information
        for sample in results['samples']:
            enhanced_sample = {
                'sample_path': sample['sample_path'],
                'filename': sample['filename'],
                'file_size_mb': sample['file_size_mb'],
                'npy_source': sample['npy_path'],
                'class_id': sample['main_class_id'],
                'class_info': MAIN_BANKNOTE_CLASSES.get(sample['main_class_id'], {}),
                'all_classes': sample.get('class_names', {})
            }
            enhanced_results['samples'].append(enhanced_sample)
        
        return enhanced_results
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"❌ Generate previews error: {str(e)}")
        return {'success': False, 'message': f"❌ Error: {str(e)}", 'total_generated': 0}

def get_class_samples(data_dir: Union[str, Path],
                     class_id: int,
                     split: str = 'train',
                     max_samples: int = 10) -> Dict[str, Any]:
    """🏷️ Get samples untuk specific class"""
    if class_id not in MAIN_BANKNOTE_CLASSES:
        return {'success': False, 'message': f"❌ Class {class_id} is not a main banknote class", 'samples': []}
    
    return get_samples(data_dir, split, max_samples, [class_id])

def _enhance_sample_info(sample: Dict[str, Any], primary_class: int, split_path: Path) -> Optional[Dict[str, Any]]:
    """🔧 Enhance sample info menggunakan naming manager"""
    try:
        from ..core.file_processor import FileProcessor
        
        naming_manager = create_file_naming_manager()
        fp = FileProcessor()
        npy_path = Path(sample['npy_path'])
        
        file_info = fp.get_file_info(npy_path)
        
        # Parse filename untuk metadata
        parsed = naming_manager.parse_filename(npy_path.name)
        if not parsed:
            return None
        
        # Generate corresponding label filename
        label_name = naming_manager.generate_corresponding_filename(npy_path.name, parsed['type'], '.txt')
        label_path = split_path / 'labels' / label_name
        
        class_info = MAIN_BANKNOTE_CLASSES[primary_class].copy()
        class_info['class_id'] = primary_class
        
        enhanced = {
            'npy_path': str(npy_path),
            'filename': npy_path.name,
            'file_size_mb': file_info.get('size_mb', 0),
            'class_ids': sample.get('class_ids', []),
            'class_names': sample.get('class_names', {}),
            'primary_class': {
                'class_id': primary_class,
                'nominal': class_info['nominal'],
                'display': class_info['display'],
                'value': class_info['value']
            },
            'uuid': parsed['uuid'],
            'nominal': parsed['nominal'],
            'has_labels': label_path.exists(),
            'label_path': str(label_path) if label_path.exists() else None
        }
        
        # Add denormalized image path
        if 'potential_sample_path' in sample:
            enhanced['denormalized_path'] = sample['potential_sample_path']
        
        return enhanced
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.debug(f"⚠️ Error enhancing sample: {str(e)}")
        return None

def get_samples_summary(data_dir: Union[str, Path],
                       splits: List[str] = None) -> Dict[str, Any]:
    """📊 Get samples summary"""
    try:
        from ..core.stats_collector import StatsCollector
        
        splits = splits or ['train', 'valid', 'test']
        stats_collector = StatsCollector()
        dataset_stats = stats_collector.collect_dataset_stats(data_dir, splits)
        
        main_class_distribution = {}
        for class_id, count in dataset_stats['class_distribution'].items():
            if class_id in MAIN_BANKNOTE_CLASSES:
                main_class_distribution[class_id] = {
                    'count': count,
                    'class_info': MAIN_BANKNOTE_CLASSES[class_id]
                }
        
        return {
            'success': True,
            'total_splits': dataset_stats['overview']['total_splits'],
            'total_preprocessed_files': dataset_stats['by_type']['preprocessed'],
            'main_class_distribution': main_class_distribution,
            'by_split': {
                split: {
                    'preprocessed_count': data['file_counts'].get('preprocessed', 0),
                    'main_classes': {
                        cid: count for cid, count in data['class_distribution'].items()
                        if cid in MAIN_BANKNOTE_CLASSES
                    }
                }
                for split, data in dataset_stats['by_split'].items()
            }
        }
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"❌ Samples summary error: {str(e)}")
        return {'success': False, 'message': f"❌ Error: {str(e)}"}