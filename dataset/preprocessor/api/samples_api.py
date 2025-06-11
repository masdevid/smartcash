"""
File: smartcash/dataset/preprocessor/api/samples_api.py
Deskripsi: Samples management API untuk main banknotes layer
"""

from typing import Dict, Any, List, Union, Optional, Callable
from pathlib import Path

from smartcash.common.logger import get_logger
from ..config.defaults import MAIN_BANKNOTE_CLASSES
from ..utils.sample_generator import SampleGenerator
from ..utils.metadata_extractor import MetadataExtractor

def get_samples(data_dir: Union[str, Path],
               split: str = 'train',
               max_samples: int = 50,
               class_filter: List[int] = None) -> Dict[str, Any]:
    """🎲 Get samples dari main banknotes layer (7 classes)
    
    Args:
        data_dir: Directory berisi preprocessed data
        split: Target split ('train', 'valid', 'test')
        max_samples: Maximum samples per class
        class_filter: Filter specific class IDs (None = all main classes)
        
    Returns:
        Dict berisi sample information
        
    Example:
        >>> samples = get_samples('data/preprocessed', 'train', max_samples=10)
        >>> for sample in samples['samples']:
        >>>     print(f"Class: {sample['display_name']}, File: {sample['npy_path']}")
    """
    try:
        logger = get_logger(__name__)
        data_path = Path(data_dir)
        split_path = data_path / split
        
        if not split_path.exists():
            return {
                'success': False,
                'message': f"❌ Split directory not found: {split_path}",
                'samples': []
            }
        
        # Filter class IDs (only main banknotes)
        target_classes = class_filter or list(MAIN_BANKNOTE_CLASSES.keys())
        target_classes = [cid for cid in target_classes if cid in MAIN_BANKNOTE_CLASSES]
        
        if not target_classes:
            return {
                'success': False,
                'message': "❌ No valid main banknote classes specified",
                'samples': []
            }
        
        sample_generator = SampleGenerator()
        metadata_extractor = MetadataExtractor()
        
        # Get raw samples data
        raw_samples = sample_generator.get_random_samples(data_dir, max_samples * len(target_classes), split)
        
        # Filter dan enhance samples
        enhanced_samples = []
        class_counts = {cid: 0 for cid in target_classes}
        
        for sample in raw_samples:
            # Get main class ID dari sample
            main_class_ids = [cid for cid in sample.get('class_ids', []) if cid in target_classes]
            
            if main_class_ids:
                primary_class = main_class_ids[0]  # Take first main class
                
                # Check jika masih butuh samples untuk class ini
                if class_counts[primary_class] < max_samples:
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
        return {
            'success': False,
            'message': f"❌ Error: {str(e)}",
            'samples': []
        }

def generate_sample_previews(data_dir: Union[str, Path],
                           output_dir: Union[str, Path],
                           splits: List[str] = None,
                           max_per_class: int = 5) -> Dict[str, Any]:
    """🖼️ Generate denormalized sample images untuk preview
    
    Args:
        data_dir: Directory berisi preprocessed data
        output_dir: Output directory untuk sample images
        splits: Target splits (default: ['train', 'valid'])
        max_per_class: Maximum samples per class
        
    Returns:
        Dict dengan generation results
        
    Example:
        >>> result = generate_sample_previews('data/preprocessed', 'data/samples')
        >>> print(f"Generated {result['total_generated']} sample images")
    """
    try:
        logger = get_logger(__name__)
        splits = splits or ['train', 'valid']
        
        sample_generator = SampleGenerator({
            'max_samples': max_per_class * len(MAIN_BANKNOTE_CLASSES)
        })
        
        results = sample_generator.generate_samples(data_dir, output_dir, splits, max_per_class)
        
        # Enhance results dengan class information
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
        return {
            'success': False,
            'message': f"❌ Error: {str(e)}",
            'total_generated': 0
        }

def get_class_samples(data_dir: Union[str, Path],
                     class_id: int,
                     split: str = 'train',
                     max_samples: int = 10) -> Dict[str, Any]:
    """🏷️ Get samples untuk specific class
    
    Args:
        data_dir: Directory berisi preprocessed data
        class_id: Target class ID (must be main banknote class)
        split: Target split
        max_samples: Maximum samples
        
    Returns:
        Dict berisi class-specific samples
    """
    if class_id not in MAIN_BANKNOTE_CLASSES:
        return {
            'success': False,
            'message': f"❌ Class {class_id} is not a main banknote class",
            'samples': []
        }
    
    return get_samples(data_dir, split, max_samples, [class_id])

def _enhance_sample_info(sample: Dict[str, Any], primary_class: int, split_path: Path) -> Optional[Dict[str, Any]]:
    """🔧 Enhance sample information dengan additional metadata"""
    try:
        from ..core.file_processor import FileProcessor
        
        fp = FileProcessor()
        npy_path = Path(sample['npy_path'])
        
        # Get file info
        file_info = fp.get_file_info(npy_path)
        
        # Get corresponding label path
        label_path = split_path / 'labels' / f"{npy_path.stem}.txt"
        
        # Get class information
        class_info = MAIN_BANKNOTE_CLASSES[primary_class].copy()
        class_info['class_id'] = primary_class
        
        # Get all class names untuk multi-class objects
        all_class_names = sample.get('class_names', {})
        
        enhanced = {
            'npy_path': str(npy_path),
            'filename': npy_path.name,
            'file_size_mb': file_info.get('size_mb', 0),
            'class_ids': sample.get('class_ids', []),
            'class_names': all_class_names,
            'primary_class': {
                'class_id': primary_class,
                'nominal': class_info['nominal'],
                'display': class_info['display'],
                'value': class_info['value']
            },
            'uuid': sample.get('uuid', 'unknown'),
            'has_labels': label_path.exists(),
            'label_path': str(label_path) if label_path.exists() else None
        }
        
        # Add denormalized image path jika ada
        if 'potential_sample_path' in sample:
            enhanced['denormalized_path'] = sample['potential_sample_path']
        
        return enhanced
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.debug(f"⚠️ Error enhancing sample: {str(e)}")
        return None

def get_samples_summary(data_dir: Union[str, Path],
                       splits: List[str] = None) -> Dict[str, Any]:
    """📊 Get summary of available samples
    
    Args:
        data_dir: Directory berisi preprocessed data
        splits: Target splits untuk analysis
        
    Returns:
        Dict dengan sample statistics
    """
    try:
        from ..core.stats_collector import StatsCollector
        
        splits = splits or ['train', 'valid', 'test']
        stats_collector = StatsCollector()
        
        # Collect dataset stats
        dataset_stats = stats_collector.collect_dataset_stats(data_dir, splits)
        
        # Filter hanya main banknote classes
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
        return {
            'success': False,
            'message': f"❌ Error: {str(e)}"
        }