"""
File: smartcash/dataset/preprocessor/api/stats_api.py
Deskripsi: Statistics API untuk dataset analysis
"""

from typing import Dict, Any, List, Union
from pathlib import Path

from smartcash.common.logger import get_logger
from ..core.stats_collector import StatsCollector
from ..config.defaults import MAIN_BANKNOTE_CLASSES, LAYER_CLASSES

def get_dataset_stats(data_dir: Union[str, Path],
                     splits: List[str] = None,
                     include_details: bool = True) -> Dict[str, Any]:
    """📊 Get comprehensive dataset statistics
    
    Args:
        data_dir: Base data directory
        splits: Target splits untuk analysis
        include_details: Include detailed breakdown
        
    Returns:
        Dict dengan comprehensive statistics
        
    Example:
        >>> stats = get_dataset_stats('data')
        >>> print(f"Total files: {stats['overview']['total_files']}")
        >>> print(f"Main banknotes: {stats['main_banknotes']['total_objects']}")
    """
    try:
        logger = get_logger(__name__)
        splits = splits or ['train', 'valid', 'test']
        
        stats_collector = StatsCollector()
        raw_stats = stats_collector.collect_dataset_stats(data_dir, splits)
        
        # Enhance dengan main banknotes analysis
        enhanced_stats = {
            'success': True,
            'data_dir': str(data_dir),
            'analyzed_splits': splits,
            'overview': raw_stats['overview'],
            'file_types': {
                'raw_images': raw_stats['by_type']['raw'],
                'preprocessed_npy': raw_stats['by_type']['preprocessed'],
                'augmented_npy': raw_stats['by_type']['augmented'],
                'sample_images': raw_stats['by_type']['samples']
            },
            'main_banknotes': _analyze_main_banknotes(raw_stats['class_distribution']),
            'layers': _analyze_layers(raw_stats['class_distribution']),
            'file_sizes': raw_stats['file_sizes']
        }
        
        if include_details:
            enhanced_stats['by_split'] = raw_stats['by_split']
            enhanced_stats['detailed_class_distribution'] = _get_detailed_class_info(raw_stats['class_distribution'])
        
        return enhanced_stats
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"❌ Dataset stats error: {str(e)}")
        return {
            'success': False,
            'message': f"❌ Error: {str(e)}"
        }

def get_file_stats(directory: Union[str, Path],
                  file_type: str = 'all') -> Dict[str, Any]:
    """📁 Get file-specific statistics
    
    Args:
        directory: Target directory
        file_type: File type filter ('raw', 'preprocessed', 'augmented', 'samples', 'all')
        
    Returns:
        Dict dengan file statistics
    """
    try:
        from ..core.file_processor import FileProcessor
        
        dir_path = Path(directory)
        if not dir_path.exists():
            return {
                'success': False,
                'message': f"❌ Directory not found: {directory}"
            }
        
        fp = FileProcessor()
        
        # Scan files berdasarkan type
        if file_type == 'all':
            file_groups = fp.scan_preprocessed_files(dir_path)
        else:
            extensions = {'.npy'} if file_type in ['preprocessed', 'augmented'] else None
            prefix = {'raw': 'rp_', 'preprocessed': 'pre_', 'augmented': 'aug_', 'samples': 'sample_'}.get(file_type, '')
            files = fp.scan_files(dir_path, prefix, extensions)
            file_groups = {file_type: files}
        
        # Calculate statistics
        stats = {
            'success': True,
            'directory': str(directory),
            'file_type': file_type,
            'total_files': sum(len(files) for files in file_groups.values()),
            'by_type': {},
            'total_size_mb': 0
        }
        
        for ftype, files in file_groups.items():
            if files:
                # Calculate size stats
                sizes = [fp.get_file_info(f).get('size_mb', 0) for f in files]
                total_size = sum(sizes)
                avg_size = total_size / len(sizes) if sizes else 0
                
                stats['by_type'][ftype] = {
                    'count': len(files),
                    'total_size_mb': round(total_size, 2),
                    'avg_size_mb': round(avg_size, 2),
                    'min_size_mb': round(min(sizes), 2) if sizes else 0,
                    'max_size_mb': round(max(sizes), 2) if sizes else 0
                }
                
                stats['total_size_mb'] += total_size
        
        stats['total_size_mb'] = round(stats['total_size_mb'], 2)
        
        return stats
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"❌ File stats error: {str(e)}")
        return {
            'success': False,
            'message': f"❌ Error: {str(e)}"
        }

def get_class_distribution_stats(data_dir: Union[str, Path],
                               splits: List[str] = None) -> Dict[str, Any]:
    """🏷️ Get detailed class distribution statistics
    
    Args:
        data_dir: Base data directory
        splits: Target splits
        
    Returns:
        Dict dengan class distribution analysis
    """
    try:
        stats_collector = StatsCollector()
        dataset_stats = stats_collector.collect_dataset_stats(data_dir, splits or ['train', 'valid', 'test'])
        
        class_dist = dataset_stats['class_distribution']
        
        # Analyze by layers
        layer_analysis = {}
        for layer_name, class_range in LAYER_CLASSES.items():
            layer_classes = {cid: count for cid, count in class_dist.items() if cid in class_range}
            layer_total = sum(layer_classes.values())
            
            layer_analysis[layer_name] = {
                'total_objects': layer_total,
                'active_classes': len(layer_classes),
                'class_distribution': layer_classes,
                'avg_objects_per_class': round(layer_total / max(len(layer_classes), 1), 1)
            }
        
        # Main banknotes detail
        main_banknotes = {}
        for class_id in MAIN_BANKNOTE_CLASSES:
            count = class_dist.get(class_id, 0)
            main_banknotes[class_id] = {
                'count': count,
                'percentage': round((count / max(sum(class_dist.values()), 1)) * 100, 1),
                'class_info': MAIN_BANKNOTE_CLASSES[class_id]
            }
        
        return {
            'success': True,
            'total_objects': sum(class_dist.values()),
            'total_classes': len(class_dist),
            'by_layer': layer_analysis,
            'main_banknotes': main_banknotes,
            'class_balance': _analyze_class_balance(class_dist)
        }
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"❌ Class distribution error: {str(e)}")
        return {
            'success': False,
            'message': f"❌ Error: {str(e)}"
        }

def export_stats_report(data_dir: Union[str, Path],
                       output_path: Union[str, Path],
                       splits: List[str] = None) -> Dict[str, Any]:
    """📄 Export comprehensive statistics report
    
    Args:
        data_dir: Base data directory
        output_path: Output JSON file path
        splits: Target splits
        
    Returns:
        Dict dengan export results
    """
    try:
        # Get comprehensive stats
        dataset_stats = get_dataset_stats(data_dir, splits, include_details=True)
        
        if not dataset_stats['success']:
            return dataset_stats
        
        # Add class distribution details
        class_dist_stats = get_class_distribution_stats(data_dir, splits)
        if class_dist_stats['success']:
            dataset_stats['class_analysis'] = class_dist_stats
        
        # Export report
        stats_collector = StatsCollector()
        success = stats_collector.export_stats_report(dataset_stats, output_path)
        
        return {
            'success': success,
            'output_path': str(output_path),
            'message': f"✅ Report exported to {output_path}" if success else "❌ Export failed"
        }
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"❌ Export report error: {str(e)}")
        return {
            'success': False,
            'message': f"❌ Error: {str(e)}"
        }

# === Helper functions ===

def _analyze_main_banknotes(class_distribution: Dict[int, int]) -> Dict[str, Any]:
    """🏦 Analyze main banknotes distribution"""
    main_classes = {cid: count for cid, count in class_distribution.items() if cid in MAIN_BANKNOTE_CLASSES}
    total_main = sum(main_classes.values())
    
    return {
        'total_objects': total_main,
        'active_classes': len(main_classes),
        'class_count': {cid: count for cid, count in main_classes.items()},
        'percentage_of_total': round((total_main / max(sum(class_distribution.values()), 1)) * 100, 1)
    }

def _analyze_layers(class_distribution: Dict[int, int]) -> Dict[str, Any]:
    """📊 Analyze layer distribution"""
    layer_stats = {}
    
    for layer_name, class_range in LAYER_CLASSES.items():
        layer_count = sum(class_distribution.get(cid, 0) for cid in class_range)
        layer_stats[layer_name] = {
            'total_objects': layer_count,
            'active_classes': len([cid for cid in class_range if class_distribution.get(cid, 0) > 0])
        }
    
    return layer_stats

def _get_detailed_class_info(class_distribution: Dict[int, int]) -> Dict[str, Any]:
    """📋 Get detailed class information"""
    detailed = {}
    
    for class_id, count in class_distribution.items():
        if class_id in MAIN_BANKNOTE_CLASSES:
            detailed[str(class_id)] = {
                'count': count,
                'class_info': MAIN_BANKNOTE_CLASSES[class_id],
                'layer': 'l1_main'
            }
        else:
            # Find layer
            layer = 'unknown'
            for layer_name, class_range in LAYER_CLASSES.items():
                if class_id in class_range:
                    layer = layer_name
                    break
            
            detailed[str(class_id)] = {
                'count': count,
                'class_id': class_id,
                'layer': layer
            }
    
    return detailed

def _analyze_class_balance(class_distribution: Dict[int, int]) -> Dict[str, Any]:
    """⚖️ Analyze class balance"""
    if not class_distribution:
        return {'balanced': True, 'imbalance_ratio': 1.0}
    
    counts = list(class_distribution.values())
    min_count = min(counts)
    max_count = max(counts)
    
    imbalance_ratio = max_count / max(min_count, 1)
    balanced = imbalance_ratio <= 3.0  # Threshold untuk balanced
    
    return {
        'balanced': balanced,
        'imbalance_ratio': round(imbalance_ratio, 2),
        'min_count': min_count,
        'max_count': max_count,
        'recommendation': 'Consider data augmentation for minority classes' if not balanced else 'Well balanced'
    }