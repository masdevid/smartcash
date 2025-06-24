"""
File: smartcash/dataset/preprocessor/core/stats_collector.py
Deskripsi: Updated stats collector menggunakan FileNamingManager patterns
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Union
from collections import defaultdict

from smartcash.common.logger import get_logger
from smartcash.common.utils.file_naming_manager import create_file_naming_manager
from ..config.defaults import MAIN_BANKNOTE_CLASSES, LAYER_CLASSES

class StatsCollector:
    """📊 Updated stats collector dengan FileNamingManager integration"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = get_logger(__name__)
        self.naming_manager = create_file_naming_manager()
    
    def collect_dataset_stats(self, data_dir: Union[str, Path], 
                            splits: List[str] = None) -> Dict[str, Any]:
        """📈 Collect stats menggunakan naming manager"""
        splits = splits or ['train', 'valid', 'test']
        data_path = Path(data_dir)
        
        stats = {
            'overview': {'total_splits': 0, 'total_files': 0, 'total_size_mb': 0},
            'by_type': {file_type: 0 for file_type in self.naming_manager.get_supported_types()},
            'by_split': {},
            'class_distribution': defaultdict(int),
            'layer_distribution': defaultdict(int),
            'file_sizes': {'avg_image_mb': 0, 'avg_npy_mb': 0}
        }
        
        for split in splits:
            split_path = data_path / split
            if split_path.exists():
                split_stats = self._analyze_split(split_path)
                stats['by_split'][split] = split_stats
                stats['overview']['total_splits'] += 1
                
                # Aggregate counts
                for file_type, count in split_stats['file_counts'].items():
                    if file_type in stats['by_type']:
                        stats['by_type'][file_type] += count
                    stats['overview']['total_files'] += count
                
                # Aggregate distributions
                for class_id, count in split_stats['class_distribution'].items():
                    stats['class_distribution'][class_id] += count
                
                for layer, count in split_stats['layer_distribution'].items():
                    stats['layer_distribution'][layer] += count
                
                stats['overview']['total_size_mb'] += split_stats['total_size_mb']
        
        # Calculate averages
        if stats['overview']['total_files'] > 0:
            total_image_size = sum(
                split_data.get('avg_sizes', {}).get('images', 0) * split_data['file_counts'].get('raw', 0)
                for split_data in stats['by_split'].values()
            )
            total_npy_size = sum(
                split_data.get('avg_sizes', {}).get('npy', 0) * (
                    split_data['file_counts'].get('preprocessed', 0) + 
                    split_data['file_counts'].get('augmented', 0)
                )
                for split_data in stats['by_split'].values()
            )
            
            total_images = stats['by_type']['raw'] + stats['by_type']['sample']
            total_npy = stats['by_type']['preprocessed'] + stats['by_type']['augmented']
            
            stats['file_sizes']['avg_image_mb'] = round(total_image_size / max(total_images, 1), 2)
            stats['file_sizes']['avg_npy_mb'] = round(total_npy_size / max(total_npy, 1), 2)
        
        return stats
    
    def _analyze_split(self, split_path: Path) -> Dict[str, Any]:
        """📊 Analyze split menggunakan naming manager"""
        from .file_processor import FileProcessor
        
        fp = FileProcessor()
        images_dir = split_path / 'images'
        labels_dir = split_path / 'labels'
        
        # Scan files by type menggunakan naming manager
        file_counts = {}
        all_files = []
        
        for file_type in self.naming_manager.get_supported_types():
            if images_dir.exists():
                files = fp.scan_files_by_type(images_dir, file_type)
                file_counts[file_type] = len(files)
                all_files.extend(files)
            else:
                file_counts[file_type] = 0
        
        # Class dan layer distribution
        class_dist, layer_dist = self._analyze_labels(labels_dir)
        
        # File sizes
        avg_sizes = self._calculate_average_sizes({
            'images': [f for f in all_files if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}],
            'npy': [f for f in all_files if f.suffix.lower() == '.npy']
        })
        
        total_size = sum(fp.get_file_info(f).get('size_mb', 0) for f in all_files)
        
        return {
            'file_counts': file_counts,
            'class_distribution': class_dist,
            'layer_distribution': layer_dist,
            'avg_sizes': avg_sizes,
            'total_size_mb': round(total_size, 2)
        }
    
    def _analyze_labels(self, labels_dir: Path) -> tuple:
        """📋 Analyze labels untuk class/layer distribution"""
        class_dist = defaultdict(int)
        layer_dist = defaultdict(int)
        
        if not labels_dir.exists():
            return dict(class_dist), dict(layer_dist)
        
        from .file_processor import FileProcessor
        fp = FileProcessor()
        
        # Scan all label files
        label_files = fp.scan_files(labels_dir, extensions={'.txt'})
        
        for label_file in label_files:
            bboxes = fp.read_yolo_label(label_file)
            for bbox in bboxes:
                if len(bbox) >= 1:
                    class_id = int(bbox[0])
                    class_dist[class_id] += 1
                    
                    # Determine layer
                    for layer_name, class_range in LAYER_CLASSES.items():
                        if class_id in class_range:
                            layer_dist[layer_name] += 1
                            break
        
        return dict(class_dist), dict(layer_dist)
    
    def _calculate_average_sizes(self, file_groups: Dict[str, List[Path]]) -> Dict[str, float]:
        """📏 Calculate average file sizes by type"""
        from .file_processor import FileProcessor
        fp = FileProcessor()
        
        avg_sizes = {}
        for file_type, files in file_groups.items():
            if files:
                total_size = sum(fp.get_file_info(f).get('size_mb', 0) for f in files)
                avg_sizes[file_type] = round(total_size / len(files), 2)
            else:
                avg_sizes[file_type] = 0
        
        return avg_sizes
    
    def get_class_info(self, class_id: int) -> Dict[str, Any]:
        """🏷️ Get class information"""
        if class_id in MAIN_BANKNOTE_CLASSES:
            return MAIN_BANKNOTE_CLASSES[class_id]
        
        for layer_name, class_range in LAYER_CLASSES.items():
            if class_id in class_range:
                return {'layer': layer_name, 'class_id': class_id, 'display': f"{layer_name}_{class_id:02d}"}
        
        return {'class_id': class_id, 'display': f"unknown_{class_id}"}
    
    def get_layer_summary(self, class_distribution: Dict[int, int]) -> Dict[str, Any]:
        """📊 Summarize by layers"""
        layer_summary = {}
        
        for layer_name, class_range in LAYER_CLASSES.items():
            layer_count = sum(class_distribution.get(cid, 0) for cid in class_range)
            layer_classes = [cid for cid in class_range if class_distribution.get(cid, 0) > 0]
            
            layer_summary[layer_name] = {
                'total_objects': layer_count,
                'active_classes': len(layer_classes),
                'class_list': layer_classes
            }
        
        return layer_summary
    
    def export_stats_report(self, stats: Dict[str, Any], output_path: Union[str, Path]) -> bool:
        """📄 Export comprehensive stats report"""
        try:
            import json
            
            enhanced_stats = stats.copy()
            
            # Add class info
            enhanced_stats['class_info'] = {}
            for class_id in enhanced_stats['class_distribution'].keys():
                enhanced_stats['class_info'][class_id] = self.get_class_info(class_id)
            
            # Add layer summary
            enhanced_stats['layer_summary'] = self.get_layer_summary(enhanced_stats['class_distribution'])
            
            # Add naming manager info
            enhanced_stats['naming_manager_stats'] = self.naming_manager.get_nominal_statistics()
            
            with open(output_path, 'w') as f:
                json.dump(enhanced_stats, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Export error: {str(e)}")
            return False