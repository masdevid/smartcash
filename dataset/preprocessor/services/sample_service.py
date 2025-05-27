"""
File: smartcash/dataset/preprocessor/services/sample_service.py
Deskripsi: Service untuk generate samples raw → preprocessed dengan label classes untuk research visualization
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import cv2

from smartcash.common.logger import get_logger
from smartcash.dataset.preprocessor.processors.image_processor import ImageProcessor


class PreprocessingSampleService:
    """🎯 Service untuk generate samples dan comparison raw → preprocessed untuk research."""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        """Initialize sample service dengan configuration."""
        self.config = config
        self.logger = logger or get_logger()
        self.image_processor = ImageProcessor(config, logger)
        
    def get_preprocessing_samples(self, raw_dir: str, preprocessed_dir: str, 
                                splits: List[str] = None, sample_count: int = 5) -> Dict[str, Any]:
        """🎯 Generate comparison samples raw → preprocessed dengan label classes."""
        splits = splits or ['train', 'valid', 'test']
        
        raw_path = Path(raw_dir)
        prep_path = Path(preprocessed_dir)
        
        if not raw_path.exists():
            return {'status': 'error', 'message': f'Raw directory tidak ditemukan: {raw_dir}'}
        
        samples = {}
        for split in splits:
            split_samples = self._get_split_samples(raw_path / split, prep_path / split, sample_count)
            samples[split] = split_samples
        
        # Generate summary statistics
        summary = self._generate_sample_summary(samples)
        
        return {
            'status': 'success',
            'samples': samples,
            'summary': summary,
            'message': f'Generated samples untuk {len(splits)} splits'
        }
    
    def _get_split_samples(self, raw_split_dir: Path, prep_split_dir: Path, sample_count: int) -> List[Dict[str, Any]]:
        """Get samples untuk single split dengan before/after comparison."""
        if not raw_split_dir.exists():
            return []
        
        # Find raw images
        raw_images_dir = raw_split_dir / 'images' if (raw_split_dir / 'images').exists() else raw_split_dir
        raw_images = [f for f in raw_images_dir.glob('*.*') 
                     if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']][:sample_count]
        
        samples = []
        for raw_img in raw_images:
            sample_info = self._create_sample_info(raw_img, raw_split_dir, prep_split_dir)
            if sample_info:
                samples.append(sample_info)
        
        return samples
    
    def _create_sample_info(self, raw_img_path: Path, raw_split_dir: Path, prep_split_dir: Path) -> Optional[Dict[str, Any]]:
        """Create comprehensive sample info dengan before/after data."""
        try:
            original_name = raw_img_path.stem
            expected_prep_name = f"raw_{original_name}"
            
            # Paths
            raw_label_path = self._find_label_path(raw_img_path, raw_split_dir)
            prep_img_path = prep_split_dir / 'images' / f"{expected_prep_name}.jpg"
            prep_label_path = prep_split_dir / 'labels' / f"{expected_prep_name}.txt"
            
            # Extract label classes
            raw_classes = self._extract_classes(raw_label_path) if raw_label_path and raw_label_path.exists() else []
            prep_classes = self._extract_classes(prep_label_path) if prep_label_path.exists() else []
            
            # Image dimensions
            raw_dims = self._get_image_dimensions(raw_img_path)
            prep_dims = self._get_image_dimensions(prep_img_path) if prep_img_path.exists() else None
            
            return {
                'original_name': original_name,
                'preprocessed_name': expected_prep_name,
                'raw': {
                    'image_path': str(raw_img_path),
                    'label_path': str(raw_label_path) if raw_label_path else None,
                    'classes': raw_classes,
                    'dimensions': raw_dims,
                    'file_size_mb': round(raw_img_path.stat().st_size / (1024*1024), 2),
                    'exists': True
                },
                'preprocessed': {
                    'image_path': str(prep_img_path),
                    'label_path': str(prep_label_path),
                    'classes': prep_classes,
                    'dimensions': prep_dims,
                    'file_size_mb': round(prep_img_path.stat().st_size / (1024*1024), 2) if prep_img_path.exists() else 0,
                    'exists': prep_img_path.exists()
                },
                'comparison': {
                    'processed': prep_img_path.exists(),
                    'classes_match': raw_classes == prep_classes,
                    'dimension_changed': raw_dims != prep_dims if prep_dims else True,
                    'size_reduction_percent': self._calculate_size_reduction(raw_img_path, prep_img_path) if prep_img_path.exists() else 0
                }
            }
            
        except Exception as e:
            self.logger.debug(f"🔧 Sample creation error: {str(e)}")
            return None
    
    def _find_label_path(self, img_path: Path, split_dir: Path) -> Optional[Path]:
        """Find corresponding label file dengan multiple fallback strategies."""
        label_name = f"{img_path.stem}.txt"
        
        # Strategy 1: labels subdirectory
        labels_dir = split_dir / 'labels'
        if labels_dir.exists():
            label_path = labels_dir / label_name
            if label_path.exists():
                return label_path
        
        # Strategy 2: same directory as image
        label_path = img_path.parent / label_name
        if label_path.exists():
            return label_path
        
        # Strategy 3: parent directory
        label_path = split_dir / label_name
        if label_path.exists():
            return label_path
        
        return None
    
    def _extract_classes(self, label_path: Path) -> List[Dict[str, Any]]:
        """Extract classes dan bbox info dari label file."""
        if not label_path or not label_path.exists():
            return []
        
        classes = []
        try:
            with open(label_path, 'r') as f:
                for line_idx, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(float(parts[0]))
                        bbox = [float(x) for x in parts[1:5]]
                        classes.append({
                            'class_id': class_id,
                            'bbox': bbox,
                            'line': line_idx + 1
                        })
        except Exception as e:
            self.logger.debug(f"🔧 Class extraction error: {str(e)}")
        
        return classes
    
    def _get_image_dimensions(self, img_path: Path) -> Optional[Tuple[int, int]]:
        """Get image dimensions (width, height)."""
        try:
            img = cv2.imread(str(img_path))
            if img is not None:
                h, w = img.shape[:2]
                return (w, h)
        except Exception:
            pass
        return None
    
    def _calculate_size_reduction(self, raw_path: Path, prep_path: Path) -> float:
        """Calculate size reduction percentage."""
        try:
            raw_size = raw_path.stat().st_size
            prep_size = prep_path.stat().st_size
            if raw_size > 0:
                return round(((raw_size - prep_size) / raw_size) * 100, 1)
        except Exception:
            pass
        return 0.0
    
    def _generate_sample_summary(self, samples: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Generate summary statistics untuk samples."""
        total_samples = sum(len(split_samples) for split_samples in samples.values())
        processed_count = sum(
            sum(1 for sample in split_samples if sample['comparison']['processed'])
            for split_samples in samples.values()
        )
        
        # Class distribution
        all_classes = []
        for split_samples in samples.values():
            for sample in split_samples:
                all_classes.extend([cls['class_id'] for cls in sample['raw']['classes']])
        
        class_counts = {}
        for cls_id in all_classes:
            class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
        
        return {
            'total_samples': total_samples,
            'processed_count': processed_count,
            'processing_rate': round((processed_count / total_samples) * 100, 1) if total_samples > 0 else 0,
            'splits_with_data': len([s for s in samples.values() if s]),
            'class_distribution': class_counts,
            'unique_classes': len(class_counts)
        }
    
    def get_sample_visualization_data(self, samples_result: Dict[str, Any]) -> Dict[str, Any]:
        """🎯 Prepare data untuk visualization di UI research."""
        if samples_result.get('status') != 'success':
            return samples_result
        
        samples = samples_result['samples']
        viz_data = {
            'comparison_pairs': [],
            'class_statistics': {},
            'dimension_analysis': {},
            'file_size_analysis': {}
        }
        
        # Prepare comparison pairs untuk visualization
        for split, split_samples in samples.items():
            for sample in split_samples:
                if sample['comparison']['processed']:
                    viz_data['comparison_pairs'].append({
                        'split': split,
                        'name': sample['original_name'],
                        'raw_path': sample['raw']['image_path'],
                        'prep_path': sample['preprocessed']['image_path'],
                        'classes': sample['raw']['classes']
                    })
        
        # Class statistics
        all_classes = []
        for split_samples in samples.values():
            for sample in split_samples:
                all_classes.extend([cls['class_id'] for cls in sample['raw']['classes']])
        
        viz_data['class_statistics'] = {
            'distribution': dict([(str(k), v) for k, v in samples_result['summary']['class_distribution'].items()]),
            'total_detections': len(all_classes),
            'unique_classes': samples_result['summary']['unique_classes']
        }
        
        return {
            'status': 'success',
            'visualization_data': viz_data,
            'summary': samples_result['summary']
        }

# One-liner factories dan utilities
create_sample_service = lambda config, logger=None: PreprocessingSampleService(config, logger)
get_preprocessing_samples = lambda config, raw_dir, prep_dir, splits=None: PreprocessingSampleService(config).get_preprocessing_samples(raw_dir, prep_dir, splits)
extract_sample_classes = lambda samples: [cls for split in samples.values() for sample in split for cls in sample['raw']['classes']]