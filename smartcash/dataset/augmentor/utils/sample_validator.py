"""
File: smartcash/dataset/augmentor/utils/sample_validator.py
Description: Validation utilities for augmented samples to filter out invalid samples
"""

from typing import List, Tuple, Dict, Any
from smartcash.common.logger import get_logger

logger = get_logger(__name__)


class AugmentedSampleValidator:
    """🔍 Validator to filter out augmented samples with no valid bounding boxes or labels"""
    
    def __init__(self, min_bbox_size: float = 0.001, min_valid_boxes: int = 1):
        """
        Initialize validator with filtering criteria.
        
        Args:
            min_bbox_size: Minimum width/height for valid bounding boxes (default: 0.001)
            min_valid_boxes: Minimum number of valid bounding boxes required (default: 1)
        """
        self.min_bbox_size = min_bbox_size
        self.min_valid_boxes = min_valid_boxes
        self.logger = logger
    
    def validate_augmented_sample(self, bboxes: List, class_labels: List) -> Tuple[bool, List, List, Dict[str, Any]]:
        """
        Validate an augmented sample and filter out invalid bounding boxes.
        
        Args:
            bboxes: List of bounding boxes in YOLO format [x_center, y_center, width, height]
            class_labels: List of class labels corresponding to bounding boxes
            
        Returns:
            Tuple of (is_valid, valid_bboxes, valid_class_labels, validation_info)
        """
        if not bboxes or not class_labels:
            return False, [], [], {
                'reason': 'empty_input',
                'original_count': 0,
                'valid_count': 0,
                'filtered_count': 0
            }
        
        if len(bboxes) != len(class_labels):
            return False, [], [], {
                'reason': 'mismatched_lengths',
                'bbox_count': len(bboxes),
                'label_count': len(class_labels),
                'valid_count': 0,
                'filtered_count': 0
            }
        
        valid_bboxes = []
        valid_class_labels = []
        filtered_reasons = {
            'too_small': 0,
            'invalid_coords': 0,
            'out_of_bounds': 0
        }
        
        # Filter bounding boxes
        for bbox, class_label in zip(bboxes, class_labels):
            try:
                # Convert to float and validate
                x, y, w, h = [float(coord) for coord in bbox[:4]]
                
                # Check if coordinates are valid numbers
                if not all(isinstance(coord, (int, float)) and not (coord != coord) for coord in [x, y, w, h]):  # Check for NaN
                    filtered_reasons['invalid_coords'] += 1
                    continue
                
                # Normalize coordinates to [0, 1] range
                x = max(0.0, min(1.0, x))
                y = max(0.0, min(1.0, y))
                w = max(0.0, min(1.0, w))
                h = max(0.0, min(1.0, h))
                
                # Check minimum size requirements
                if w < self.min_bbox_size or h < self.min_bbox_size:
                    filtered_reasons['too_small'] += 1
                    continue
                
                # Check if bbox is within valid bounds (with some tolerance)
                if (x - w/2 < -0.01 or x + w/2 > 1.01 or 
                    y - h/2 < -0.01 or y + h/2 > 1.01):
                    filtered_reasons['out_of_bounds'] += 1
                    continue
                
                # If we reach here, the bounding box is valid
                valid_bboxes.append([x, y, w, h])
                valid_class_labels.append(int(class_label))
                
            except (ValueError, IndexError, TypeError) as e:
                filtered_reasons['invalid_coords'] += 1
                continue
        
        # Check if we have enough valid bounding boxes
        is_valid = len(valid_bboxes) >= self.min_valid_boxes
        
        validation_info = {
            'reason': 'valid' if is_valid else 'insufficient_valid_boxes',
            'original_count': len(bboxes),
            'valid_count': len(valid_bboxes),
            'filtered_count': len(bboxes) - len(valid_bboxes),
            'filter_reasons': filtered_reasons,
            'min_required': self.min_valid_boxes
        }
        
        return is_valid, valid_bboxes, valid_class_labels, validation_info
    
    def get_validation_summary(self, validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary of validation results for reporting.
        
        Args:
            validation_results: List of validation_info dictionaries
            
        Returns:
            Summary dictionary with validation statistics
        """
        if not validation_results:
            return {'total_samples': 0, 'valid_samples': 0, 'invalid_samples': 0}
        
        total_samples = len(validation_results)
        valid_samples = sum(1 for result in validation_results if result['reason'] == 'valid')
        invalid_samples = total_samples - valid_samples
        
        # Aggregate filter reasons
        aggregated_reasons = {
            'too_small': 0,
            'invalid_coords': 0,
            'out_of_bounds': 0,
            'empty_input': 0,
            'mismatched_lengths': 0,
            'insufficient_valid_boxes': 0
        }
        
        total_original_boxes = 0
        total_valid_boxes = 0
        total_filtered_boxes = 0
        
        for result in validation_results:
            reason = result.get('reason', 'unknown')
            if reason in aggregated_reasons:
                aggregated_reasons[reason] += 1
            
            total_original_boxes += result.get('original_count', 0)
            total_valid_boxes += result.get('valid_count', 0)
            total_filtered_boxes += result.get('filtered_count', 0)
            
            # Aggregate individual filter reasons
            filter_reasons = result.get('filter_reasons', {})
            for filter_type, count in filter_reasons.items():
                if filter_type in aggregated_reasons:
                    aggregated_reasons[filter_type] += count
        
        summary = {
            'total_samples': total_samples,
            'valid_samples': valid_samples,
            'invalid_samples': invalid_samples,
            'valid_rate': (valid_samples / total_samples * 100) if total_samples > 0 else 0,
            'box_statistics': {
                'total_original_boxes': total_original_boxes,
                'total_valid_boxes': total_valid_boxes,
                'total_filtered_boxes': total_filtered_boxes,
                'box_retention_rate': (total_valid_boxes / total_original_boxes * 100) if total_original_boxes > 0 else 0
            },
            'filter_reasons': aggregated_reasons,
            'validation_config': {
                'min_bbox_size': self.min_bbox_size,
                'min_valid_boxes': self.min_valid_boxes
            }
        }
        
        return summary
    
    def log_validation_summary(self, summary: Dict[str, Any], target_split: str = ''):
        """
        Log validation summary with clear statistics.
        
        Args:
            summary: Summary dictionary from get_validation_summary()
            target_split: Target split name for logging context
        """
        split_context = f" for {target_split}" if target_split else ""
        
        # Main statistics
        self.logger.info(f"🔍 Augmented Sample Validation{split_context}:")
        self.logger.info(f"   Valid samples: {summary['valid_samples']}/{summary['total_samples']} ({summary['valid_rate']:.1f}%)")
        
        # Box statistics
        box_stats = summary['box_statistics']
        self.logger.info(f"   Bounding boxes: {box_stats['total_valid_boxes']}/{box_stats['total_original_boxes']} retained ({box_stats['box_retention_rate']:.1f}%)")
        
        # Filter reasons (only log if there are filtered samples)
        if summary['invalid_samples'] > 0:
            reasons = summary['filter_reasons']
            filtered_reasons = [(reason, count) for reason, count in reasons.items() if count > 0]
            
            if filtered_reasons:
                self.logger.info(f"   Filtering reasons:")
                for reason, count in filtered_reasons:
                    if reason in ['too_small', 'invalid_coords', 'out_of_bounds']:
                        self.logger.info(f"     • {reason.replace('_', ' ').title()}: {count} boxes")
                    else:
                        self.logger.info(f"     • {reason.replace('_', ' ').title()}: {count} samples")


def create_sample_validator(config: Dict[str, Any] = None) -> AugmentedSampleValidator:
    """
    Factory function to create a sample validator with configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured AugmentedSampleValidator instance
    """
    if config is None:
        return AugmentedSampleValidator()
    
    validation_config = config.get('validation', {})
    augmentation_config = config.get('augmentation', {})
    
    # Get validation parameters
    min_bbox_size = validation_config.get('min_bbox_size', 0.001)
    min_valid_boxes = validation_config.get('min_valid_boxes', 1)
    
    # Allow override from augmentation config for backward compatibility
    if 'min_bbox_size' in augmentation_config:
        min_bbox_size = augmentation_config['min_bbox_size']
    
    return AugmentedSampleValidator(
        min_bbox_size=min_bbox_size,
        min_valid_boxes=min_valid_boxes
    )