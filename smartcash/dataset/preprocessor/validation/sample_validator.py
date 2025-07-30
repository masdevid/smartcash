"""
File: smartcash/dataset/preprocessor/validation/sample_validator.py
Description: Invalid sample validator with hierarchical class validation and quarantine functionality
"""

import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union, Optional
from smartcash.common.logger import get_logger


class InvalidSampleValidator:
    """🔍 Validator for filtering out invalid samples with hierarchical label validation"""
    
    def __init__(self, 
                 min_bbox_size: float = 0.001, 
                 min_valid_boxes: int = 1,
                 quarantine_dir: Union[str, Path] = None):
        """
        Initialize validator with filtering criteria and quarantine directory.
        
        Args:
            min_bbox_size: Minimum width/height for valid bounding boxes (default: 0.001)
            min_valid_boxes: Minimum number of valid bounding boxes required (default: 1)
            quarantine_dir: Directory to move invalid samples to (default: data/invalid)
        """
        self.min_bbox_size = min_bbox_size
        self.min_valid_boxes = min_valid_boxes
        self.logger = get_logger(__name__)
        
        # Setup quarantine directory
        if quarantine_dir is None:
            self.quarantine_dir = Path('data/invalid')
        else:
            self.quarantine_dir = Path(quarantine_dir)
        
        # Class hierarchy validation mappings based on model_constants.py
        self.class_hierarchy = self._setup_class_hierarchy()
        
        # Validation statistics
        self.validation_stats = {
            'total_processed': 0,
            'invalid_samples': 0,
            'quarantined_samples': 0,
            'auto_fixed_samples': 0,
            'invalid_reasons': {
                'empty_labels': 0,
                'invalid_bbox_format': 0,
                'bbox_too_small': 0,
                'bbox_out_of_bounds': 0,
                'invalid_class_hierarchy': 0,
                'insufficient_valid_boxes': 0,
                'file_corruption': 0
            },
            'auto_fix_stats': {
                'layer_3_removed': 0,
                'mismatched_layer_2_removed': 0,
                'orphaned_layer_1_removed': 0,
                'total_labels_removed': 0
            }
        }
    
    def _setup_class_hierarchy(self) -> Dict[str, Any]:
        """Setup hierarchical class validation rules based on model constants."""
        return {
            'layer_1': {
                'class_ids': list(range(0, 7)),  # 0-6: banknote classes
                'names': ['001', '002', '005', '010', '020', '050', '100'],
                'compatible_layer_2': {
                    0: [7],   # 001 -> l2_001 (index 7)
                    1: [8],   # 002 -> l2_002 (index 8) 
                    2: [9],   # 005 -> l2_005 (index 9)
                    3: [10],  # 010 -> l2_010 (index 10)
                    4: [11],  # 020 -> l2_020 (index 11)
                    5: [12],  # 050 -> l2_050 (index 12)
                    6: [13],  # 100 -> l2_100 (index 13)
                }
            },
            'layer_2': {
                'class_ids': list(range(7, 14)),  # 7-13: denomination features
                'names': ['l2_001', 'l2_002', 'l2_005', 'l2_010', 'l2_020', 'l2_050', 'l2_100'],
                'compatible_layer_3': list(range(14, 17))  # All l2 classes can have l3 features
            },
            'layer_3': {
                'class_ids': list(range(14, 17)),  # 14-16: common features
                'names': ['l3_sign', 'l3_text', 'l3_thread']
            }
        }
    
    def validate_sample(self, image_path: Union[str, Path], 
                       label_path: Union[str, Path] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate a single image-label pair.
        
        Args:
            image_path: Path to image file
            label_path: Path to label file (optional, will be auto-detected)
            
        Returns:
            Tuple of (is_valid, validation_info)
        """
        self.validation_stats['total_processed'] += 1
        
        img_path = Path(image_path)
        
        # Auto-detect label file if not provided
        if label_path is None:
            label_path = self._find_corresponding_label(img_path)
        
        if label_path is None:
            reason = 'missing_label'
            self.validation_stats['invalid_reasons']['empty_labels'] += 1
            return False, {
                'reason': reason,
                'image_path': str(img_path),
                'label_path': None,
                'issues': ['Label file not found']
            }
        
        label_path = Path(label_path)
        
        # Check file existence
        if not img_path.exists():
            reason = 'missing_image'
            self.validation_stats['invalid_reasons']['file_corruption'] += 1
            return False, {
                'reason': reason,
                'image_path': str(img_path),
                'label_path': str(label_path),
                'issues': ['Image file not found']
            }
        
        if not label_path.exists():
            reason = 'missing_label'
            self.validation_stats['invalid_reasons']['empty_labels'] += 1
            return False, {
                'reason': reason,
                'image_path': str(img_path),
                'label_path': str(label_path),
                'issues': ['Label file not found']
            }
        
        # Parse label file and validate
        try:
            bboxes, class_labels = self._parse_yolo_label(label_path)
            
            if not bboxes or not class_labels:
                reason = 'empty_labels'
                self.validation_stats['invalid_reasons']['empty_labels'] += 1
                return False, {
                    'reason': reason,
                    'image_path': str(img_path),
                    'label_path': str(label_path),
                    'issues': ['No valid bounding boxes found in label file']
                }
            
            # Validate bounding boxes and class hierarchy
            is_valid, validation_info = self._validate_labels(bboxes, class_labels)
            validation_info.update({
                'image_path': str(img_path),
                'label_path': str(label_path)
            })
            
            if not is_valid:
                self.validation_stats['invalid_samples'] += 1
            
            return is_valid, validation_info
            
        except Exception as e:
            reason = 'file_corruption'
            self.validation_stats['invalid_reasons']['file_corruption'] += 1
            return False, {
                'reason': reason,
                'image_path': str(img_path),
                'label_path': str(label_path),
                'issues': [f'Error parsing label file: {str(e)}']
            }
    
    def _validate_labels(self, bboxes: List[List[float]], 
                        class_labels: List[int]) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate bounding boxes and hierarchical class relationships with auto-fix capability.
        
        Args:
            bboxes: List of YOLO format bounding boxes [x, y, w, h]
            class_labels: List of class IDs
            
        Returns:
            Tuple of (is_valid, validation_info)
        """
        issues = []
        valid_bboxes = []
        valid_class_labels = []
        
        if len(bboxes) != len(class_labels):
            self.validation_stats['invalid_reasons']['invalid_bbox_format'] += 1
            return False, {
                'reason': 'mismatched_lengths',
                'issues': [f'Mismatch: {len(bboxes)} bboxes vs {len(class_labels)} class labels'],
                'valid_count': 0,
                'original_count': len(bboxes)
            }
        
        # Validate individual bounding boxes
        for bbox, class_id in zip(bboxes, class_labels):
            bbox_valid, bbox_issues = self._validate_single_bbox(bbox, class_id)
            
            if bbox_valid:
                valid_bboxes.append(bbox)
                valid_class_labels.append(class_id)
            else:
                issues.extend(bbox_issues)
        
        # Check if we have enough valid boxes
        if len(valid_bboxes) < self.min_valid_boxes:
            self.validation_stats['invalid_reasons']['insufficient_valid_boxes'] += 1
            return False, {
                'reason': 'insufficient_valid_boxes',
                'issues': issues + [f'Only {len(valid_bboxes)} valid boxes, need at least {self.min_valid_boxes}'],
                'valid_count': len(valid_bboxes),
                'original_count': len(bboxes)
            }
        
        # Validate hierarchical class relationships with auto-fix
        hierarchy_result = self._validate_class_hierarchy_with_autofix(valid_bboxes, valid_class_labels)
        
        if not hierarchy_result['valid']:
            self.validation_stats['invalid_reasons']['invalid_class_hierarchy'] += 1
            return False, {
                'reason': 'invalid_class_hierarchy',
                'issues': issues + hierarchy_result['issues'],
                'valid_count': len(valid_bboxes),
                'original_count': len(bboxes),
                'class_hierarchy_issues': hierarchy_result['issues']
            }
        
        # Use auto-fixed labels if available
        final_bboxes = hierarchy_result.get('fixed_bboxes', valid_bboxes)
        final_class_labels = hierarchy_result.get('fixed_class_labels', valid_class_labels)
        
        # Track auto-fix statistics
        if hierarchy_result.get('auto_fixed', False):
            self.validation_stats['auto_fixed_samples'] += 1
            
            # Count removed labels by type
            for class_id, reason in hierarchy_result.get('removed_labels', []):
                self.validation_stats['auto_fix_stats']['total_labels_removed'] += 1
                
                if 'layer_3 removed' in reason:
                    self.validation_stats['auto_fix_stats']['layer_3_removed'] += 1
                elif 'removed - no matching' in reason and class_id in self.class_hierarchy['layer_2']['class_ids']:
                    self.validation_stats['auto_fix_stats']['mismatched_layer_2_removed'] += 1
                elif 'removed - no matching' in reason and class_id in self.class_hierarchy['layer_1']['class_ids']:
                    self.validation_stats['auto_fix_stats']['orphaned_layer_1_removed'] += 1
        
        # All validations passed
        return True, {
            'reason': 'valid',
            'issues': issues,
            'valid_count': len(final_bboxes),
            'original_count': len(bboxes),
            'valid_bboxes': final_bboxes,
            'valid_class_labels': final_class_labels,
            'auto_fixed': hierarchy_result.get('auto_fixed', False),
            'removed_labels': hierarchy_result.get('removed_labels', [])
        }
    
    def _validate_single_bbox(self, bbox: List[float], class_id: int) -> Tuple[bool, List[str]]:
        """Validate a single bounding box."""
        issues = []
        
        try:
            # Convert to float and validate format
            if len(bbox) < 4:
                issues.append(f'Bbox has {len(bbox)} values, need 4 (x, y, w, h)')
                self.validation_stats['invalid_reasons']['invalid_bbox_format'] += 1
                return False, issues
            
            x, y, w, h = [float(coord) for coord in bbox[:4]]
            
            # Check for NaN or infinite values
            if not all(isinstance(coord, (int, float)) and not (coord != coord) and coord != float('inf') and coord != float('-inf') for coord in [x, y, w, h]):
                issues.append('Bbox contains NaN or infinite values')
                self.validation_stats['invalid_reasons']['invalid_bbox_format'] += 1
                return False, issues
            
            # Normalize coordinates to [0, 1] range
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))
            
            # Check minimum size requirements
            if w < self.min_bbox_size or h < self.min_bbox_size:
                issues.append(f'Bbox too small: {w:.6f}x{h:.6f}, minimum: {self.min_bbox_size}')
                self.validation_stats['invalid_reasons']['bbox_too_small'] += 1
                return False, issues
            
            # Check if bbox is within valid bounds (with some tolerance)
            if (x - w/2 < -0.01 or x + w/2 > 1.01 or 
                y - h/2 < -0.01 or y + h/2 > 1.01):
                issues.append(f'Bbox out of bounds: center=({x:.3f},{y:.3f}), size=({w:.3f},{h:.3f})')
                self.validation_stats['invalid_reasons']['bbox_out_of_bounds'] += 1
                return False, issues
            
            # Validate class ID range
            if not (0 <= class_id <= 16):
                issues.append(f'Invalid class ID: {class_id}, valid range: 0-16')
                self.validation_stats['invalid_reasons']['invalid_bbox_format'] += 1
                return False, issues
            
            return True, []
            
        except (ValueError, TypeError) as e:
            issues.append(f'Bbox format error: {str(e)}')
            self.validation_stats['invalid_reasons']['invalid_bbox_format'] += 1
            return False, issues
    
    def _validate_class_hierarchy(self, class_labels: List[int]) -> Tuple[bool, List[str]]:
        """
        Validate hierarchical class relationships.
        
        Updated Rules:
        1. If labels contain more than one label, ignore l3_* (layer_3 classes)
        2. If l2_* exists, ensure it matches with layer_1
        3. Single labels are valid: either just 001 OR just l2_001
        """
        issues = []
        
        # Group classes by layer
        layer_1_classes = [c for c in class_labels if c in self.class_hierarchy['layer_1']['class_ids']]
        layer_2_classes = [c for c in class_labels if c in self.class_hierarchy['layer_2']['class_ids']]
        layer_3_classes = [c for c in class_labels if c in self.class_hierarchy['layer_3']['class_ids']]
        
        # Rule 1: If labels contain more than one label, ignore l3_* classes
        if len(class_labels) > 1 and layer_3_classes:
            # Filter out layer_3 classes from validation but log as info
            self.logger.debug(f"Multi-label sample detected, ignoring layer_3 classes: {layer_3_classes}")
            # We don't fail validation for this, just ignore l3 classes in multi-label scenarios
        
        # Rule 2: If l2_* exists, ensure it matches with layer_1 (if layer_1 also exists)
        if layer_2_classes and layer_1_classes:
            # Check that each l2 class has a matching l1 class
            for l2_class in layer_2_classes:
                # Find the corresponding layer_1 class for this layer_2 class
                corresponding_l1 = None
                for l1_class, compatible_l2s in self.class_hierarchy['layer_1']['compatible_layer_2'].items():
                    if l2_class in compatible_l2s:
                        corresponding_l1 = l1_class
                        break
                
                # Check if the corresponding l1 class exists in the sample
                if corresponding_l1 is not None and corresponding_l1 not in layer_1_classes:
                    l2_name = self.class_hierarchy['layer_2']['names'][l2_class - 7]
                    l1_name = self.class_hierarchy['layer_1']['names'][corresponding_l1]
                    issues.append(
                        f'Layer_2 class {l2_class} ({l2_name}) requires matching layer_1 class '
                        f'{corresponding_l1} ({l1_name}), but it is missing'
                    )
                elif corresponding_l1 is None:
                    issues.append(f'Unknown layer_2 class mapping for class {l2_class}')
        
        # Rule 3: Single labels are valid - either just layer_1 OR just layer_2
        # This is implicitly handled - no additional validation needed
        
        # Additional check: If layer_1 exists with layer_2, verify the pairing is correct
        if layer_1_classes and layer_2_classes:
            for l1_class in layer_1_classes:
                compatible_l2_classes = self.class_hierarchy['layer_1']['compatible_layer_2'].get(l1_class, [])
                
                # Check if there's at least one compatible l2 class present
                has_compatible_l2 = any(l2_class in layer_2_classes for l2_class in compatible_l2_classes)
                
                if not has_compatible_l2:
                    l1_name = self.class_hierarchy['layer_1']['names'][l1_class]
                    expected_l2_names = [self.class_hierarchy['layer_2']['names'][l2_id - 7] for l2_id in compatible_l2_classes]
                    issues.append(
                        f'Layer_1 class {l1_class} ({l1_name}) found with layer_2 classes, '
                        f'but no matching layer_2 class found. Expected: {compatible_l2_classes} ({expected_l2_names})'
                    )
        
        return len(issues) == 0, issues
    
    def _validate_class_hierarchy_with_autofix(self, bboxes: List[List[float]], 
                                              class_labels: List[int]) -> Dict[str, Any]:
        """
        Validate hierarchical class relationships with auto-fix capability.
        
        Rules with auto-fix:
        1. If labels contain more than one label, ignore l3_* (layer_3 classes)
        2. If l2_* exists, ensure it matches with layer_1 - auto-fix by removing invalid l2_*
        3. Single labels are valid: either just 001 OR just l2_001
        
        Example: 001, l2_001, l2_100 -> auto-fix to 001, l2_001 (remove l2_100)
        
        Returns:
            Dict with validation results and auto-fixed labels if applicable
        """
        issues = []
        auto_fixed = False
        removed_labels = []
        
        # Group classes by layer with their bbox indices
        layer_1_indices = [(i, c) for i, c in enumerate(class_labels) if c in self.class_hierarchy['layer_1']['class_ids']]
        layer_2_indices = [(i, c) for i, c in enumerate(class_labels) if c in self.class_hierarchy['layer_2']['class_ids']]
        layer_3_indices = [(i, c) for i, c in enumerate(class_labels) if c in self.class_hierarchy['layer_3']['class_ids']]
        
        # Start with all valid indices
        valid_indices = list(range(len(class_labels)))
        
        # Rule 1: If labels contain more than one label, ignore l3_* classes
        if len(class_labels) > 1 and layer_3_indices:
            self.logger.debug(f"Multi-label sample detected, removing layer_3 classes: {[c for _, c in layer_3_indices]}")
            for idx, class_id in layer_3_indices:
                valid_indices.remove(idx)
                removed_labels.append((class_id, f"layer_3 removed in multi-label sample"))
                auto_fixed = True
        
        # Rule 2: Handle layer_1 and layer_2 matching - auto-fix mismatches
        if layer_1_indices and layer_2_indices:
            # Both layer_1 and layer_2 exist - ensure they match
            # Check each layer_2 class
            indices_to_remove = []
            for idx, l2_class in layer_2_indices:
                if idx not in valid_indices:  # Already removed
                    continue
                    
                # Find which l1 class this l2 should match
                corresponding_l1 = None
                for l1_class, compatible_l2s in self.class_hierarchy['layer_1']['compatible_layer_2'].items():
                    if l2_class in compatible_l2s:
                        corresponding_l1 = l1_class
                        break
                
                # Check if the corresponding l1 class is present
                matching_l1_found = False
                if corresponding_l1 is not None:
                    l1_present = any(l1_class == corresponding_l1 for _, l1_class in layer_1_indices if _ in valid_indices)
                    if l1_present:
                        matching_l1_found = True
                
                # If no matching l1 found, mark this l2 for removal
                if not matching_l1_found:
                    indices_to_remove.append(idx)
                    l2_name = self.class_hierarchy['layer_2']['names'][l2_class - 7]
                    if corresponding_l1 is not None:
                        l1_name = self.class_hierarchy['layer_1']['names'][corresponding_l1]
                        removed_labels.append((l2_class, f"{l2_name} removed - no matching {l1_name} found"))
                    else:
                        removed_labels.append((l2_class, f"{l2_name} removed - unknown mapping"))
                    auto_fixed = True
            
            # Remove invalid l2 classes
            for idx in indices_to_remove:
                if idx in valid_indices:
                    valid_indices.remove(idx)
            
            # Now check if any l1 classes lack their required l2 matches
            l1_indices_to_remove = []
            for idx, l1_class in layer_1_indices:
                if idx not in valid_indices:  # Already removed
                    continue
                    
                compatible_l2_classes = self.class_hierarchy['layer_1']['compatible_layer_2'].get(l1_class, [])
                
                # Check if any compatible l2 class is present
                has_compatible_l2 = any(
                    l2_class in compatible_l2_classes 
                    for _, l2_class in layer_2_indices 
                    if _ in valid_indices
                )
                
                # If no compatible l2 found, mark this l1 for removal
                if not has_compatible_l2:
                    l1_indices_to_remove.append(idx)
                    l1_name = self.class_hierarchy['layer_1']['names'][l1_class]
                    expected_l2_names = [self.class_hierarchy['layer_2']['names'][l2_id - 7] for l2_id in compatible_l2_classes]
                    removed_labels.append((l1_class, f"{l1_name} removed - no matching {expected_l2_names} found"))
                    auto_fixed = True
            
            # Remove l1 classes without matching l2
            for idx in l1_indices_to_remove:
                if idx in valid_indices:
                    valid_indices.remove(idx)
        
        elif layer_2_indices and not layer_1_indices:
            # Only layer_2 classes exist - this is valid for single labels, no auto-fix needed
            # But if there are multiple mismatched l2 classes, we should keep only one
            if len([idx for idx in layer_2_indices if idx in valid_indices]) > 1:
                # Multiple layer_2 classes without layer_1 - this is problematic
                # We can't determine which is correct, so remove all but the first
                l2_indices_in_valid = [(idx, class_id) for idx, class_id in layer_2_indices if idx in valid_indices]
                
                for idx, class_id in l2_indices_in_valid[1:]:  # Keep first, remove rest
                    valid_indices.remove(idx)
                    l2_name = self.class_hierarchy['layer_2']['names'][class_id - 7]
                    removed_labels.append((class_id, f"{l2_name} removed - multiple orphaned l2 classes"))
                    auto_fixed = True
        
        # Check if we still have valid labels after auto-fix
        if not valid_indices:
            return {
                'valid': False,
                'issues': ['No valid labels remaining after auto-fix attempts'],
                'auto_fixed': auto_fixed,
                'removed_labels': removed_labels
            }
        
        # Create the fixed bboxes and class_labels
        fixed_bboxes = [bboxes[i] for i in valid_indices]
        fixed_class_labels = [class_labels[i] for i in valid_indices]
        
        # Final validation check on the fixed labels
        final_validation_result = self._validate_class_hierarchy(fixed_class_labels)
        if not final_validation_result[0]:
            return {
                'valid': False,
                'issues': final_validation_result[1],
                'auto_fixed': auto_fixed,
                'removed_labels': removed_labels
            }
        
        return {
            'valid': True,
            'issues': [],
            'fixed_bboxes': fixed_bboxes,
            'fixed_class_labels': fixed_class_labels,
            'auto_fixed': auto_fixed,
            'removed_labels': removed_labels
        }
    
    def quarantine_invalid_sample(self, image_path: Union[str, Path], 
                                 label_path: Union[str, Path], 
                                 validation_info: Dict[str, Any]) -> bool:
        """
        Move an invalid sample to the quarantine directory.
        
        Args:
            image_path: Path to invalid image
            label_path: Path to invalid label
            validation_info: Validation information dict
            
        Returns:
            True if successfully quarantined, False otherwise
        """
        try:
            img_path = Path(image_path)
            label_path = Path(label_path) if label_path else None
            
            # Setup quarantine structure
            quarantine_images = self.quarantine_dir / 'images'
            quarantine_labels = self.quarantine_dir / 'labels'
            quarantine_info = self.quarantine_dir / 'validation_info'
            
            for dir_path in [quarantine_images, quarantine_labels, quarantine_info]:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            # Move image
            img_dest = quarantine_images / img_path.name
            if img_path.exists():
                shutil.move(str(img_path), str(img_dest))
            
            # Move label if it exists
            if label_path and label_path.exists():
                label_dest = quarantine_labels / label_path.name
                shutil.move(str(label_path), str(label_dest))
            
            # Save validation info
            info_file = quarantine_info / f"{img_path.stem}_validation.txt"
            with open(info_file, 'w') as f:
                f.write(f"Validation failed for: {img_path.name}\n")
                f.write(f"Reason: {validation_info.get('reason', 'unknown')}\n")
                f.write(f"Issues:\n")
                for issue in validation_info.get('issues', []):
                    f.write(f"  - {issue}\n")
                f.write(f"\nValidation info: {validation_info}\n")
            
            self.validation_stats['quarantined_samples'] += 1
            self.logger.info(f"🗂️ Quarantined invalid sample: {img_path.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to quarantine {image_path}: {str(e)}")
            return False
    
    def _find_corresponding_label(self, image_path: Path) -> Optional[Path]:
        """Find corresponding label file for an image."""
        # Standard naming: same stem, .txt extension
        label_path = image_path.with_suffix('.txt')
        if label_path.exists():
            return label_path
        
        # Check in labels directory structure
        if image_path.parent.name == 'images':
            labels_dir = image_path.parent.parent / 'labels'
            label_in_labels_dir = labels_dir / f"{image_path.stem}.txt"
            if label_in_labels_dir.exists():
                return label_in_labels_dir
        
        return None
    
    def _parse_yolo_label(self, label_path: Path) -> Tuple[List[List[float]], List[int]]:
        """Parse YOLO format label file."""
        bboxes = []
        class_labels = []
        
        try:
            with open(label_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    try:
                        parts = line.split()
                        if len(parts) < 5:
                            self.logger.debug(f"Skipping invalid line {line_num} in {label_path}: insufficient values")
                            continue
                        
                        class_id = int(float(parts[0]))
                        bbox = [float(x) for x in parts[1:5]]
                        
                        bboxes.append(bbox)
                        class_labels.append(class_id)
                        
                    except (ValueError, IndexError) as e:
                        self.logger.debug(f"Skipping invalid line {line_num} in {label_path}: {str(e)}")
                        continue
                        
        except Exception as e:
            self.logger.error(f"Error reading {label_path}: {str(e)}")
            raise
        
        return bboxes, class_labels
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get comprehensive validation statistics including auto-fix information."""
        total = self.validation_stats['total_processed']
        invalid = self.validation_stats['invalid_samples']
        quarantined = self.validation_stats['quarantined_samples']
        auto_fixed = self.validation_stats['auto_fixed_samples']
        
        return {
            'total_processed': total,
            'valid_samples': total - invalid,
            'invalid_samples': invalid,
            'quarantined_samples': quarantined,
            'auto_fixed_samples': auto_fixed,
            'valid_rate': ((total - invalid) / max(total, 1)) * 100,
            'invalid_rate': (invalid / max(total, 1)) * 100,
            'quarantine_rate': (quarantined / max(total, 1)) * 100,
            'auto_fix_rate': (auto_fixed / max(total, 1)) * 100,
            'invalid_reasons': self.validation_stats['invalid_reasons'].copy(),
            'auto_fix_stats': self.validation_stats['auto_fix_stats'].copy(),
            'quarantine_directory': str(self.quarantine_dir)
        }
    
    def log_validation_summary(self, summary: Dict[str, Any] = None):
        """Log validation summary with clear statistics including auto-fix information."""
        if summary is None:
            summary = self.get_validation_summary()
        
        self.logger.info("🔍 Invalid Sample Validation Summary:")
        self.logger.info(f"   Total processed: {summary['total_processed']}")
        self.logger.info(f"   Valid samples: {summary['valid_samples']} ({summary['valid_rate']:.1f}%)")
        self.logger.info(f"   Invalid samples: {summary['invalid_samples']} ({summary['invalid_rate']:.1f}%)")
        self.logger.info(f"   Auto-fixed samples: {summary['auto_fixed_samples']} ({summary['auto_fix_rate']:.1f}%)")
        self.logger.info(f"   Quarantined: {summary['quarantined_samples']} ({summary['quarantine_rate']:.1f}%)")
        
        # Log auto-fix statistics if any
        if summary['auto_fixed_samples'] > 0:
            auto_fix_stats = summary['auto_fix_stats']
            self.logger.info("   Auto-fix breakdown:")
            if auto_fix_stats['layer_3_removed'] > 0:
                self.logger.info(f"     • Layer_3 labels removed: {auto_fix_stats['layer_3_removed']}")
            if auto_fix_stats['mismatched_layer_2_removed'] > 0:
                self.logger.info(f"     • Mismatched Layer_2 labels removed: {auto_fix_stats['mismatched_layer_2_removed']}")
            if auto_fix_stats['orphaned_layer_1_removed'] > 0:
                self.logger.info(f"     • Orphaned Layer_1 labels removed: {auto_fix_stats['orphaned_layer_1_removed']}")
            self.logger.info(f"     • Total labels removed: {auto_fix_stats['total_labels_removed']}")
        
        # Log detailed invalid reasons if any
        if summary['invalid_samples'] > 0:
            self.logger.info("   Invalid reasons breakdown:")
            for reason, count in summary['invalid_reasons'].items():
                if count > 0:
                    reason_display = reason.replace('_', ' ').title()
                    self.logger.info(f"     • {reason_display}: {count}")
        
        if summary['quarantined_samples'] > 0:
            self.logger.info(f"   Quarantine directory: {summary['quarantine_directory']}")


def create_invalid_sample_validator(config: Dict[str, Any] = None) -> InvalidSampleValidator:
    """
    Factory function to create an invalid sample validator with configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured InvalidSampleValidator instance
    """
    if config is None:
        config = {}
    
    validation_config = config.get('validation', {})
    data_config = config.get('data', {})
    
    # Get validation parameters
    min_bbox_size = validation_config.get('min_bbox_size', 0.001)
    min_valid_boxes = validation_config.get('min_valid_boxes', 1)
    
    # Setup quarantine directory
    quarantine_dir = data_config.get('invalid_dir', 'data/invalid')
    
    return InvalidSampleValidator(
        min_bbox_size=min_bbox_size,
        min_valid_boxes=min_valid_boxes,
        quarantine_dir=quarantine_dir
    )