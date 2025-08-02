#!/usr/bin/env python3
"""
Layer_1 label deduplication processor for SmartCash dataset preprocessing.

This module handles the removal of duplicate layer_1 labels (classes 0-6),
keeping only the largest bounding box for each layer_1 class while preserving
all layer_2 and layer_3 labels unchanged.

Single Responsibility: Clean up duplicate layer_1 labels by retaining the largest bbox per class.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

from smartcash.common.logger import get_logger

logger = get_logger(__name__)


class LabelDeduplicator:
    """
    Handles deduplication of layer_1 labels in YOLO format annotations.
    
    For each layer_1 class (0-6: Indonesian banknote denominations), keeps only 
    the bounding box with the largest area, removing smaller duplicate bounding boxes.
    
    Layer_2 and Layer_3 classes are preserved unchanged to maintain hierarchical
    relationships required by the SmartCash multi-layer detection system.
    """
    
    def __init__(self, backup_enabled: bool = True):
        """
        Initialize the label deduplicator.
        
        Args:
            backup_enabled: Whether to create backup files before modification
        """
        self.backup_enabled = backup_enabled
        self.stats = {
            'files_processed': 0,
            'duplicates_removed': 0,
            'classes_affected': set(),
            'errors': 0
        }
        
        logger.debug("✅ Label deduplicator initialized")
        logger.debug(f"   • Backup enabled: {backup_enabled}")
    
    def calculate_bbox_area(self, bbox: List[float]) -> float:
        """
        Calculate the area of a YOLO format bounding box.
        
        Args:
            bbox: YOLO format [x_center, y_center, width, height] (normalized 0-1)
            
        Returns:
            Area of the bounding box
        """
        if len(bbox) < 4:
            return 0.0
        
        # YOLO format: [x_center, y_center, width, height]
        width = bbox[2]
        height = bbox[3]
        
        return width * height
    
    def parse_yolo_line(self, line: str) -> Optional[Tuple[int, List[float]]]:
        """
        Parse a single YOLO annotation line.
        
        Args:
            line: YOLO format line "class_id x_center y_center width height"
            
        Returns:
            Tuple of (class_id, bbox_coords) or None if invalid
        """
        try:
            parts = line.strip().split()
            if len(parts) < 5:
                return None
                
            class_id = int(parts[0])
            bbox = [float(x) for x in parts[1:5]]
            
            # Validate bbox coordinates (should be normalized 0-1)
            if not all(0.0 <= coord <= 1.0 for coord in bbox):
                logger.warning(f"Invalid bbox coordinates: {bbox}")
                
            return class_id, bbox
            
        except (ValueError, IndexError) as e:
            logger.warning(f"Error parsing YOLO line '{line}': {e}")
            return None
    
    def deduplicate_labels(self, annotations: List[str], layer1_classes_only: bool = True) -> List[str]:
        """
        Remove duplicate labels, keeping only the largest bbox per class.
        
        Args:
            annotations: List of YOLO format annotation lines
            layer1_classes_only: If True, only deduplicate layer_1 classes (0-6)
            
        Returns:
            List of deduplicated annotation lines
        """
        if not annotations:
            return annotations
        
        # Parse all annotations
        parsed_annotations = []
        for line in annotations:
            if line.strip():
                parsed = self.parse_yolo_line(line)
                if parsed:
                    parsed_annotations.append((parsed[0], parsed[1], line.strip()))
        
        if not parsed_annotations:
            return annotations
        
        # Define layer_1 classes (SmartCash banknote denominations: 0-6)
        LAYER_1_CLASSES = set(range(7))  # Classes 0, 1, 2, 3, 4, 5, 6
        
        # Separate layer_1 and other classes
        layer1_annotations = []
        other_annotations = []
        
        for class_id, bbox, original_line in parsed_annotations:
            if class_id in LAYER_1_CLASSES:
                layer1_annotations.append((class_id, bbox, original_line))
            else:
                other_annotations.append((class_id, bbox, original_line))
        
        # Group layer_1 classes by class_id and find largest bbox for each class
        layer1_groups = {}
        for class_id, bbox, original_line in layer1_annotations:
            if class_id not in layer1_groups:
                layer1_groups[class_id] = []
            
            area = self.calculate_bbox_area(bbox)
            layer1_groups[class_id].append((area, bbox, original_line))
        
        # Keep only the largest bbox for each layer_1 class
        deduplicated_lines = []
        duplicates_count = 0
        
        for class_id, bboxes in layer1_groups.items():
            if len(bboxes) > 1:
                # Multiple bboxes for same layer_1 class - keep the largest
                bboxes.sort(key=lambda x: x[0], reverse=True)  # Sort by area (descending)
                largest = bboxes[0]
                
                duplicates_count += len(bboxes) - 1
                self.stats['classes_affected'].add(class_id)
                
                logger.debug(f"Layer_1 Class {class_id}: Removed {len(bboxes) - 1} duplicates, kept largest (area: {largest[0]:.6f})")
                
                deduplicated_lines.append(largest[2])  # Keep the original line
            else:
                # Single bbox for this layer_1 class - keep it
                deduplicated_lines.append(bboxes[0][2])
        
        # Add all non-layer_1 classes unchanged (layer_2, layer_3, etc.)
        for class_id, bbox, original_line in other_annotations:
            deduplicated_lines.append(original_line)
        
        self.stats['duplicates_removed'] += duplicates_count
        
        if duplicates_count > 0:
            logger.debug(f"🎯 Layer_1 deduplication: Removed {duplicates_count} duplicates, preserved all layer_2/layer_3 labels")
        
        return deduplicated_lines
    
    def create_backup(self, file_path: Path) -> bool:
        """
        Create a backup of the original file.
        
        Args:
            file_path: Path to the file to backup
            
        Returns:
            True if backup was created successfully
        """
        if not self.backup_enabled:
            return True
            
        try:
            backup_path = file_path.with_suffix(f"{file_path.suffix}.backup")
            
            # Don't overwrite existing backups
            if backup_path.exists():
                logger.debug(f"Backup already exists: {backup_path}")
                return True
                
            # Copy original to backup
            import shutil
            shutil.copy2(file_path, backup_path)
            
            logger.debug(f"✅ Backup created: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create backup for {file_path}: {e}")
            return False
    
    def process_annotation_file(self, file_path: Path) -> bool:
        """
        Process a single YOLO annotation file to remove duplicate labels.
        
        Args:
            file_path: Path to the .txt annotation file
            
        Returns:
            True if processing was successful
        """
        try:
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                return False
            
            # Read original annotations
            with open(file_path, 'r', encoding='utf-8') as f:
                original_lines = f.readlines()
            
            if not original_lines:
                logger.debug(f"Empty annotation file: {file_path}")
                return True
            
            # Remove duplicates
            original_count = len([line for line in original_lines if line.strip()])
            deduplicated_lines = self.deduplicate_labels(original_lines)
            new_count = len(deduplicated_lines)
            
            # Only write if there were changes
            if new_count != original_count:
                # Create backup before modifying
                if not self.create_backup(file_path):
                    logger.error(f"Skipping {file_path} - backup failed")
                    return False
                
                # Write deduplicated annotations
                with open(file_path, 'w', encoding='utf-8') as f:
                    for line in deduplicated_lines:
                        f.write(f"{line}\n")
                
                logger.info(f"✅ {file_path.name}: {original_count} → {new_count} labels ({original_count - new_count} duplicates removed)")
            else:
                logger.debug(f"No duplicates found in: {file_path.name}")
            
            self.stats['files_processed'] += 1
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing {file_path}: {e}")
            self.stats['errors'] += 1
            return False
    
    def process_directory(self, annotations_dir: Path, pattern: str = "*.txt") -> Dict[str, Any]:
        """
        Process all annotation files in a directory.
        
        Args:
            annotations_dir: Directory containing YOLO annotation files
            pattern: File pattern to match (default: "*.txt")
            
        Returns:
            Processing statistics
        """
        try:
            annotations_dir = Path(annotations_dir)
            
            if not annotations_dir.exists():
                raise FileNotFoundError(f"Directory not found: {annotations_dir}")
            
            if not annotations_dir.is_dir():
                raise NotADirectoryError(f"Not a directory: {annotations_dir}")
            
            # Find all annotation files
            annotation_files = list(annotations_dir.glob(pattern))
            
            if not annotation_files:
                logger.warning(f"No annotation files found in: {annotations_dir}")
                return self.get_stats()
            
            logger.info(f"🔍 Found {len(annotation_files)} annotation files to process")
            logger.info(f"📁 Processing directory: {annotations_dir}")
            
            # Process each file
            success_count = 0
            for file_path in annotation_files:
                if self.process_annotation_file(file_path):
                    success_count += 1
            
            logger.info(f"✅ Processing complete: {success_count}/{len(annotation_files)} files processed successfully")
            
            return self.get_stats()
            
        except Exception as e:
            logger.error(f"❌ Error processing directory {annotations_dir}: {e}")
            self.stats['errors'] += 1
            return self.get_stats()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dictionary containing processing statistics
        """
        return {
            'files_processed': self.stats['files_processed'],
            'duplicates_removed': self.stats['duplicates_removed'],
            'classes_affected': list(self.stats['classes_affected']),
            'errors': self.stats['errors'],
            'success_rate': (self.stats['files_processed'] / max(1, self.stats['files_processed'] + self.stats['errors'])) * 100
        }
    
    def print_stats(self):
        """Print processing statistics to the console."""
        stats = self.get_stats()
        
        logger.info("📊 Label Deduplication Statistics:")
        logger.info(f"   • Files processed: {stats['files_processed']}")
        logger.info(f"   • Duplicates removed: {stats['duplicates_removed']}")
        logger.info(f"   • Classes affected: {stats['classes_affected']}")
        logger.info(f"   • Errors: {stats['errors']}")
        logger.info(f"   • Success rate: {stats['success_rate']:.1f}%")


def deduplicate_labels_in_directory(
    annotations_dir: str,
    backup_enabled: bool = True,
    pattern: str = "*.txt"
) -> Dict[str, Any]:
    """
    Convenience function to deduplicate labels in a directory.
    
    Args:
        annotations_dir: Directory containing YOLO annotation files
        backup_enabled: Whether to create backup files
        pattern: File pattern to match
        
    Returns:
        Processing statistics
        
    Example:
        >>> stats = deduplicate_labels_in_directory("data/labels/train")
        >>> print(f"Removed {stats['duplicates_removed']} duplicates")
    """
    deduplicator = LabelDeduplicator(backup_enabled=backup_enabled)
    
    try:
        stats = deduplicator.process_directory(Path(annotations_dir), pattern)
        deduplicator.print_stats()
        return stats
        
    except Exception as e:
        logger.error(f"❌ Label deduplication failed: {e}")
        return deduplicator.get_stats()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Remove duplicate labels from YOLO annotation files")
    parser.add_argument("annotations_dir", help="Directory containing YOLO annotation files")
    parser.add_argument("--no-backup", action="store_true", help="Disable backup file creation")
    parser.add_argument("--pattern", default="*.txt", help="File pattern to match (default: *.txt)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    
    # Run deduplication
    stats = deduplicate_labels_in_directory(
        args.annotations_dir,
        backup_enabled=not args.no_backup,
        pattern=args.pattern
    )
    
    print(f"\n✅ Deduplication complete!")
    print(f"Files processed: {stats['files_processed']}")
    print(f"Duplicates removed: {stats['duplicates_removed']}")
    print(f"Classes affected: {stats['classes_affected']}")