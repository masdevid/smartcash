#!/usr/bin/env python3
"""
API interface for label cleanup operations in SmartCash dataset preprocessing.

This module provides a clean, high-level API for label deduplication and cleanup
operations, integrating with the existing preprocessor architecture.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import json

from smartcash.common.logger import get_logger
from smartcash.dataset.preprocessor.core.label_deduplicator import LabelDeduplicator, deduplicate_labels_in_directory

logger = get_logger(__name__)


class LabelCleanupAPI:
    """
    High-level API for label cleanup operations.
    
    Provides a unified interface for various label cleanup tasks including
    deduplication, validation, and statistics reporting.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the label cleanup API.
        
        Args:
            config: Configuration dictionary with cleanup settings
        """
        self.config = config or {}
        self.backup_enabled = self.config.get('backup_enabled', True)
        self.verbose = self.config.get('verbose', False)
        
        logger.info("🔧 Label Cleanup API initialized")
        if self.verbose:
            logger.info(f"   • Configuration: {self.config}")
    
    def deduplicate_layer1_labels(
        self,
        annotations_dir: str,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Remove duplicate layer_1 labels (classes 0-6), keeping only the largest bounding box per class.
        
        Layer_2 and layer_3 labels are preserved unchanged to maintain hierarchical relationships.
        
        Args:
            annotations_dir: Directory containing YOLO annotation files
            dry_run: If True, simulate the operation without making changes
            
        Returns:
            Dictionary with operation results and statistics
        """
        try:
            annotations_path = Path(annotations_dir)
            
            if not annotations_path.exists():
                raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")
            
            logger.info(f"🧹 Starting layer_1 label deduplication")
            logger.info(f"   • Directory: {annotations_path}")
            logger.info(f"   • Dry run: {dry_run}")
            logger.info(f"   • Backup enabled: {self.backup_enabled}")
            
            if dry_run:
                # For dry run, analyze without making changes
                return self._analyze_duplicates(annotations_path)
            else:
                # Perform actual deduplication
                stats = deduplicate_labels_in_directory(
                    str(annotations_path),
                    backup_enabled=self.backup_enabled,
                    pattern="*.txt"
                )
                
                return {
                    'operation': 'deduplicate_layer1_labels',
                    'status': 'completed',
                    'directory': str(annotations_path),
                    'dry_run': False,
                    'statistics': stats
                }
                
        except Exception as e:
            logger.error(f"❌ Label deduplication failed: {e}")
            return {
                'operation': 'deduplicate_layer1_labels',
                'status': 'failed',
                'error': str(e),
                'directory': annotations_dir,
                'dry_run': dry_run
            }
    
    def _analyze_duplicates(self, annotations_dir: Path) -> Dict[str, Any]:
        """
        Analyze duplicate labels without making changes (dry run).
        
        Args:
            annotations_dir: Directory containing annotation files
            
        Returns:
            Analysis results
        """
        try:
            annotation_files = list(annotations_dir.glob("*.txt"))
            
            if not annotation_files:
                return {
                    'operation': 'analyze_duplicates',
                    'status': 'completed',
                    'dry_run': True,
                    'files_found': 0,
                    'potential_duplicates': 0,
                    'classes_with_duplicates': []
                }
            
            deduplicator = LabelDeduplicator(backup_enabled=False)
            potential_duplicates = 0
            classes_with_duplicates = set()
            files_analyzed = 0
            
            for file_path in annotation_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    if not lines:
                        continue
                    
                    # Parse annotations and check for duplicates
                    class_counts = {}
                    for line in lines:
                        if line.strip():
                            parsed = deduplicator.parse_yolo_line(line)
                            if parsed:
                                class_id, _ = parsed
                                class_counts[class_id] = class_counts.get(class_id, 0) + 1
                    
                    # Count duplicates
                    for class_id, count in class_counts.items():
                        if count > 1:
                            potential_duplicates += count - 1
                            classes_with_duplicates.add(class_id)
                    
                    files_analyzed += 1
                    
                except Exception as e:
                    logger.warning(f"Error analyzing {file_path}: {e}")
                    continue
            
            logger.info(f"📊 Dry run analysis complete:")
            logger.info(f"   • Files analyzed: {files_analyzed}")
            logger.info(f"   • Potential duplicates: {potential_duplicates}")
            logger.info(f"   • Classes with duplicates: {list(classes_with_duplicates)}")
            
            return {
                'operation': 'analyze_duplicates',
                'status': 'completed',
                'dry_run': True,
                'files_analyzed': files_analyzed,
                'potential_duplicates': potential_duplicates,
                'classes_with_duplicates': list(classes_with_duplicates)
            }
            
        except Exception as e:
            logger.error(f"❌ Duplicate analysis failed: {e}")
            return {
                'operation': 'analyze_duplicates',
                'status': 'failed',
                'error': str(e),
                'dry_run': True
            }
    
    def validate_labels(self, annotations_dir: str) -> Dict[str, Any]:
        """
        Validate label files for format correctness and potential issues.
        
        Args:
            annotations_dir: Directory containing annotation files
            
        Returns:
            Validation results
        """
        try:
            annotations_path = Path(annotations_dir)
            
            if not annotations_path.exists():
                raise FileNotFoundError(f"Directory not found: {annotations_dir}")
            
            annotation_files = list(annotations_path.glob("*.txt"))
            
            validation_results = {
                'total_files': len(annotation_files),
                'valid_files': 0,
                'invalid_files': 0,
                'errors': [],
                'warnings': [],
                'statistics': {
                    'total_labels': 0,
                    'classes_found': set(),
                    'bbox_issues': 0
                }
            }
            
            deduplicator = LabelDeduplicator(backup_enabled=False)
            
            for file_path in annotation_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    file_valid = True
                    file_labels = 0
                    
                    for line_num, line in enumerate(lines, 1):
                        if not line.strip():
                            continue
                            
                        parsed = deduplicator.parse_yolo_line(line)
                        if parsed is None:
                            validation_results['errors'].append({
                                'file': file_path.name,
                                'line': line_num,
                                'issue': 'Invalid YOLO format',
                                'content': line.strip()
                            })
                            file_valid = False
                        else:
                            class_id, bbox = parsed
                            validation_results['statistics']['classes_found'].add(class_id)
                            file_labels += 1
                            
                            # Check for bbox coordinate issues
                            if not all(0.0 <= coord <= 1.0 for coord in bbox):
                                validation_results['warnings'].append({
                                    'file': file_path.name,
                                    'line': line_num,
                                    'issue': 'Bbox coordinates out of range [0,1]',
                                    'bbox': bbox
                                })
                                validation_results['statistics']['bbox_issues'] += 1
                    
                    validation_results['statistics']['total_labels'] += file_labels
                    
                    if file_valid:
                        validation_results['valid_files'] += 1
                    else:
                        validation_results['invalid_files'] += 1
                        
                except Exception as e:
                    validation_results['errors'].append({
                        'file': file_path.name,
                        'issue': f'File read error: {e}'
                    })
                    validation_results['invalid_files'] += 1
            
            # Convert set to list for JSON serialization
            validation_results['statistics']['classes_found'] = list(validation_results['statistics']['classes_found'])
            
            logger.info(f"📋 Label validation complete:")
            logger.info(f"   • Total files: {validation_results['total_files']}")
            logger.info(f"   • Valid files: {validation_results['valid_files']}")
            logger.info(f"   • Invalid files: {validation_results['invalid_files']}")
            logger.info(f"   • Total labels: {validation_results['statistics']['total_labels']}")
            logger.info(f"   • Classes found: {validation_results['statistics']['classes_found']}")
            
            return {
                'operation': 'validate_labels',
                'status': 'completed',
                'results': validation_results
            }
            
        except Exception as e:
            logger.error(f"❌ Label validation failed: {e}")
            return {
                'operation': 'validate_labels',
                'status': 'failed',
                'error': str(e)
            }
    
    def cleanup_and_validate(
        self,
        annotations_dir: str,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Perform comprehensive label cleanup and validation.
        
        Args:
            annotations_dir: Directory containing annotation files
            dry_run: If True, analyze without making changes
            
        Returns:
            Combined results from cleanup and validation
        """
        logger.info("🔄 Starting comprehensive label cleanup and validation")
        
        # First, validate current state
        validation_before = self.validate_labels(annotations_dir)
        
        # Then deduplicate (if not dry run)
        dedup_results = self.deduplicate_layer1_labels(annotations_dir, dry_run=dry_run)
        
        # Finally, validate after cleanup (if not dry run)
        validation_after = None
        if not dry_run and dedup_results.get('status') == 'completed':
            validation_after = self.validate_labels(annotations_dir)
        
        return {
            'operation': 'cleanup_and_validate',
            'status': 'completed',
            'dry_run': dry_run,
            'validation_before': validation_before,
            'deduplication': dedup_results,
            'validation_after': validation_after
        }


def create_label_cleanup_api(config: Optional[Dict[str, Any]] = None) -> LabelCleanupAPI:
    """
    Factory function to create a LabelCleanupAPI instance.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured LabelCleanupAPI instance
    """
    return LabelCleanupAPI(config)