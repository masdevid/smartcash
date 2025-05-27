"""
File: smartcash/dataset/augmentor/services/split_validator.py
Deskripsi: Validator service menggunakan SRP detector reuse
"""

from smartcash.dataset.augmentor.utils.dataset_detector import detect_split_structure, count_dataset_files_split_aware

class SplitValidator:
    """Validator menggunakan SRP detector reuse"""
    
    def __init__(self, config: Dict[str, Any], paths: Dict[str, str], communicator=None):
        self.config, self.paths, self.comm = config, paths, communicator
        self.progress = create_progress_tracker(communicator)
    
    def validate_dataset_with_uuid_check(self, target_split: str) -> Dict[str, Any]:
        """Validate menggunakan SRP detector"""
        try:
            dataset_info = detect_split_structure(self.paths['raw_dir'])
            
            if dataset_info['status'] == 'error' or dataset_info['total_images'] == 0:
                return {'valid': False, 'message': f"Basic validation failed: {dataset_info.get('message')}"}
            
            # Count files untuk specific split
            images_count, labels_count = count_dataset_files_split_aware(self.paths['raw_dir'], target_split)
            
            return {
                'valid': True, 'total_files': images_count, 'target_split': target_split,
                'split_structure': dataset_info.get('structure_type') == 'split_based',
                'message': f"Validation passed: {images_count} images, {labels_count} labels"
            }
            
        except Exception as e:
            return {'valid': False, 'message': f'Validation error: {str(e)}'}
    
    def validate_augmentation_consistency(self, target_split: str) -> Dict[str, Any]:
        """Validate consistency post-augmentation"""
        try:
            aug_info = detect_split_structure(f"{self.paths['aug_dir']}/{target_split}")
            
            return {
                'consistent': aug_info['status'] == 'success' and aug_info['total_images'] > 0,
                'total_files': aug_info.get('total_images', 0), 'issues': [],
                'message': f"Consistency check: {aug_info.get('total_images', 0)} augmented files"
            }
            
        except Exception as e:
            return {'consistent': False, 'issues': [f'Validation error: {str(e)}']}
    
    def get_status(self) -> Dict[str, Any]:
        """Get validator status"""
        return {'validator_ready': True, 'split_aware': True, 'detector_integrated': True}