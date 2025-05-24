"""
File: smartcash/ui/dataset/preprocessing/utils/validation_helper.py
Deskripsi: Helper untuk dataset validation dan compatibility checks
"""

from typing import Dict, Any, Tuple
from pathlib import Path
from smartcash.dataset.utils.path_validator import get_path_validator

class ValidationHelper:
    """Helper untuk dataset validation dan compatibility checks."""
    
    def __init__(self, ui_components: Dict[str, Any], logger=None):
        self.ui_components = ui_components
        self.logger = logger
        self.path_validator = get_path_validator(logger)
    
    def check_dataset_exists(self) -> Tuple[bool, str]:
        """Check dataset existence dengan comprehensive validation."""
        data_dir = self.ui_components.get('data_dir', 'data')
        validation_result = self.path_validator.validate_dataset_structure(data_dir)
        
        if not validation_result['valid']:
            return False, f"Dataset tidak ditemukan: {data_dir}"
        
        # Check critical issues
        critical_issues = [i for i in validation_result['issues'] if '❌' in i]
        if critical_issues:
            return False, f"Critical issues: {', '.join(critical_issues)}"
        
        if validation_result['total_images'] == 0:
            return False, "Dataset kosong, tidak ada gambar ditemukan"
        
        return True, f"Dataset valid: {validation_result['total_images']} gambar"
    
    def check_preprocessed_data(self) -> Tuple[bool, list, str, int]:
        """Check preprocessed data dengan comprehensive info."""
        preprocessed_dir = self.ui_components.get('preprocessed_dir', 'data/preprocessed')
        validation_result = self.path_validator.validate_preprocessed_structure(preprocessed_dir)
        
        if not validation_result['valid']:
            return False, [], "Tidak ada data preprocessing", 0
        
        available_splits = []
        total_files = validation_result['total_processed']
        
        for split, info in validation_result['splits'].items():
            if info['processed'] > 0:
                available_splits.append(f"{split} ({info['processed']:,} files)")
        
        if not available_splits:
            return False, [], "Tidak ada data valid untuk cleanup", 0
        
        return True, available_splits, f"Total {total_files:,} files", total_files
    
    def check_existing_preprocessed_for_split(self, split_config: str) -> list:
        """Check existing preprocessed data untuk split tertentu."""
        preprocessed_dir = Path(self.ui_components.get('preprocessed_dir', 'data/preprocessed'))
        preprocessed_validation = self.path_validator.validate_preprocessed_structure(str(preprocessed_dir))
        existing_data = []
        
        if split_config == 'all':
            for split in ['train', 'valid', 'test']:
                if (preprocessed_validation['splits'].get(split, {}).get('processed', 0) > 0):
                    existing_data.append(split)
        else:
            # Handle val->valid mapping
            check_split = 'valid' if split_config == 'val' else split_config
            if (preprocessed_validation['splits'].get(check_split, {}).get('processed', 0) > 0):
                existing_data.append(check_split)
        
        return existing_data
    
    def analyze_class_distribution(self, data_dir: str) -> Dict[str, Any]:
        """Analyze class distribution dengan error handling."""
        try:
            from smartcash.dataset.services.explorer.class_explorer import ClassExplorer
            
            config = {
                'data': {'dir': data_dir},
                'classes': {}
            }
            
            # Safe constructor call
            explorer = ClassExplorer(config, data_dir, self.logger, 4)
            class_result = explorer.analyze_distribution('train')
            
            if class_result.get('status') == 'success':
                return {
                    'total_classes': class_result.get('class_count', 0),
                    'class_counts': class_result.get('counts', {}),
                    'imbalance_score': class_result.get('imbalance_score', 0),
                    'analysis_success': True
                }
        except Exception as e:
            self.logger and self.logger.debug(f"🔍 Class analysis tidak tersedia: {str(e)}")
        
        return {'analysis_success': False}
    
    def check_preprocessing_compatibility(self, raw_validation: Dict[str, Any]) -> Dict[str, Any]:
        """Check preprocessing compatibility dan requirements."""
        compatibility = {
            'ready_for_preprocessing': True,
            'blocking_issues': [],
            'warnings': [],
            'estimated_time': 0,
            'estimated_output_size': 0
        }
        
        # Check blocking issues
        if not raw_validation['valid']:
            compatibility['ready_for_preprocessing'] = False
            compatibility['blocking_issues'].append("Dataset tidak ditemukan atau tidak valid")
            return compatibility
        
        critical_issues = [i for i in raw_validation['issues'] if '❌' in i]
        if critical_issues:
            compatibility['ready_for_preprocessing'] = False
            compatibility['blocking_issues'].extend(critical_issues)
        
        # Check warnings
        warning_issues = [i for i in raw_validation['issues'] if '⚠️' in i]
        compatibility['warnings'].extend(warning_issues)
        
        # Estimate processing time dan output size
        total_images = raw_validation['total_images']
        if total_images > 0:
            # Rough estimates based on image count
            compatibility['estimated_time'] = max(1, total_images * 0.1)  # ~0.1s per image
            compatibility['estimated_output_size'] = total_images * 0.2  # ~0.2MB per processed image
        
        return compatibility
    
    def generate_smart_recommendations(self, raw_validation: Dict[str, Any], 
                                     preprocessed_validation: Dict[str, Any],
                                     class_analysis: Dict[str, Any]) -> list:
        """Generate smart recommendations based on analysis."""
        recommendations = []
        
        # Dataset status recommendations
        if not raw_validation['valid']:
            recommendations.append({
                'type': 'critical',
                'icon': '📥',
                'title': 'Download Dataset',
                'message': 'Download dataset terlebih dahulu sebelum preprocessing'
            })
            return recommendations
        
        # Preprocessing recommendations
        if preprocessed_validation['total_processed'] == 0:
            recommendations.append({
                'type': 'action',
                'icon': '🚀',
                'title': 'Siap untuk Preprocessing',
                'message': 'Dataset siap untuk diproses. Klik "Mulai Preprocessing" untuk memulai.'
            })
        elif preprocessed_validation['total_processed'] < raw_validation['total_images']:
            recommendations.append({
                'type': 'info',
                'icon': '🔄',
                'title': 'Preprocessing Partial',
                'message': 'Sebagian dataset sudah diproses. Gunakan force reprocess untuk memproses ulang.'
            })
        
        # Performance recommendations
        total_images = raw_validation['total_images']
        if total_images > 10000:
            recommended_workers = min(8, max(4, total_images // 2500))
            recommendations.append({
                'type': 'performance',
                'icon': '⚡',
                'title': 'Optimasi Performance',
                'message': f'Dataset besar terdeteksi. Gunakan {recommended_workers} workers untuk performa optimal.'
            })
        
        # Class imbalance recommendations
        if class_analysis.get('analysis_success') and class_analysis.get('imbalance_score', 0) > 5:
            recommendations.append({
                'type': 'warning',
                'icon': '⚖️',
                'title': 'Class Imbalance Detected',
                'message': f'Ketidakseimbangan kelas tinggi (skor: {class_analysis["imbalance_score"]:.1f}). Pertimbangkan augmentasi data.'
            })
        
        # Storage recommendations
        estimated_size = total_images * 0.2  # MB
        if estimated_size > 1000:  # > 1GB
            recommendations.append({
                'type': 'info',
                'icon': '💾',
                'title': 'Storage Requirements',
                'message': f'Preprocessing akan menggunakan ~{estimated_size/1000:.1f}GB storage. Pastikan ruang tersedia.'
            })
        
        return recommendations

def get_validation_helper(ui_components: Dict[str, Any], logger=None) -> ValidationHelper:
    """Factory function untuk mendapatkan validation helper."""
    if 'validation_helper' not in ui_components:
        ui_components['validation_helper'] = ValidationHelper(ui_components, logger)
    return ui_components['validation_helper']