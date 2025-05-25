"""
File: smartcash/ui/dataset/preprocessing/utils/validation_helper.py
Deskripsi: Enhanced validation helper dengan symlink detection dan path validation yang konsisten
"""

from typing import Dict, Any, Tuple, List
from pathlib import Path
from smartcash.dataset.utils.path_validator import get_path_validator

class ValidationHelper:
    """Enhanced validation helper dengan symlink detection dan consistent path validation."""
    
    def __init__(self, ui_components: Dict[str, Any], logger=None):
        self.ui_components = ui_components
        self.logger = logger
        self.path_validator = get_path_validator(logger)
    
    def check_source_dataset(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Check source dataset di /data/{train,valid,test} dengan detailed info."""
        data_dir = ui_components.get('data_dir', 'data')
        validation_result = self.path_validator.validate_dataset_structure(data_dir)
        
        if not validation_result['valid']:
            return False, f"Source dataset tidak ditemukan: {data_dir}", validation_result
        
        # Check critical issues
        critical_issues = [i for i in validation_result['issues'] if '❌' in i]
        if critical_issues:
            return False, f"Critical issues di source: {', '.join(critical_issues[:2])}", validation_result
        
        if validation_result['total_images'] == 0:
            return False, "Source dataset kosong, tidak ada gambar ditemukan", validation_result
        
        success_msg = f"Source dataset valid: {validation_result['total_images']:,} gambar di {len(self.path_validator.detect_available_splits(data_dir))} split"
        return True, success_msg, validation_result
    
    def check_preprocessed_dataset(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Check preprocessed dataset dengan enhanced symlink detection."""
        preprocessed_dir = self.ui_components.get('preprocessed_dir', 'data/preprocessed')
        validation_result = self.path_validator.validate_preprocessed_structure(preprocessed_dir)
        
        if not validation_result['valid']:
            return False, f"Preprocessed directory tidak ditemukan: {preprocessed_dir}", validation_result
        
        if validation_result['total_processed'] == 0:
            return False, "Belum ada data preprocessing", validation_result
        
        # Enhanced info dengan symlink detection
        available_splits = []
        symlink_info = self._detect_symlinks_in_splits(Path(preprocessed_dir))
        
        for split, info in validation_result['splits'].items():
            if info['processed'] > 0:
                split_symlinks = symlink_info.get(split, {}).get('total_symlinks', 0)
                if split_symlinks > 0:
                    available_splits.append(f"{split}({info['processed']:,}, {split_symlinks} symlinks)")
                else:
                    available_splits.append(f"{split}({info['processed']:,})")
        
        # Add symlink info ke validation result
        validation_result['symlink_info'] = symlink_info
        
        success_msg = f"Preprocessed dataset: {validation_result['total_processed']:,} gambar di {len(available_splits)} split"
        return True, success_msg, validation_result
    
    def _detect_symlinks_in_splits(self, preprocessed_dir: Path) -> Dict[str, Dict[str, Any]]:
        """Detect symlinks per split dengan detailed categorization."""
        symlink_info = {}
        
        try:
            if not preprocessed_dir.exists():
                return symlink_info
            
            for split in ['train', 'valid', 'test']:
                split_path = preprocessed_dir / split
                split_info = {
                    'total_symlinks': 0,
                    'augmentation_symlinks': 0,
                    'regular_symlinks': 0,
                    'broken_symlinks': 0,
                    'symlink_targets': []
                }
                
                if split_path.exists():
                    for item in split_path.rglob('*'):
                        if item.is_symlink():
                            split_info['total_symlinks'] += 1
                            
                            try:
                                target = item.resolve()
                                target_str = str(target).lower()
                                
                                # Categorize symlinks
                                if any(pattern in target_str for pattern in ['augment', 'synthetic', 'generated']):
                                    split_info['augmentation_symlinks'] += 1
                                    split_info['symlink_targets'].append({
                                        'source': item.name,
                                        'target': str(target),
                                        'type': 'augmentation'
                                    })
                                else:
                                    split_info['regular_symlinks'] += 1
                                    split_info['symlink_targets'].append({
                                        'source': item.name,
                                        'target': str(target),
                                        'type': 'regular'
                                    })
                            except Exception:
                                # Broken symlink
                                split_info['broken_symlinks'] += 1
                                split_info['symlink_targets'].append({
                                    'source': item.name,
                                    'target': 'broken',
                                    'type': 'broken'
                                })
                
                symlink_info[split] = split_info
                
        except Exception as e:
            self.logger and self.logger.debug(f"🔗 Symlink detection error: {str(e)}")
        
        return symlink_info
    
    def check_dataset_exists(self) -> Tuple[bool, str]:
        """Check dataset existence dengan focus pada source dataset."""
        source_valid, source_msg, _ = self.check_source_dataset()
        return source_valid, source_msg
    
    def check_preprocessed_data(self) -> Tuple[bool, list, str, int]:
        """Check preprocessed data dengan enhanced symlink info untuk cleanup operations."""
        preprocessed_valid, preprocessed_msg, validation_result = self.check_preprocessed_dataset()
        
        if not preprocessed_valid:
            return False, [], preprocessed_msg, 0
        
        available_splits = []
        total_files = validation_result['total_processed']
        symlink_info = validation_result.get('symlink_info', {})
        
        for split, info in validation_result['splits'].items():
            if info['processed'] > 0:
                split_symlinks = symlink_info.get(split, {}).get('total_symlinks', 0)
                aug_symlinks = symlink_info.get(split, {}).get('augmentation_symlinks', 0)
                
                if split_symlinks > 0:
                    if aug_symlinks > 0:
                        available_splits.append(f"{split} ({info['processed']:,} files, {aug_symlinks} aug symlinks)")
                    else:
                        available_splits.append(f"{split} ({info['processed']:,} files, {split_symlinks} symlinks)")
                else:
                    available_splits.append(f"{split} ({info['processed']:,} files)")
        
        return True, available_splits, f"Total {total_files:,} files", total_files
    
    def check_existing_preprocessed_for_split(self, split_config: str) -> List[str]:
        """Check existing preprocessed data untuk split tertentu dengan consistent val->valid mapping."""
        _, _, validation_result = self.check_preprocessed_dataset()
        existing_data = []
        
        if split_config == 'all':
            for split in ['train', 'valid', 'test']:
                if validation_result['splits'].get(split, {}).get('processed', 0) > 0:
                    existing_data.append(split)
        else:
            # Consistent val->valid mapping
            check_split = 'valid' if split_config == 'val' else split_config
            if validation_result['splits'].get(check_split, {}).get('processed', 0) > 0:
                existing_data.append(check_split)
        
        return existing_data
    
    def get_symlink_safety_info(self) -> Dict[str, Any]:
        """Get informasi keamanan symlink untuk cleanup operations."""
        _, _, validation_result = self.check_preprocessed_dataset()
        symlink_info = validation_result.get('symlink_info', {})
        
        safety_info = {
            'total_symlinks': 0,
            'augmentation_symlinks': 0,
            'safe_to_cleanup': True,
            'symlink_warning': '',
            'split_details': {}
        }
        
        for split, split_info in symlink_info.items():
            safety_info['total_symlinks'] += split_info.get('total_symlinks', 0)
            safety_info['augmentation_symlinks'] += split_info.get('augmentation_symlinks', 0)
            safety_info['split_details'][split] = split_info
        
        # Generate warning message
        if safety_info['augmentation_symlinks'] > 0:
            safety_info['symlink_warning'] = (
                f"⚠️ {safety_info['augmentation_symlinks']} symlink augmentasi terdeteksi. "
                "Symlink akan dihapus, tetapi data augmentasi asli tetap aman."
            )
        elif safety_info['total_symlinks'] > 0:
            safety_info['symlink_warning'] = (
                f"🔗 {safety_info['total_symlinks']} symlink terdeteksi dan akan dihapus."
            )
        
        return safety_info
    
    def get_dataset_comparison(self) -> Dict[str, Any]:
        """Get comprehensive comparison antara source dan preprocessed dataset."""
        source_valid, source_msg, source_data = self.check_source_dataset()
        preprocessed_valid, preprocessed_msg, preprocessed_data = self.check_preprocessed_dataset()
        
        comparison = {
            'source': {
                'valid': source_valid,
                'message': source_msg,
                'data': source_data,
                'path': self.ui_components.get('data_dir', 'data')
            },
            'preprocessed': {
                'valid': preprocessed_valid,
                'message': preprocessed_msg,
                'data': preprocessed_data,
                'path': self.ui_components.get('preprocessed_dir', 'data/preprocessed')
            }
        }
        
        # Add comparison metrics dengan symlink awareness
        if source_valid and preprocessed_valid:
            source_count = source_data['total_images']
            processed_count = preprocessed_data['total_processed']
            symlink_count = sum(
                split_info.get('total_symlinks', 0) 
                for split_info in preprocessed_data.get('symlink_info', {}).values()
            )
            
            comparison['metrics'] = {
                'source_images': source_count,
                'processed_images': processed_count,
                'symlink_count': symlink_count,
                'processing_percentage': (processed_count / source_count * 100) if source_count > 0 else 0,
                'needs_processing': processed_count < source_count,
                'processing_complete': processed_count >= source_count,
                'has_symlinks': symlink_count > 0
            }
        
        return comparison
    
    def analyze_class_distribution(self, data_dir: str) -> Dict[str, Any]:
        """Analyze class distribution dengan enhanced error handling."""
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
    
    def check_preprocessing_compatibility(self, source_validation: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced preprocessing compatibility check dengan symlink consideration."""
        compatibility = {
            'ready_for_preprocessing': True,
            'blocking_issues': [],
            'warnings': [],
            'estimated_time': 0,
            'estimated_output_size': 0,
            'source_path': self.ui_components.get('data_dir', 'data'),
            'preprocessed_path': self.ui_components.get('preprocessed_dir', 'data/preprocessed'),
            'symlink_considerations': []
        }
        
        # Check blocking issues from source
        if not source_validation['valid']:
            compatibility['ready_for_preprocessing'] = False
            compatibility['blocking_issues'].append("Source dataset tidak ditemukan atau tidak valid")
            return compatibility
        
        critical_issues = [i for i in source_validation['issues'] if '❌' in i]
        if critical_issues:
            compatibility['ready_for_preprocessing'] = False
            compatibility['blocking_issues'].extend(critical_issues)
        
        # Check warnings
        warning_issues = [i for i in source_validation['issues'] if '⚠️' in i]
        compatibility['warnings'].extend(warning_issues)
        
        # Enhanced estimates berdasarkan source data
        total_images = source_validation['total_images']
        if total_images > 0:
            # Improved time estimation
            base_time_per_image = 0.1  # seconds per image base
            if total_images > 5000:
                base_time_per_image = 0.08  # Faster untuk batch besar
            elif total_images < 1000:
                base_time_per_image = 0.15  # Slower untuk batch kecil
            
            compatibility['estimated_time'] = max(1, total_images * base_time_per_image)
            compatibility['estimated_output_size'] = total_images * 0.25  # ~0.25MB per processed image
        
        # Check existing symlinks
        symlink_safety = self.get_symlink_safety_info()
        if symlink_safety['total_symlinks'] > 0:
            compatibility['symlink_considerations'].append(
                f"🔗 {symlink_safety['total_symlinks']} symlink existing akan dipertahankan"
            )
            
            if symlink_safety['augmentation_symlinks'] > 0:
                compatibility['symlink_considerations'].append(
                    f"📎 {symlink_safety['augmentation_symlinks']} augmentation symlinks terdeteksi"
                )
        
        return compatibility
    
    def generate_smart_recommendations(self, source_data: Dict[str, Any], 
                                     preprocessed_data: Dict[str, Any],
                                     class_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate smart recommendations dengan symlink awareness."""
        recommendations = []
        
        # Source dataset recommendations
        if not source_data.get('valid', False):
            recommendations.append({
                'type': 'critical',
                'icon': '📥',
                'title': 'Download Dataset Required',
                'message': 'Download dataset source terlebih dahulu dari Roboflow sebelum preprocessing',
                'priority': 1
            })
            return recommendations
        
        source_images = source_data.get('total_images', 0)
        processed_images = preprocessed_data.get('total_processed', 0)
        symlink_info = preprocessed_data.get('symlink_info', {})
        
        # Symlink-aware recommendations
        total_symlinks = sum(split_info.get('total_symlinks', 0) for split_info in symlink_info.values())
        aug_symlinks = sum(split_info.get('augmentation_symlinks', 0) for split_info in symlink_info.values())
        
        if total_symlinks > 0:
            if aug_symlinks > 0:
                recommendations.append({
                    'type': 'info',
                    'icon': '🔗',
                    'title': 'Augmentation Symlinks Detected',
                    'message': f'{aug_symlinks} augmentation symlinks terdeteksi. Data original aman saat cleanup.',
                    'priority': 2
                })
            
            recommendations.append({
                'type': 'info',
                'icon': '📎',
                'title': 'Symlink Management',
                'message': f'{total_symlinks} total symlinks akan dikelola dengan aman selama operasi.',
                'priority': 3
            })
        
        # Preprocessing status recommendations
        if processed_images == 0:
            recommendations.append({
                'type': 'action',
                'icon': '🚀',
                'title': 'Siap untuk Preprocessing',
                'message': f'Dataset source siap dengan {source_images:,} gambar. Klik "Mulai Preprocessing" untuk memulai.',
                'priority': 4
            })
        elif processed_images < source_images:
            processing_pct = (processed_images / source_images) * 100
            recommendations.append({
                'type': 'info',
                'icon': '🔄',
                'title': 'Preprocessing Partial',
                'message': f'Dataset {processing_pct:.1f}% sudah diproses ({processed_images:,}/{source_images:,}). Gunakan force reprocess untuk memproses ulang.',
                'priority': 5
            })
        else:
            recommendations.append({
                'type': 'success',
                'icon': '✅',
                'title': 'Preprocessing Complete',
                'message': f'Semua {source_images:,} gambar sudah diproses. Siap untuk augmentasi atau training.',
                'priority': 6
            })
        
        # Performance recommendations untuk dataset besar
        if source_images > 10000:
            recommended_workers = min(8, max(4, source_images // 2500))
            recommendations.append({
                'type': 'performance',
                'icon': '⚡',
                'title': 'Optimasi Performance',
                'message': f'Dataset besar terdeteksi ({source_images:,} gambar). Gunakan {recommended_workers} workers untuk performa optimal.',
                'priority': 7
            })
        
        # Class imbalance recommendations
        if class_analysis.get('analysis_success') and class_analysis.get('imbalance_score', 0) > 5:
            recommendations.append({
                'type': 'warning',
                'icon': '⚖️',
                'title': 'Class Imbalance Detected',
                'message': f'Ketidakseimbangan kelas tinggi (skor: {class_analysis["imbalance_score"]:.1f}). Pertimbangkan augmentasi data.',
                'priority': 8
            })
        
        # Storage recommendations
        estimated_size = source_images * 0.25  # MB
        if estimated_size > 1000:  # > 1GB
            recommendations.append({
                'type': 'info',
                'icon': '💾',
                'title': 'Storage Requirements',
                'message': f'Preprocessing akan menggunakan ~{estimated_size/1000:.1f}GB storage tambahan. Pastikan ruang tersedia.',
                'priority': 9
            })
        
        # Sort by priority
        recommendations.sort(key=lambda x: x.get('priority', 10))
        
        return recommendations

def get_validation_helper(ui_components: Dict[str, Any], logger=None) -> ValidationHelper:
    """Factory function untuk mendapatkan enhanced validation helper dengan symlink support."""
    if 'validation_helper' not in ui_components:
        ui_components['validation_helper'] = ValidationHelper(ui_components, logger)
    return ui_components['validation_helper']