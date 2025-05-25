"""
File: smartcash/dataset/preprocessor/operations/dataset_checker.py  
Deskripsi: Fixed dataset checker dengan proper error handling untuk 'images' key issue
"""

from typing import Dict, Any, List
from pathlib import Path

from smartcash.common.logger import get_logger
from smartcash.dataset.utils.path_validator import get_path_validator


class DatasetChecker:
    """Fixed dataset checker dengan comprehensive error handling."""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        """Initialize dataset checker dengan configuration."""
        self.config = config
        self.logger = logger or get_logger()
        self.path_validator = get_path_validator(logger)
        
    def check_source_dataset(self, detailed: bool = True) -> Dict[str, Any]:
        """Check source dataset dengan comprehensive validation."""
        data_dir = self.config.get('data', {}).get('dir', 'data')
        
        try:
            # Basic structure validation
            structure_result = self.path_validator.validate_dataset_structure(data_dir)
            
            if not structure_result['valid']:
                return {
                    'valid': False,
                    'status': 'invalid',
                    'message': f'Source dataset tidak valid: {data_dir}',
                    'data_dir': data_dir,
                    'issues': structure_result['issues']
                }
            
            # Enhanced analysis jika detailed
            if detailed:
                analysis_result = self.analyze_dataset_structure(structure_result)
                structure_result.update(analysis_result)
            
            # Generate summary report
            report = self.generate_dataset_report(structure_result, 'source')
            
            return {
                'valid': True,
                'status': 'valid',
                'message': f"Source dataset valid: {structure_result['total_images']} gambar",
                'data_dir': data_dir,
                'total_images': structure_result['total_images'],
                'total_labels': structure_result['total_labels'],
                'splits': structure_result['splits'],
                'analysis': structure_result.get('analysis', {}),
                'report': report,
                'issues': structure_result['issues']
            }
            
        except Exception as e:
            return {
                'valid': False,
                'status': 'error',
                'message': f'Error checking source dataset: {str(e)}',
                'data_dir': data_dir
            }
    
    def check_preprocessed_dataset(self, detailed: bool = True) -> Dict[str, Any]:
        """FIXED: Check preprocessed dataset dengan proper error handling."""
        preprocessed_dir = self.config.get('preprocessing', {}).get('output_dir', 'data/preprocessed')
        
        try:
            # Validate preprocessed structure
            structure_result = self.path_validator.validate_preprocessed_structure(preprocessed_dir)
            
            if not structure_result.get('valid', False):
                return {
                    'valid': False,
                    'status': 'empty',
                    'message': 'Belum ada data preprocessed',
                    'preprocessed_dir': preprocessed_dir,
                    'total_processed': 0
                }
            
            total_processed = structure_result.get('total_processed', 0)
            if total_processed == 0:
                return {
                    'valid': False,
                    'status': 'empty', 
                    'message': 'Belum ada data preprocessed',
                    'preprocessed_dir': preprocessed_dir,
                    'total_processed': 0
                }
            
            # Enhanced analysis untuk preprocessed data
            if detailed:
                analysis_result = self._analyze_preprocessed_structure(structure_result)
                structure_result.update(analysis_result)
            
            # Generate report
            report = self.generate_dataset_report(structure_result, 'preprocessed')
            
            return {
                'valid': True,
                'status': 'available',
                'message': f"Preprocessed dataset: {total_processed} gambar",
                'preprocessed_dir': preprocessed_dir,
                'total_processed': total_processed,
                'splits': structure_result['splits'],
                'analysis': structure_result.get('analysis', {}),
                'report': report
            }
            
        except Exception as e:
            return {
                'valid': False,
                'status': 'error',
                'message': f'Error checking preprocessed dataset: {str(e)}',
                'preprocessed_dir': preprocessed_dir
            }
    
    def analyze_dataset_structure(self, structure_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze dataset structure dengan detailed metrics."""
        analysis = {
            'split_distribution': {},
            'image_label_ratio': {},
            'quality_indicators': {},
            'recommendations': []
        }
        
        total_images = structure_result.get('total_images', 0)
        total_labels = structure_result.get('total_labels', 0)
        splits = structure_result.get('splits', {})
        
        # Split distribution analysis
        for split, split_data in splits.items():
            if split_data.get('exists', False):
                # FIXED: Safe access to images count
                images_count = split_data.get('images', 0)
                if images_count > 0:
                    distribution_pct = (images_count / total_images) * 100 if total_images > 0 else 0
                    analysis['split_distribution'][split] = {
                        'count': images_count,
                        'percentage': round(distribution_pct, 1)
                    }
                    
                    # Image-label ratio per split
                    labels_count = split_data.get('labels', 0)
                    ratio = labels_count / images_count if images_count > 0 else 0
                    analysis['image_label_ratio'][split] = round(ratio, 2)
        
        # Quality indicators
        analysis['quality_indicators'] = {
            'total_completeness': round((total_labels / total_images) * 100, 1) if total_images > 0 else 0,
            'split_balance': self._calculate_split_balance(analysis['split_distribution']),
            'labeling_consistency': self._assess_labeling_consistency(analysis['image_label_ratio'])
        }
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_analysis_recommendations(analysis)
        
        return {'analysis': analysis}
    
    def generate_dataset_report(self, structure_result: Dict[str, Any], dataset_type: str) -> str:
        """Generate comprehensive dataset report."""
        lines = [f"📋 {dataset_type.title()} Dataset Report", "=" * 50]
        
        if dataset_type == 'source':
            total_key = 'total_images'
            lines.append(f"\n📁 Dataset Directory: {structure_result.get('data_dir', 'N/A')}")
        else:
            total_key = 'total_processed'
            lines.append(f"\n📁 Preprocessed Directory: {structure_result.get('preprocessed_dir', 'N/A')}")
        
        total_count = structure_result.get(total_key, 0)
        lines.append(f"📊 Total Images: {total_count:,}")
        
        # Split details dengan safe access
        lines.append(f"\n📂 Split Breakdown:")
        splits = structure_result.get('splits', {})
        for split in ['train', 'valid', 'test']:
            split_data = splits.get(split, {})
            if split_data.get('exists', False):
                if dataset_type == 'source':
                    count = split_data.get('images', 0)
                    labels = split_data.get('labels', 0)
                    lines.append(f"   • {split}: {count:,} gambar, {labels:,} label")
                else:
                    count = split_data.get('processed', 0)
                    lines.append(f"   • {split}: {count:,} processed")
            else:
                lines.append(f"   • {split}: ❌ Tidak tersedia")
        
        return "\n".join(lines)
    
    def _analyze_preprocessed_structure(self, structure_result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced analysis untuk preprocessed dataset."""
        analysis = {
            'processing_distribution': {},
            'symlink_analysis': structure_result.get('symlink_analysis', {}),
            'recommendations': []
        }
        
        total_processed = structure_result.get('total_processed', 0)
        splits = structure_result.get('splits', {})
        
        # Processing distribution
        for split, split_data in splits.items():
            if split_data.get('exists', False):
                processed_count = split_data.get('processed', 0)
                if processed_count > 0:
                    distribution_pct = (processed_count / total_processed) * 100
                    analysis['processing_distribution'][split] = {
                        'count': processed_count,
                        'percentage': round(distribution_pct, 1)
                    }
        
        return {'analysis': analysis}
    
    def _calculate_split_balance(self, split_distribution: Dict[str, Dict[str, Any]]) -> str:
        """Calculate balance score untuk split distribution."""
        if not split_distribution:
            return "No Data"
        
        percentages = [split_data['percentage'] for split_data in split_distribution.values()]
        
        if len(percentages) > 1:
            mean_pct = sum(percentages) / len(percentages)
            variance = sum((p - mean_pct) ** 2 for p in percentages) / len(percentages)
            cv = (variance ** 0.5) / mean_pct if mean_pct > 0 else 0
            
            if cv < 0.2:
                return "Excellent"
            elif cv < 0.4:
                return "Good"
            elif cv < 0.6:
                return "Fair"
            else:
                return "Poor"
        
        return "Single Split"
    
    def _assess_labeling_consistency(self, image_label_ratios: Dict[str, float]) -> str:
        """Assess labeling consistency across splits."""
        if not image_label_ratios:
            return "No Data"
        
        ratios = list(image_label_ratios.values())
        ideal_ratios = [abs(r - 1.0) for r in ratios]
        avg_deviation = sum(ideal_ratios) / len(ideal_ratios)
        
        if avg_deviation < 0.05:
            return "Excellent"
        elif avg_deviation < 0.15:
            return "Good"
        elif avg_deviation < 0.3:
            return "Fair"
        else:
            return "Poor"
    
    def _generate_analysis_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations berdasarkan analysis results."""
        recommendations = []
        
        split_balance = analysis['quality_indicators'].get('split_balance', '')
        if split_balance in ['Fair', 'Poor']:
            recommendations.append("Pertimbangkan rebalancing split dataset")
        
        labeling_consistency = analysis['quality_indicators'].get('labeling_consistency', '')
        if labeling_consistency in ['Fair', 'Poor']:
            recommendations.append("Periksa konsistensi labeling")
        
        completeness = analysis['quality_indicators'].get('total_completeness', 0)
        if completeness < 90:
            recommendations.append("Dataset memiliki missing labels")
        
        return recommendations