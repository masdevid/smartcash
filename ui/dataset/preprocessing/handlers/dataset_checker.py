"""
File: smartcash/ui/dataset/preprocessing/handlers/dataset_checker.py
Deskripsi: Handler khusus untuk check dataset dengan comprehensive analysis
"""

from typing import Dict, Any
from smartcash.ui.components.status_panel import update_status_panel
from smartcash.ui.dataset.preprocessing.utils import get_validation_helper
from smartcash.dataset.utils.path_validator import get_path_validator

def setup_dataset_checker(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Setup handler untuk check dataset dengan comprehensive analysis."""
    logger = ui_components.get('logger')
    validation_helper = get_validation_helper(ui_components, logger)
    path_validator = get_path_validator(logger)
    
    def _check_dataset_structure() -> Dict[str, Any]:
        """Comprehensive dataset structure check."""
        data_dir = ui_components.get('data_dir', 'data')
        preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
        
        # Validate raw dataset
        raw_validation = path_validator.validate_dataset_structure(data_dir)
        
        # Validate preprocessed dataset
        preprocessed_validation = path_validator.validate_preprocessed_structure(preprocessed_dir)
        
        # Analyze class distribution
        class_analysis = validation_helper.analyze_class_distribution(data_dir)
        
        # Check preprocessing compatibility
        compatibility_check = validation_helper.check_preprocessing_compatibility(raw_validation)
        
        return {
            'raw': raw_validation,
            'preprocessed': preprocessed_validation,
            'class_analysis': class_analysis,
            'compatibility': compatibility_check,
            'available_splits': path_validator.detect_available_splits(data_dir),
            'recommendations': validation_helper.generate_smart_recommendations(
                raw_validation, preprocessed_validation, class_analysis
            )
        }
    
    def _format_comprehensive_summary(check_result: Dict[str, Any]) -> str:
        """Format comprehensive dataset summary."""
        lines = ["📊 Comprehensive Dataset Analysis", "=" * 60]
        
        # Raw dataset section
        raw = check_result['raw']
        lines.append(f"\n📁 Raw Dataset: {raw['data_dir']}")
        
        if raw['valid']:
            lines.append(f"   📊 Total: {raw['total_images']:,} gambar, {raw['total_labels']:,} label")
            lines.append(f"   📂 Splits: {', '.join(check_result['available_splits'])}")
            
            # Split breakdown
            for split in ['train', 'valid', 'test']:
                split_info = raw['splits'][split]
                if split_info['exists']:
                    status = "✅" if split_info['images'] > 0 else "⚪"
                    percentage = (split_info['images'] / max(raw['total_images'], 1)) * 100
                    lines.append(f"   {status} {split}: {split_info['images']:,} gambar ({percentage:.1f}%)")
                else:
                    lines.append(f"   ❌ {split}: Tidak ditemukan")
        else:
            lines.append("   ❌ Dataset tidak ditemukan atau tidak valid")
        
        # Preprocessed dataset section
        preprocessed = check_result['preprocessed']
        lines.append(f"\n🔧 Preprocessed Dataset: {preprocessed['preprocessed_dir']}")
        
        if preprocessed['valid'] and preprocessed['total_processed'] > 0:
            lines.append(f"   📊 Total: {preprocessed['total_processed']:,} gambar terproses")
            
            for split in ['train', 'valid', 'test']:
                split_info = preprocessed['splits'][split]
                if split_info['exists'] and split_info['processed'] > 0:
                    percentage = (split_info['processed'] / max(preprocessed['total_processed'], 1)) * 100
                    lines.append(f"   ✅ {split}: {split_info['processed']:,} gambar ({percentage:.1f}%)")
                else:
                    lines.append(f"   ⚪ {split}: Belum diproses")
        else:
            lines.append("   ⚪ Belum ada data preprocessed")
        
        # Class analysis section
        class_analysis = check_result.get('class_analysis', {})
        if class_analysis.get('analysis_success'):
            lines.append("\n🏷️ Class Analysis:")
            total_classes = class_analysis.get('total_classes', 0)
            imbalance_score = class_analysis.get('imbalance_score', 0)
            lines.append(f"   📊 Jumlah kelas: {total_classes}")
            
            if imbalance_score > 5:
                lines.append(f"   ⚠️ Ketidakseimbangan: {imbalance_score:.1f}/10 (tinggi)")
            else:
                lines.append(f"   ✅ Keseimbangan: {imbalance_score:.1f}/10 (baik)")
        
        # Compatibility section
        compatibility = check_result.get('compatibility', {})
        lines.append("\n🔄 Preprocessing Compatibility:")
        
        if compatibility.get('ready_for_preprocessing', False):
            lines.append("   ✅ Siap untuk preprocessing")
            if compatibility['estimated_time'] > 0:
                lines.append(f"   ⏱️ Estimasi waktu: {compatibility['estimated_time']:.1f} detik")
            if compatibility['estimated_output_size'] > 0:
                lines.append(f"   💾 Estimasi output: {compatibility['estimated_output_size']:.1f} MB")
        else:
            lines.append("   ❌ Tidak siap untuk preprocessing")
            for issue in compatibility.get('blocking_issues', []):
                lines.append(f"       • {issue}")
        
        # Issues section
        all_issues = raw['issues'] + compatibility.get('warnings', [])
        if all_issues:
            lines.append(f"\n⚠️ Issues & Warnings ({len(all_issues)}):")
            for issue in all_issues:
                lines.append(f"   • {issue}")
        
        # Smart recommendations
        recommendations = check_result.get('recommendations', [])
        if recommendations:
            lines.append(f"\n💡 Smart Recommendations ({len(recommendations)}):")
            for rec in recommendations:
                lines.append(f"   {rec['icon']} {rec['title']}: {rec['message']}")
        
    
    def _on_check_dataset_click(b):
        """Handler untuk tombol check dataset dengan comprehensive analysis."""
        logger and logger.info("🔍 Memulai comprehensive dataset analysis...")
        update_status_panel(ui_components['status_panel'], "Menganalisis struktur dataset...", "info")
        
        try:
            # Comprehensive dataset check
            check_result = _check_dataset_structure()
            summary = _format_comprehensive_summary(check_result)
            
            logger and logger.info(summary)
            
            # Smart status update berdasarkan compatibility
            compatibility = check_result.get('compatibility', {})
            raw = check_result['raw']
            
            if not raw['valid']:
                status_msg = "Dataset tidak ditemukan - silakan download terlebih dahulu"
                update_status_panel(ui_components['status_panel'], status_msg, "error")
            elif not compatibility.get('ready_for_preprocessing', False):
                blocking_count = len(compatibility.get('blocking_issues', []))
                status_msg = f"Dataset ditemukan dengan {blocking_count} blocking issues"
                update_status_panel(ui_components['status_panel'], status_msg, "error")
            elif compatibility.get('warnings'):
                warning_count = len(compatibility['warnings'])
                status_msg = f"Dataset siap dengan {warning_count} warnings"
                update_status_panel(ui_components['status_panel'], status_msg, "warning")
            else:
                total_images = raw['total_images']
                estimated_time = compatibility.get('estimated_time', 0)
                status_msg = f"Dataset siap: {total_images:,} gambar, estimasi {estimated_time:.1f}s"
                update_status_panel(ui_components['status_panel'], status_msg, "success")
                
        except Exception as e:
            error_msg = f"Error analyzing dataset: {str(e)}"
            logger and logger.error(f"❌ {error_msg}")
            update_status_panel(ui_components['status_panel'], error_msg, "error")
    
    # Setup event handler
    ui_components['check_button'].on_click(_on_check_dataset_click)
    
    logger and logger.debug("✅ Dataset checker setup selesai")
    
    return ui_components