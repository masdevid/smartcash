"""
File: smartcash/ui/dataset/preprocessing/handlers/check_dataset_handler.py
Deskripsi: Fixed handler untuk check dataset dengan path validator dan val->valid mapping
"""

from typing import Dict, Any
from smartcash.ui.components.status_panel import update_status_panel
from smartcash.dataset.utils.path_validator import get_path_validator

__all__ = ['setup_check_dataset_handler']


def setup_check_dataset_handler(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Setup handler untuk check dataset dengan fixed path validation."""
    logger = ui_components.get('logger')
    path_validator = get_path_validator(logger)
    
    def _check_dataset_structure() -> Dict[str, Any]:
        """Check struktur dataset menggunakan path validator."""
        data_dir = ui_components.get('data_dir', 'data')
        preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
        
        # Validate raw dataset
        raw_validation = path_validator.validate_dataset_structure(data_dir)
        
        # Validate preprocessed dataset  
        preprocessed_validation = path_validator.validate_preprocessed_structure(preprocessed_dir)
        
        # Analisis kelas jika ada data valid
        class_analysis = {}
        if raw_validation['total_images'] > 0:
            class_analysis = _analyze_class_distribution(data_dir)
        
        return {
            'raw': raw_validation,
            'preprocessed': preprocessed_validation,
            'class_analysis': class_analysis,
            'available_splits': path_validator.detect_available_splits(data_dir)
        }
    
    def _analyze_class_distribution(data_dir: str) -> Dict[str, Any]:
        """Analisis distribusi kelas menggunakan explorer service."""
        try:
            from smartcash.dataset.services.explorer.class_explorer import ClassExplorer
            
            config = {
                'data': {'dir': data_dir},
                'classes': {}
            }
            
            # Fixed constructor call - 4 parameters only
            explorer = ClassExplorer(config, data_dir, logger, 4)  # num_workers=4
            class_result = explorer.analyze_distribution('train')
            
            if class_result.get('status') == 'success':
                return {
                    'total_classes': class_result.get('class_count', 0),
                    'class_counts': class_result.get('counts', {}),
                    'imbalance_score': class_result.get('imbalance_score', 0)
                }
        except Exception as e:
            if logger:
                logger.debug(f"🔍 Error analisis kelas: {str(e)}")
        
        return {}
    
    def _format_dataset_summary(check_result: Dict[str, Any]) -> str:
        """Format hasil check menjadi summary yang informatif."""
        lines = ["📊 Dataset Check Summary", "=" * 50]
        
        # Raw dataset info
        raw = check_result['raw']
        lines.append(f"\n📁 Raw Dataset: {raw['data_dir']}")
        
        if raw['valid']:
            lines.append(f"   📊 Total: {raw['total_images']:,} gambar, {raw['total_labels']:,} label")
            lines.append(f"   📂 Splits tersedia: {', '.join(check_result['available_splits'])}")
            
            # Per split breakdown
            for split in ['train', 'valid', 'test']:
                split_info = raw['splits'][split]
                if split_info['exists']:
                    status = "✅" if split_info['images'] > 0 else "⚪"
                    percentage = (split_info['images'] / max(raw['total_images'], 1)) * 100
                    lines.append(f"   {status} {split}: {split_info['images']:,} gambar ({percentage:.1f}%)")
                else:
                    lines.append(f"   ❌ {split}: Tidak ditemukan")
            
            # Class analysis
            class_analysis = check_result.get('class_analysis', {})
            if class_analysis:
                total_classes = class_analysis.get('total_classes', 0)
                imbalance_score = class_analysis.get('imbalance_score', 0)
                lines.append(f"   🏷️ Kelas: {total_classes} jenis")
                
                if imbalance_score > 5:
                    lines.append(f"   ⚠️ Ketidakseimbangan: {imbalance_score:.1f}/10 (tinggi)")
                else:
                    lines.append(f"   ✅ Keseimbangan: {imbalance_score:.1f}/10 (baik)")
        else:
            lines.append("   ❌ Dataset tidak ditemukan")
        
        # Preprocessed dataset info
        preprocessed = check_result['preprocessed']
        lines.append(f"\n🔧 Preprocessed Dataset: {preprocessed['preprocessed_dir']}")
        
        if preprocessed['valid']:
            lines.append(f"   📊 Total: {preprocessed['total_processed']:,} gambar terproses")
            
            for split in ['train', 'valid', 'test']:
                split_info = preprocessed['splits'][split]
                if split_info['exists'] and split_info['processed'] > 0:
                    status = "✅"
                    percentage = (split_info['processed'] / max(preprocessed['total_processed'], 1)) * 100
                    lines.append(f"   {status} {split}: {split_info['processed']:,} gambar ({percentage:.1f}%)")
                else:
                    lines.append(f"   ⚪ {split}: Belum diproses")
        else:
            lines.append("   ⚪ Belum ada data preprocessed")
        
        # Issues
        if raw['issues']:
            lines.append(f"\n⚠️ Issues ({len(raw['issues'])}):")
            for issue in raw['issues']:
                lines.append(f"   • {issue}")
        else:
            lines.append("\n✅ Tidak ada issues ditemukan")
        
        # Recommendations
        lines.append("\n💡 Rekomendasi:")
        if not raw['valid']:
            lines.append("   • 📥 Download dataset terlebih dahulu")
        elif raw['issues']:
            critical_issues = [i for i in raw['issues'] if '❌' in i]
            if critical_issues:
                lines.append("   • 🔧 Perbaiki critical issues sebelum preprocessing")
            else:
                lines.append("   • ⚠️ Dataset masih bisa diproses meski ada warnings")
        elif preprocessed['total_processed'] == 0:
            lines.append("   • 🚀 Dataset siap untuk preprocessing")
        else:
            lines.append("   • 🔄 Gunakan force reprocess jika perlu update")
        
        # Performance recommendations
        if raw['total_images'] > 10000:
            recommended_workers = min(8, max(4, raw['total_images'] // 2500))
            lines.append(f"   • ⚡ Dataset besar, gunakan {recommended_workers} workers optimal")
        
        return "\n".join(lines)
    
    def _on_check_dataset_click(b):
        """Handler untuk tombol check dataset."""
        if logger:
            logger.info("🔍 Memulai pengecekan dataset...")
        
        update_status_panel(ui_components['status_panel'], "Memeriksa struktur dataset...", "info")
        
        try:
            # Comprehensive dataset check
            check_result = _check_dataset_structure()
            summary = _format_dataset_summary(check_result)
            
            if logger:
                logger.info(summary)
            
            # Smart status update
            raw = check_result['raw']
            critical_issues = [i for i in raw['issues'] if '❌' in i]
            
            if critical_issues:
                status_msg = f"Dataset ditemukan dengan {len(critical_issues)} critical issues"
                update_status_panel(ui_components['status_panel'], status_msg, "error")
            elif raw['issues']:
                status_msg = f"Dataset valid dengan {len(raw['issues'])} warnings"
                update_status_panel(ui_components['status_panel'], status_msg, "warning")
            elif raw['valid'] and raw['total_images'] > 0:
                status_msg = f"Dataset siap: {raw['total_images']:,} gambar tersedia"
                update_status_panel(ui_components['status_panel'], status_msg, "success")
            else:
                update_status_panel(ui_components['status_panel'], "Dataset tidak ditemukan", "error")
                
        except Exception as e:
            error_msg = f"Error check dataset: {str(e)}"
            if logger:
                logger.error(f"❌ {error_msg}")
            update_status_panel(ui_components['status_panel'], error_msg, "error")
    
    # Setup event handler
    ui_components['check_button'].on_click(_on_check_dataset_click)
    
    if logger:
        logger.debug("✅ Check dataset handler setup selesai")
    
    return ui_components