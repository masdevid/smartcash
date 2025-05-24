"""
File: smartcash/ui/dataset/preprocessing/handlers/dataset_checker.py
Deskripsi: Fixed lightweight dataset checker tanpa class analysis yang berat
"""

from typing import Dict, Any
from smartcash.ui.components.status_panel import update_status_panel
from smartcash.ui.dataset.preprocessing.utils import get_validation_helper, get_ui_state_manager
from smartcash.dataset.utils.path_validator import get_path_validator

def setup_dataset_checker(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Setup handler untuk check dataset dengan lightweight validation."""
    logger = ui_components.get('logger')
    validation_helper = get_validation_helper(ui_components, logger)
    ui_state = get_ui_state_manager(ui_components)
    path_validator = get_path_validator(logger)
    
    def _lightweight_dataset_check() -> Dict[str, Any]:
        """Lightweight dataset structure check tanpa class analysis."""
        data_dir = ui_components.get('data_dir', 'data')
        preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
        
        # Validate raw dataset (fast)
        raw_validation = path_validator.validate_dataset_structure(data_dir)
        
        # Validate preprocessed dataset (fast)
        preprocessed_validation = path_validator.validate_preprocessed_structure(preprocessed_dir)
        
        # Skip class analysis - terlalu lambat untuk quick check
        
        # Check preprocessing readiness (fast)
        compatibility_check = validation_helper.check_preprocessing_compatibility(raw_validation)
        
        return {
            'raw': raw_validation,
            'preprocessed': preprocessed_validation,
            'compatibility': compatibility_check,
            'available_splits': path_validator.detect_available_splits(data_dir)
        }
    
    def _format_quick_summary(check_result: Dict[str, Any]) -> str:
        """Format quick dataset summary untuk fast reporting."""
        lines = ["📊 Quick Dataset Check", "=" * 40]
        
        # Raw dataset section (essential info only)
        raw = check_result['raw']
        lines.append(f"\n📁 Raw Dataset: {raw['data_dir']}")
        
        if raw['valid']:
            lines.append(f"   ✅ Total: {raw['total_images']:,} gambar, {raw['total_labels']:,} label")
            lines.append(f"   📂 Splits: {', '.join(check_result['available_splits'])}")
            
            # Quick split status
            for split in ['train', 'valid', 'test']:
                split_info = raw['splits'][split]
                if split_info['exists'] and split_info['images'] > 0:
                    lines.append(f"   ✅ {split}: {split_info['images']:,} gambar")
        else:
            lines.append("   ❌ Dataset tidak ditemukan - silakan download terlebih dahulu")
        
        # Preprocessed status (quick check)
        preprocessed = check_result['preprocessed']
        lines.append(f"\n🔧 Preprocessed Status:")
        
        if preprocessed['valid'] and preprocessed['total_processed'] > 0:
            lines.append(f"   ✅ Tersedia: {preprocessed['total_processed']:,} gambar terproses")
        else:
            lines.append("   ⚪ Belum ada - siap untuk preprocessing")
        
        # Next steps recommendation
        compatibility = check_result.get('compatibility', {})
        lines.append(f"\n🎯 Status & Next Steps:")
        
        if not raw['valid']:
            lines.append("   📥 Download dataset terlebih dahulu")
        elif not compatibility.get('ready_for_preprocessing', False):
            blocking_issues = compatibility.get('blocking_issues', [])
            lines.append(f"   ⚠️ {len(blocking_issues)} blocking issues ditemukan")
        elif preprocessed['total_processed'] == 0:
            lines.append("   🚀 Siap untuk preprocessing - klik 'Mulai Preprocessing'")
        else:
            lines.append("   ✅ Dataset dan preprocessing lengkap - siap untuk augmentasi")
        
        return "\n".join(lines)
    
    def _on_check_dataset_click(b):
        """Handler untuk tombol check dataset dengan lightweight approach."""
        # Check operation state
        can_start, message = ui_state.can_start_operation('validation')
        if not can_start:
            logger and logger.warning(f"⚠️ {message}")
            update_status_panel(ui_components['status_panel'], message, "warning")
            return
        
        # Set button processing state
        ui_state.set_button_processing('check_button', True, "Checking...")
        
        logger and logger.info("🔍 Memulai quick dataset check...")
        update_status_panel(ui_components['status_panel'], "Checking dataset struktur...", "info")
        
        try:
            # Quick lightweight check
            check_result = _lightweight_dataset_check()
            summary = _format_quick_summary(check_result)
            
            logger and logger.info(summary)
            
            # Smart status update berdasarkan hasil
            raw = check_result['raw']
            compatibility = check_result.get('compatibility', {})
            preprocessed = check_result['preprocessed']
            
            if not raw['valid']:
                status_msg = "Dataset tidak ditemukan - silakan download terlebih dahulu"
                panel_type = "error"
            elif not compatibility.get('ready_for_preprocessing', False):
                blocking_count = len(compatibility.get('blocking_issues', []))
                status_msg = f"Dataset ditemukan dengan {blocking_count} issues yang perlu diperbaiki"
                panel_type = "warning"
            elif preprocessed['total_processed'] == 0:
                total_images = raw['total_images']
                status_msg = f"Dataset siap untuk preprocessing: {total_images:,} gambar terdeteksi"
                panel_type = "success"
            else:
                processed_count = preprocessed['total_processed']
                raw_count = raw['total_images']
                status_msg = f"Dataset lengkap: {raw_count:,} raw, {processed_count:,} preprocessed"
                panel_type = "info"
            
            update_status_panel(ui_components['status_panel'], status_msg, panel_type)
                
        except Exception as e:
            error_msg = f"Error checking dataset: {str(e)}"
            logger and logger.error(f"❌ {error_msg}")
            update_status_panel(ui_components['status_panel'], error_msg, "error")
        
        finally:
            # Reset button state
            ui_state.set_button_processing('check_button', False, 
                                         success_text="Check Dataset")
    
    # Setup event handler
    ui_components['check_button'].on_click(_on_check_dataset_click)
    
    logger and logger.debug("✅ Lightweight dataset checker setup selesai")
    
    return ui_components