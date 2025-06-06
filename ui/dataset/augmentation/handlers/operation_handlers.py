"""
File: smartcash/ui/dataset/augmentation/handlers/operation_handlers.py
Deskripsi: Fixed operation handlers dengan granular progress tracking dan communicator integration
"""

from typing import Dict, Any
from smartcash.ui.dataset.augmentation.utils.ui_logger_utils import log_to_ui, show_progress_safe, complete_progress_safe, error_progress_safe

def execute_augmentation(ui_components: Dict[str, Any]):
    """Execute augmentation dengan granular progress tracking"""
    try:
        # Disable buttons dan show progress
        _disable_operation_buttons(ui_components)
        # Gunakan show_progress_safe dengan steps dan step_weights
        augmentation_steps = ["prepare", "augment", "normalize", "verify"]
        step_weights = {"prepare": 10, "augment": 50, "normalize": 30, "verify": 10}
        show_progress_safe(ui_components, 'Augmentasi Dataset', augmentation_steps, step_weights)
        
        log_to_ui(ui_components, "🚀 Memulai pipeline augmentasi dengan progress tracking...", 'info')
        
        service = _create_service_safely(ui_components)
        if not service:
            log_to_ui(ui_components, "Gagal membuat augmentation service", 'error', "❌ ")
            return
        
        target_split = _get_target_split_safe(ui_components)
        log_to_ui(ui_components, f"📂 Target split: {target_split}", 'info')
        
        # Create communicator untuk service
        communicator = _create_service_communicator(ui_components)
        
        # Execute dengan granular progress callback
        result = service.run_full_augmentation_pipeline(
            target_split=target_split,
            progress_callback=_create_granular_progress_callback(ui_components)
        )
        
        _handle_service_result(ui_components, result, 'augmentation pipeline')
        
    except Exception as e:
        error_msg = f"Pipeline error: {str(e)}"
        log_to_ui(ui_components, error_msg, 'error', "❌ ")
        error_progress_safe(ui_components, error_msg)
    finally:
        # Re-enable buttons
        _enable_operation_buttons(ui_components)

def execute_check(ui_components: Dict[str, Any]):
    """Execute check dengan detailed progress steps"""
    try:
        _disable_operation_buttons(ui_components)
        # Gunakan show_progress_safe dengan steps dan step_weights
        check_steps = ["locate", "analyze_raw", "analyze_augmented", "analyze_preprocessed"]
        step_weights = {"locate": 10, "analyze_raw": 30, "analyze_augmented": 30, "analyze_preprocessed": 30}
        show_progress_safe(ui_components, 'Pengecekan Dataset', check_steps, step_weights)
        
        log_to_ui(ui_components, "🔍 Memulai pengecekan dataset dengan detail...", 'info')
        
        from smartcash.dataset.augmentor.utils.dataset_detector import detect_split_structure
        from smartcash.dataset.augmentor.utils.path_operations import get_best_data_location
        
        # Step 1: Find data location (0-20%)
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'update_overall'):
            progress_tracker.update_overall(10, "🔍 Mencari lokasi data")
        elif progress_tracker and hasattr(progress_tracker, 'update'):
            # Fallback untuk kompatibilitas
            progress_tracker.update('overall', 10, "🔍 Mencari lokasi data")
        
        data_location = get_best_data_location()
        log_to_ui(ui_components, f"📁 Data location: {data_location}", 'info')
        
        # Step 2: Check raw dataset (20-50%)
        if progress_tracker and hasattr(progress_tracker, 'update_overall'):
            progress_tracker.update_overall(30, "📊 Menganalisis raw dataset")
        elif progress_tracker and hasattr(progress_tracker, 'update'):
            # Fallback untuk kompatibilitas
            progress_tracker.update('overall', 30, "📊 Menganalisis raw dataset")
        
        raw_info = detect_split_structure(data_location)
        if raw_info['status'] == 'success':
            raw_images = raw_info.get('total_images', 0)
            raw_labels = raw_info.get('total_labels', 0)
            available_splits = raw_info.get('available_splits', [])
            
            log_to_ui(ui_components, f"✅ Raw Dataset: {raw_images} gambar, {raw_labels} label", 'success')
            log_to_ui(ui_components, f"📂 Available splits: {', '.join(available_splits) if available_splits else 'flat structure'}", 'info')
        else:
            log_to_ui(ui_components, f"❌ Raw dataset tidak ditemukan: {raw_info.get('message', 'Unknown error')}", 'error')
        
        # Step 3: Check augmented dataset (50-75%)
        if progress_tracker and hasattr(progress_tracker, 'update_overall'):
            progress_tracker.update_overall(60, "🔄 Mengecek augmented dataset")
        elif progress_tracker and hasattr(progress_tracker, 'update'):
            # Fallback untuk kompatibilitas
            progress_tracker.update('overall', 60, "🔄 Mengecek augmented dataset")
        
        _check_augmented_dataset(ui_components, data_location)
        
        # Step 4: Check preprocessed dataset (75-100%)
        if progress_tracker and hasattr(progress_tracker, 'update_overall'):
            progress_tracker.update_overall(80, "🔧 Mengecek preprocessed dataset")
        elif progress_tracker and hasattr(progress_tracker, 'update'):
            # Fallback untuk kompatibilitas
            progress_tracker.update('overall', 80, "🔧 Mengecek preprocessed dataset")
        
        _check_preprocessed_dataset(ui_components, data_location)
        
        # Final summary
        ready_status = "✅ Siap" if raw_info['status'] == 'success' and raw_images > 0 else "❌ Tidak siap"
        log_to_ui(ui_components, f"🎯 Status augmentasi: {ready_status}", 'success' if ready_status.startswith('✅') else 'warning')
        
        complete_progress_safe(ui_components, "Check dataset selesai dengan detail")
        
    except Exception as e:
        error_msg = f"Check error: {str(e)}"
        log_to_ui(ui_components, error_msg, 'error', "❌ ")
        error_progress_safe(ui_components, error_msg)
    finally:
        _enable_operation_buttons(ui_components)

def _create_service_safely(ui_components: Dict[str, Any]):
    """Create augmentation service dengan communicator integration"""
    try:
        from smartcash.dataset.augmentor.service import create_service_from_ui
        
        # Add communicator ke ui_components jika belum ada
        if 'comm' not in ui_components:
            ui_components['comm'] = _create_service_communicator(ui_components)
        
        return create_service_from_ui(ui_components)
        
    except ImportError as e:
        log_to_ui(ui_components, f"Service import error: {str(e)}", 'error', "❌ ")
        return None
    except Exception as e:
        log_to_ui(ui_components, f"Service creation error: {str(e)}", 'error', "❌ ")
        return None

def _create_service_communicator(ui_components: Dict[str, Any]):
    """Create communicator untuk service layer"""
    try:
        from smartcash.dataset.augmentor.communicator import create_communicator
        return create_communicator(ui_components)
    except ImportError:
        return None

def _create_granular_progress_callback(ui_components: Dict[str, Any]):
    """Create granular progress callback dengan detailed steps"""
    import time
    last_update = {'time': 0, 'percentage': -1, 'step': ''}
    
    def callback(step: str, current: int, total: int, message: str):
        current_time = time.time()
        percentage = min(100, max(0, int((current / max(1, total)) * 100)))
        
        # Update lebih sering untuk step changes atau milestone
        should_update = (
            step != last_update['step'] or  # Step change
            percentage in [0, 25, 50, 75, 100] or  # Milestone
            (current_time - last_update['time'] > 0.5)  # Time threshold
        )
        
        if should_update:
            last_update.update({'time': current_time, 'percentage': percentage, 'step': step})
            
            # Log granular progress
            step_emoji = {'overall': '🎯', 'step': '📊', 'current': '⚡'}.get(step, '📈')
            progress_msg = f"{step_emoji} {step.title()}: {percentage}% - {message}"
            log_to_ui(ui_components, progress_msg, 'info')
            
            # Update progress tracker dengan step context
            progress_tracker = ui_components.get('progress_tracker')
            if progress_tracker:
                # Mapping step ke API yang benar
                if step == 'overall':
                    # Gunakan update_overall untuk progress keseluruhan
                    if hasattr(progress_tracker, 'update_overall'):
                        progress_tracker.update_overall(percentage, message)
                    elif hasattr(progress_tracker, 'update'):
                        # Fallback untuk kompatibilitas
                        progress_tracker.update('level1', percentage, message)
                else:
                    # Gunakan update_current untuk progress langkah saat ini
                    if hasattr(progress_tracker, 'update_current'):
                        progress_tracker.update_current(percentage, message)
                    elif hasattr(progress_tracker, 'update'):
                        # Fallback untuk kompatibilitas
                        progress_tracker.update('level2', percentage, message)
    
    return callback

def _check_augmented_dataset(ui_components: Dict[str, Any], data_location: str):
    """Check augmented dataset dengan detail"""
    try:
        from smartcash.dataset.augmentor.utils.dataset_detector import detect_split_structure
        
        aug_info = detect_split_structure(f"{data_location}/augmented")
        if aug_info['status'] == 'success' and aug_info.get('total_images', 0) > 0:
            aug_images = aug_info.get('total_images', 0)
            aug_splits = aug_info.get('available_splits', [])
            log_to_ui(ui_components, f"✅ Augmented Dataset: {aug_images} file", 'success')
            if aug_splits:
                log_to_ui(ui_components, f"📂 Augmented splits: {', '.join(aug_splits)}", 'info')
        else:
            log_to_ui(ui_components, "🔄 Augmented Dataset: Belum ada file augmented", 'info')
    except Exception:
        log_to_ui(ui_components, "🔄 Augmented Dataset: Belum ada file augmented", 'info')

def _check_preprocessed_dataset(ui_components: Dict[str, Any], data_location: str):
    """Check preprocessed dataset dengan detail"""
    try:
        from smartcash.dataset.augmentor.utils.dataset_detector import detect_split_structure
        
        prep_info = detect_split_structure(f"{data_location}/preprocessed")
        if prep_info['status'] == 'success' and prep_info.get('total_images', 0) > 0:
            prep_images = prep_info.get('total_images', 0)
            prep_splits = prep_info.get('available_splits', [])
            log_to_ui(ui_components, f"✅ Preprocessed Dataset: {prep_images} file", 'success')
            if prep_splits:
                log_to_ui(ui_components, f"📂 Preprocessed splits: {', '.join(prep_splits)}", 'info')
        else:
            log_to_ui(ui_components, "🔧 Preprocessed Dataset: Belum ada file preprocessed", 'info')
    except Exception:
        log_to_ui(ui_components, "🔧 Preprocessed Dataset: Belum ada file preprocessed", 'info')

def _handle_service_result(ui_components: Dict[str, Any], result: Dict[str, Any], operation: str):
    """Handle service result dengan detailed feedback"""
    if result.get('status') == 'success':
        if operation == 'augmentation pipeline':
            total_generated = result.get('total_generated', 0)
            total_normalized = result.get('total_normalized', 0)
            processing_time = result.get('processing_time', 0)
            
            success_msg = f"✅ Pipeline berhasil:\n📊 Generated: {total_generated} file\n🔧 Normalized: {total_normalized} file\n⏱️ Time: {processing_time:.1f}s"
            
            # Log additional details
            if 'aug_result' in result:
                aug_result = result['aug_result']
                log_to_ui(ui_components, f"📈 Augmentation success rate: {aug_result.get('success_rate', 0):.1f}%", 'info')
            
            if 'norm_result' in result:
                norm_result = result['norm_result']
                target_dir = norm_result.get('target_dir', 'data/preprocessed')
                log_to_ui(ui_components, f"📁 Preprocessed files saved to: {target_dir}", 'info')
            
        else:
            success_msg = f"{operation.title()} berhasil"
        
        log_to_ui(ui_components, success_msg, 'success', "✅ ")
        complete_progress_safe(ui_components, success_msg.split('\n')[0])
        
    else:
        error_msg = f"{operation.title()} gagal: {result.get('message', 'Unknown error')}"
        log_to_ui(ui_components, error_msg, 'error', "❌ ")
        error_progress_safe(ui_components, error_msg)

def _disable_operation_buttons(ui_components: Dict[str, Any]):
    """Disable operation buttons dengan visual feedback"""
    button_keys = ['augment_button', 'check_button', 'cleanup_button']
    for key in button_keys:
        button = ui_components.get(key)
        if button and hasattr(button, 'disabled'):
            button.disabled = True
            # Update description dengan processing indicator
            if hasattr(button, 'description'):
                original_desc = getattr(button, '_original_description', button.description)
                if not hasattr(button, '_original_description'):
                    button._original_description = button.description
                button.description = f"⏳ {original_desc}"

def _enable_operation_buttons(ui_components: Dict[str, Any]):
    """Enable operation buttons dengan restore description"""
    button_keys = ['augment_button', 'check_button', 'cleanup_button']
    for key in button_keys:
        button = ui_components.get(key)
        if button and hasattr(button, 'disabled'):
            button.disabled = False
            # Restore original description
            if hasattr(button, '_original_description'):
                button.description = button._original_description

def _get_target_split_safe(ui_components: Dict[str, Any]) -> str:
    """Get target split dengan safe default"""
    target_split_widget = ui_components.get('target_split')
    if target_split_widget and hasattr(target_split_widget, 'value'):
        return getattr(target_split_widget, 'value', 'train')
    return 'train'