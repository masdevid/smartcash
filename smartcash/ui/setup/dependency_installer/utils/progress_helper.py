"""
File: smartcash/ui/setup/dependency_installer/utils/progress_helper.py
Deskripsi: Helper functions untuk progress tracking dengan pendekatan one-liner dan DRY
"""

from typing import Dict, Any, Optional

def update_progress_step(ui_components: Dict[str, Any], step_value: int, message: str, 
                        color: str = "#007bff") -> None:
    """
    Update progress step dengan pendekatan one-liner
    
    Args:
        ui_components: UI components yang berisi update_progress
        step_value: Nilai progress step (0-100)
        message: Pesan progress
        color: Warna progress bar
    """
    if 'update_progress' in ui_components and callable(ui_components['update_progress']):
        ui_components['update_progress']('step', step_value, message, color)

def update_overall_progress(ui_components: Dict[str, Any], progress_value: int, message: str, 
                           color: str = "#007bff") -> None:
    """
    Update overall progress dengan pendekatan one-liner
    
    Args:
        ui_components: UI components yang berisi update_progress
        progress_value: Nilai progress (0-100)
        message: Pesan progress
        color: Warna progress bar
    """
    if 'update_progress' in ui_components and callable(ui_components['update_progress']):
        ui_components['update_progress']('overall', progress_value, message, color)

def update_current_progress(ui_components: Dict[str, Any], progress_value: int, message: str, 
                           color: str = "#007bff") -> None:
    """
    Update current progress dengan pendekatan one-liner
    
    Args:
        ui_components: UI components yang berisi update_progress
        progress_value: Nilai progress (0-100)
        message: Pesan progress
        color: Warna progress bar
    """
    if 'update_progress' in ui_components and callable(ui_components['update_progress']):
        ui_components['update_progress']('current', progress_value, message, color)

def calculate_batch_progress(current_index: int, total_items: int) -> int:
    """
    Hitung persentase progress untuk batch processing
    
    Args:
        current_index: Index item saat ini (0-based)
        total_items: Total jumlah item
    
    Returns:
        Persentase progress (0-100)
    """
    if total_items <= 0:
        return 0
    return int(((current_index + 1) / total_items) * 100)

def start_operation(ui_components: Dict[str, Any], operation_name: str, total_items: int) -> None:
    """
    Mulai operasi dengan progress tracking dan logging
    
    Args:
        ui_components: UI components
        operation_name: Nama operasi
        total_items: Total jumlah item yang akan diproses
    """
    from smartcash.ui.setup.dependency_installer.utils.logger_helper import log_message
    
    # Reset progress bar
    if 'reset_progress_bar' in ui_components and callable(ui_components['reset_progress_bar']):
        ui_components['reset_progress_bar'](0, f"Memulai {operation_name}...", True)
    
    # Show for operation
    if 'show_for_operation' in ui_components and callable(ui_components['show_for_operation']):
        ui_components['show_for_operation'](operation_name)
    
    # Update progress step
    update_progress_step(ui_components, 25, f"Mempersiapkan {operation_name}...", "#007bff")
    
    # Log message
    log_message(ui_components, f"🚀 Memulai {operation_name} dengan {total_items} item", "info")
    
    # Update progress step
    update_progress_step(ui_components, 50, f"Memulai {operation_name}...", "#007bff")

def complete_operation(ui_components: Dict[str, Any], operation_name: str, stats: Dict[str, Any]) -> None:
    """
    Selesaikan operasi dengan progress tracking dan logging
    
    Args:
        ui_components: UI components
        operation_name: Nama operasi
        stats: Statistik hasil operasi
    """
    from smartcash.ui.setup.dependency_installer.utils.logger_helper import log_message
    
    # Update progress
    update_overall_progress(ui_components, 100, f"{operation_name} selesai", "#28a745")
    update_progress_step(ui_components, 100, f"{operation_name} selesai", "#28a745")
    
    # Format success message
    success_message = f"📊 {operation_name} selesai: "
    if 'success' in stats and 'total' in stats:
        success_message += f"{stats['success']}/{stats['total']} berhasil"
    if 'failed' in stats:
        success_message += f", {stats['failed']} gagal"
    if 'duration' in stats:
        success_message += f" ({stats['duration']:.1f}s)"
    
    # Log message
    log_message(ui_components, success_message, "success")
    
    # Complete operation
    if 'complete_operation' in ui_components and callable(ui_components['complete_operation']):
        ui_components['complete_operation'](operation_name, stats)

def handle_item_error(ui_components: Dict[str, Any], item_name: str, error_msg: str) -> None:
    """
    Handle error untuk item dengan progress tracking dan logging
    
    Args:
        ui_components: UI components
        item_name: Nama item yang error
        error_msg: Pesan error
    """
    from smartcash.ui.setup.dependency_installer.utils.logger_helper import log_message
    
    # Update progress
    update_current_progress(ui_components, 100, f"❌ {item_name} gagal", "#dc3545")
    
    # Log message
    log_message(ui_components, f"Gagal memproses {item_name}: {error_msg}", "error")
    
    # Error operation
    if 'error_operation' in ui_components and callable(ui_components['error_operation']):
        ui_components['error_operation'](item_name, error_msg)

def handle_item_success(ui_components: Dict[str, Any], item_name: str, message: Optional[str] = None) -> None:
    """
    Handle success untuk item dengan progress tracking dan logging
    
    Args:
        ui_components: UI components
        item_name: Nama item yang berhasil
        message: Pesan tambahan (opsional)
    """
    from smartcash.ui.setup.dependency_installer.utils.logger_helper import log_message
    
    # Update progress
    success_message = f"✅ {item_name} berhasil"
    if message:
        success_message += f": {message}"
    
    update_current_progress(ui_components, 100, success_message, "#28a745")
    
    # Log message
    log_message(ui_components, f"Berhasil memproses {item_name}", "success")
