"""
File: smartcash/ui/dataset/preprocessing/utils/ui_utils.py
Deskripsi: UI utilities untuk preprocessing operations
"""

from typing import Dict, Any

def log_preprocessing_config(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Log preprocessing configuration dalam format yang rapi"""
    preprocessing = config.get('preprocessing', {})
    performance = config.get('performance', {})
    
    normalization = preprocessing.get('normalization', {})
    target_size = normalization.get('target_size', [640, 640])
    
    config_lines = [
        "🔧 Konfigurasi Preprocessing:",
        f"📐 Resolusi: {target_size[0]}x{target_size[1]}",
        f"🎨 Normalisasi: {normalization.get('method', 'minmax')}",
        f"⚙️ Workers: {performance.get('num_workers', 8)}",
        f"🎯 Target Split: {preprocessing.get('target_split', 'all')}",
        f"✅ Validasi: {'✅' if preprocessing.get('validate', {}).get('enabled', True) else '❌'}",
        f"📊 Analysis: {'✅' if preprocessing.get('analysis', {}).get('enabled', False) else '❌'}"
    ]
    
    log_to_accordion(ui_components, '\n'.join(config_lines), 'info')

def display_preprocessing_results(ui_components: Dict[str, Any], result: Dict[str, Any]):
    """Display preprocessing results dalam format yang rapi"""
    stats = result.get('stats', {})
    
    # Main summary
    summary_lines = [
        "📊 Hasil Preprocessing:",
        f"🖼️ Total Processed: {stats.get('total_images', 0):,} gambar",
        f"⏱️ Durasi: {result.get('processing_time', 0):.1f} detik",
        f"📂 Output: {result.get('output_dir', 'N/A')}"
    ]
    
    # Splits detail
    splits = stats.get('splits', {})
    if splits:
        summary_lines.append("\n📊 Detail per Split:")
        for split_name, split_stats in splits.items():
            img_count = split_stats.get('processed', 0)
            summary_lines.append(f"  • {split_name}: {img_count:,} gambar")
    
    # UUID info jika ada
    if stats.get('uuid_processing'):
        summary_lines.append(f"\n🔤 UUID Processing: {stats.get('uuid_files', 0)} files")
    
    log_to_accordion(ui_components, '\n'.join(summary_lines), 'success')

def show_preprocessing_success(ui_components: Dict[str, Any], result: Dict[str, Any]):
    """Show preprocessing success dengan detailed stats"""
    stats = result.get('stats', {})
    
    success_lines = [
        f"✅ Preprocessing selesai: {stats.get('total_images', 0):,} gambar",
        f"📂 Output: {result.get('output_dir', 'N/A')}",
        f"⏱️ Durasi: {result.get('processing_time', 0):.1f} detik"
    ]
    
    # Add normalization info
    if stats.get('normalization_applied'):
        success_lines.append(f"🎨 Normalisasi: {stats.get('normalization_method', 'applied')}")
    
    success_message = '\n'.join(success_lines)
    show_ui_success(ui_components, success_message)

def log_to_accordion(ui_components: Dict[str, Any], message: str, level: str = 'info'):
    """Log message menggunakan ui_logger yang sudah ada"""
    logger = ui_components.get('logger')
    if logger:
        log_methods = {
            'info': logger.info,
            'success': logger.success,
            'warning': logger.warning,
            'error': logger.error
        }
        log_method = log_methods.get(level, logger.info)
        log_method(message)
    
    # Auto-expand untuk errors/warnings
    if level in ['error', 'warning'] and 'log_accordion' in ui_components:
        if hasattr(ui_components['log_accordion'], 'selected_index'):
            ui_components['log_accordion'].selected_index = 0

def clear_outputs(ui_components: Dict[str, Any]):
    """Clear UI output areas"""
    # Clear log output
    if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'clear_output'):
        with ui_components['log_output']:
            ui_components['log_output'].clear_output(wait=True)
    
    # Clear confirmation area if exists
    if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'clear_output'):
        with ui_components['confirmation_area']:
            ui_components['confirmation_area'].clear_output(wait=True)
    
    # Reset progress tracker if exists
    progress_tracker = ui_components.get('progress_tracker')
    if progress_tracker and hasattr(progress_tracker, 'reset'):
        progress_tracker.reset()

def handle_ui_error(ui_components: Dict[str, Any], error_msg: str, button_manager=None):
    """Handle error dengan UI updates dan proper state reset"""
    from smartcash.ui.utils.fallback_utils import show_status_safe
    
    logger = ui_components.get('logger')
    if logger:
        logger.error(f"❌ {error_msg}")
    
    # Update progress tracker
    progress_tracker = ui_components.get('progress_tracker')
    if progress_tracker and hasattr(progress_tracker, 'error'):
        progress_tracker.error(error_msg)
    
    # Log to accordion
    log_to_accordion(ui_components, f"❌ {error_msg}", 'error')
    
    # Show status
    show_status_safe(error_msg, 'error', ui_components)
    
    # Enable buttons
    if button_manager:
        button_manager.enable_buttons()

def show_ui_success(ui_components: Dict[str, Any], message: str, button_manager=None):
    """Show success dengan UI updates"""
    from smartcash.ui.utils.fallback_utils import show_status_safe
    
    logger = ui_components.get('logger')
    if logger:
        logger.success(f"✅ {message}")
    
    # Update progress tracker
    progress_tracker = ui_components.get('progress_tracker')
    if progress_tracker and hasattr(progress_tracker, 'complete'):
        progress_tracker.complete(message)
    
    # Log to accordion
    log_to_accordion(ui_components, f"✅ {message}", 'success')
    
    # Show status
    show_status_safe(message, 'success', ui_components)
    
    # Enable buttons
    if button_manager:
        button_manager.enable_buttons()

def is_milestone_step(step: str, progress: int) -> bool:
    """Only log major milestones to prevent browser crash"""
    milestone_steps = ['init', 'validate', 'normalize', 'resize', 'save', 'complete']
    return (step.lower() in milestone_steps or progress in [0, 25, 50, 75, 100] or progress % 25 == 0)