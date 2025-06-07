"""
File: smartcash/ui/dataset/downloader/utils/ui_utils.py
Deskripsi: Utilitas UI untuk download operations
"""

from typing import Dict, Any

def log_download_config(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Log download configuration dalam format yang rapi"""
    roboflow = config.get('data', {}).get('roboflow', {})
    download = config.get('download', {})
    
    # Mask API key untuk security
    api_key = roboflow.get('api_key', '')
    masked_key = f"{api_key[:4]}{'*' * (len(api_key) - 8)}{api_key[-4:]}" if len(api_key) > 8 else '****'
    
    config_lines = [
        "🔧 Konfigurasi Download:",
        f"🎯 Target: {roboflow.get('workspace')}/{roboflow.get('project')}:v{roboflow.get('version')}",
        f"🔑 API Key: {masked_key}",
        f"📦 Format: {roboflow.get('output_format', 'yolov5pytorch')}",
        f"🔄 UUID Rename: {'✅' if download.get('rename_files', True) else '❌'}",
        f"✅ Validasi: {'✅' if download.get('validate_download', True) else '❌'}",
        f"💾 Backup: {'✅' if download.get('backup_existing', False) else '❌'}"
    ]
    
    log_to_accordion(ui_components, '\n'.join(config_lines), 'info')

def display_check_results(ui_components: Dict[str, Any], result: Dict[str, Any]):
    """Display scan results dalam format yang rapi"""
    summary = result.get('summary', {})
    
    # Main summary
    summary_lines = [
        "📊 Ringkasan Dataset:",
        f"📂 Path: {result.get('dataset_path')}",
        f"🖼️ Total Gambar: {summary.get('total_images', 0):,}",
        f"🏷️ Total Label: {summary.get('total_labels', 0):,}",
        f"📦 Download Files: {summary.get('download_files', 0):,}"
    ]
    
    # Splits detail
    splits = result.get('splits', {})
    if splits:
        summary_lines.append("\n📊 Detail per Split:")
        for split_name, split_data in splits.items():
            if split_data.get('status') == 'success':
                img_count = split_data.get('images', 0)
                label_count = split_data.get('labels', 0)
                size_formatted = split_data.get('size_formatted', '0 B')
                summary_lines.append(f"  • {split_name}: {img_count:,} gambar, {label_count:,} label ({size_formatted})")
    
    # Downloads detail
    downloads = result.get('downloads', {})
    if downloads.get('status') == 'success':
        download_count = downloads.get('file_count', 0)
        download_size = downloads.get('size_formatted', '0 B')
        summary_lines.append(f"\n📦 Downloads: {download_count:,} file ({download_size})")
    
    log_to_accordion(ui_components, '\n'.join(summary_lines), 'success')

def show_download_success(ui_components: Dict[str, Any], result: Dict[str, Any]):
    """Show download success dengan detailed stats"""
    stats = result.get('stats', {})
    
    # Format success message
    success_lines = [
        f"✅ Download selesai: {stats.get('total_images', 0):,} gambar, {stats.get('total_labels', 0):,} label",
        f"📂 Output: {result.get('output_dir', 'N/A')}",
        f"⏱️ Durasi: {result.get('duration', 0):.1f} detik"
    ]
    
    # Add split details
    splits = stats.get('splits', {})
    if splits:
        success_lines.append("📊 Detail splits:")
        for split_name, split_stats in splits.items():
            img_count = split_stats.get('images', 0)
            label_count = split_stats.get('labels', 0)
            success_lines.append(f"  • {split_name}: {img_count:,} gambar, {label_count:,} label")
    
    # Add UUID renaming info
    if stats.get('uuid_renamed'):
        naming_stats = stats.get('naming_stats', {})
        if naming_stats:
            success_lines.append(f"🔄 UUID Renaming: {naming_stats.get('total_files', 0)} files")
    
    success_message = '\n'.join(success_lines)
    show_ui_success(ui_components, success_message)

def log_to_accordion(ui_components: Dict[str, Any], message: str, level: str = 'info'):
    """Log message menggunakan ui_logger yang sudah ada"""
    logger = ui_components.get('logger')
    if logger:
        # Map level ke ui_logger method
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
    if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'clear_output'):
        with ui_components['log_output']:
            ui_components['log_output'].clear_output(wait=True)

def handle_ui_error(ui_components: Dict[str, Any], error_msg: str, button_manager=None):
    """Handle error dengan UI updates dan proper state reset"""
    from smartcash.ui.utils.fallback_utils import show_status_safe
    
    logger = ui_components.get('logger')
    if logger:
        logger.error(f"❌ {error_msg}")
    
    # Update progress tracker
    progress_tracker = ui_components.get('progress_tracker')
    if progress_tracker:
        if hasattr(progress_tracker, 'error'):
            progress_tracker.error(error_msg)
        elif hasattr(progress_tracker, 'reset'):
            progress_tracker.reset()
    
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
    if progress_tracker:
        if hasattr(progress_tracker, 'complete'):
            progress_tracker.complete(message)
    
    # Log to accordion
    log_to_accordion(ui_components, f"✅ {message}", 'success')
    
    # Show status
    show_status_safe(message, 'success', ui_components)
    
    # Enable buttons
    if button_manager:
        button_manager.enable_buttons()

def map_step_to_current_progress(step: str, overall_progress: int) -> int:
    """Map step progress to current operation progress bar"""
    step_mapping = {
        'init': (0, 10), 'metadata': (10, 20), 'backup': (20, 25),
        'download': (25, 70), 'extract': (70, 80), 'organize': (80, 90),
        'uuid_rename': (90, 95), 'validate': (95, 98), 'cleanup': (98, 100)
    }
    
    step_key = step.lower().split('_')[0]
    if step_key in step_mapping:
        start, end = step_mapping[step_key]
        range_size = end - start
        step_progress = start + (overall_progress * range_size / 100)
        return min(100, max(0, int(step_progress)))
    return overall_progress

def is_milestone_step(step: str, progress: int) -> bool:
    """Only log major milestones to prevent browser crash"""
    milestone_steps = ['init', 'metadata', 'backup', 'extract', 'organize', 'validate', 'complete']
    return (step.lower() in milestone_steps or progress in [0, 25, 50, 75, 100] or progress % 25 == 0)