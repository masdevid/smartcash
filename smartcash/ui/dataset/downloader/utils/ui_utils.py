"""
File: smartcash/ui/dataset/downloader/utils/ui_utils.py
Deskripsi: Optimized UI utilities untuk download operations dengan confirmation management
"""

from typing import Dict, Any, Optional

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
    """Log message menggunakan logger_bridge yang sudah ada
    
    Args:
        ui_components: Dictionary berisi komponen UI
        message: Pesan yang akan di-log
        level: Level log (info, success, warning, error)
    """
    logger_bridge = ui_components.get('logger_bridge')
    if logger_bridge:
        # Map level ke logger_bridge method
        log_methods = {
            'info': logger_bridge.info,
            'success': logger_bridge.info,  # Fallback to info if success not available
            'warning': logger_bridge.warning,
            'error': logger_bridge.error
        }
        log_method = log_methods.get(level, logger_bridge.info)
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

# Error handling has been moved to error_handling.py
# Please use the centralized error handling utilities from there

# === CONFIRMATION AREA MANAGEMENT ===

def show_confirmation_area(ui_components: Dict[str, Any]):
    """Show confirmation area untuk dialog visibility"""
    confirmation_area = ui_components.get('confirmation_area')
    if confirmation_area and hasattr(confirmation_area, 'layout'):
        confirmation_area.layout.visibility = 'visible'
        confirmation_area.layout.height = 'auto'
        confirmation_area.layout.min_height = '50px'

def hide_confirmation_area(ui_components: Dict[str, Any]):
    """Hide confirmation area setelah dialog selesai"""
    confirmation_area = ui_components.get('confirmation_area')
    if confirmation_area and hasattr(confirmation_area, 'layout'):
        confirmation_area.layout.visibility = 'hidden'
        confirmation_area.layout.height = '0px'
        confirmation_area.layout.min_height = '0px'

def clear_dialog_area(ui_components: Dict[str, Any]):
    """Clear dialog/confirmation area seperti preprocessing (alias untuk hide)"""
    hide_confirmation_area(ui_components)

# === PROGRESS MAPPING UTILITIES ===

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

# === ONE-LINER UTILITIES ===

# Format utilities
format_file_count = lambda count: f"{count:,} file{'s' if count != 1 else ''}"
format_duration = lambda seconds: f"{seconds:.1f} detik" if seconds < 60 else f"{seconds/60:.1f} menit"
safe_get_widget = lambda ui_components, key, attr='value': getattr(ui_components.get(key, type('', (), {attr: None})()), attr, None)

# Status utilities
is_confirmation_visible = lambda ui_components: getattr(ui_components.get('confirmation_area', type('', (), {'layout': type('', (), {'visibility': 'hidden'})()})()), 'layout', type('', (), {'visibility': 'hidden'})()).visibility == 'visible'
has_progress_tracker = lambda ui_components: ui_components.get('progress_tracker') is not None

# Log level mapping
get_log_emoji = lambda level: {'info': 'ℹ️', 'success': '✅', 'warning': '⚠️', 'error': '❌', 'debug': '🔍'}.get(level, 'ℹ️')