"""
File: smartcash/ui/dataset/downloader/utils/ui_utils.py
Deskripsi: Optimized UI utilities untuk download operations
"""

from typing import Dict, Any, Optional

# === DISPLAY UTILITIES ===

def display_check_results(ui_components: Dict[str, Any], result: Dict[str, Any]) -> None:
    """Display scan results dalam format yang rapi untuk UI components.
    
    Args:
        ui_components: Dictionary UI components
        result: Dictionary hasil scan
    """
    summary_container = ui_components.get('summary_container')
    if not summary_container:
        return
        
    summary = result.get('summary', {})
    
    # Format HTML content
    html_lines = [
        "<div style='padding: 10px; background-color: #f0f8ff; border-radius: 5px; border-left: 5px solid #4682b4;'>",
        "<h3>📊 Ringkasan Dataset</h3>",
        f"<p>📂 Path: {result.get('dataset_path')}</p>",
        f"<p>🖼️ Total Gambar: <b>{summary.get('total_images', 0):,}</b></p>",
        f"<p>🏷️ Total Label: <b>{summary.get('total_labels', 0):,}</b></p>"
    ]
    
    # Splits detail
    splits = result.get('splits', {})
    if splits:
        html_lines.append("<h4>Detail per Split:</h4>")
        html_lines.append("<ul>")
        for split_name, split_data in splits.items():
            if split_data.get('status') == 'success':
                img_count = split_data.get('images', 0)
                label_count = split_data.get('labels', 0)
                size_formatted = split_data.get('size_formatted', '0 B')
                html_lines.append(f"<li><b>{split_name}</b>: {img_count:,} gambar, {label_count:,} label ({size_formatted})</li>")
        html_lines.append("</ul>")
    
    html_lines.append("</div>")
    
    # Update summary container
    summary_container.clear_output()
    summary_container.append_html("".join(html_lines))

# === CONFIRMATION AREA MANAGEMENT ===

def show_confirmation_area(ui_components: Dict[str, Any]) -> None:
    """Show confirmation area untuk dialog visibility.
    
    Args:
        ui_components: Dictionary UI components
    """
    confirmation_area = ui_components.get('confirmation_area')
    if confirmation_area and hasattr(confirmation_area, 'layout'):
        confirmation_area.layout.visibility = 'visible'
        confirmation_area.layout.height = 'auto'
        confirmation_area.layout.min_height = '50px'

def hide_confirmation_area(ui_components: Dict[str, Any]) -> None:
    """Hide confirmation area setelah dialog selesai.
    
    Args:
        ui_components: Dictionary UI components
    """
    confirmation_area = ui_components.get('confirmation_area')
    if confirmation_area and hasattr(confirmation_area, 'layout'):
        confirmation_area.layout.visibility = 'hidden'
        confirmation_area.layout.height = '0px'
        confirmation_area.layout.min_height = '0px'

# === PROGRESS MAPPING UTILITIES ===

def map_step_to_current_progress(step: str, overall_progress: int) -> int:
    """Map step progress to current operation progress bar.
    
    Args:
        step: Current step name
        overall_progress: Overall progress percentage (0-100)
        
    Returns:
        Mapped progress percentage (0-100)
    """
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
    """Only log major milestones to prevent browser crash.
    
    Args:
        step: Current step name
        progress: Current progress percentage
        
    Returns:
        True if this is a milestone step that should be logged
    """
    milestone_steps = ['init', 'metadata', 'backup', 'extract', 'organize', 'validate', 'complete']
    return (step.lower() in milestone_steps or progress in [0, 25, 50, 75, 100] or progress % 25 == 0)

# === FORMATTING UTILITIES ===

def format_file_count(count: int) -> str:
    """Format file count with proper pluralization.
    
    Args:
        count: Number of files
        
    Returns:
        Formatted string
    """
    return f"{count:,} file{'s' if count != 1 else ''}"

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    return f"{seconds:.1f} detik" if seconds < 60 else f"{seconds/60:.1f} menit"

def get_log_emoji(level: str) -> str:
    """Get emoji for log level.
    
    Args:
        level: Log level (info, success, warning, error, debug)
        
    Returns:
        Emoji for the log level
    """
    emoji_map = {
        'info': 'ℹ️', 
        'success': '✅', 
        'warning': '⚠️', 
        'error': '❌', 
        'debug': '🔍'
    }
    return emoji_map.get(level, 'ℹ️')

def safe_get_widget_value(ui_components: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Safely get widget value from UI components.
    
    Args:
        ui_components: Dictionary UI components
        key: Widget key
        default: Default value if widget not found
        
    Returns:
        Widget value or default
    """
    widget = ui_components.get(key)
    if widget is None:
        return default
    
    if hasattr(widget, 'value'):
        return widget.value
    return default
