"""
File: smartcash/ui/dataset/augment/components/operation_summary.py
Description: Operation summary widget - NEW summary_container component

This is the new summary_container component requested in the refactoring.
It provides operation status, statistics, and real-time feedback.
"""

import ipywidgets as widgets
from ipywidgets import HTML, Layout, VBox, HBox, Output, Button, IntProgress, FloatProgress
from IPython.display import display
from typing import Dict, Any, Optional
from smartcash.ui.core.errors.handlers import handle_ui_errors
from ..constants import (
    AUGMENT_COLORS, SECTION_STYLES, SUCCESS_MESSAGES, ERROR_MESSAGES,
    WARNING_MESSAGES, PROGRESS_PHASES
)


def _create_status_badge(status: str, message: str) -> widgets.HTML:
    """Create status badge with appropriate styling."""
    status_colors = {
        'success': AUGMENT_COLORS['success'],
        'error': AUGMENT_COLORS['danger'],
        'warning': AUGMENT_COLORS['warning'],
        'info': AUGMENT_COLORS['info'],
        'pending': AUGMENT_COLORS['light']
    }
    
    color = status_colors.get(status, AUGMENT_COLORS['info'])
    
    return widgets.HTML(f"""
    <div style="display: inline-block; padding: 4px 8px; margin: 2px; 
                background-color: {color}; color: white; border-radius: 12px; 
                font-size: 10px; font-weight: 600;">
        {message}
    </div>
    """)


def _create_progress_bar(progress: float, phase: str = "") -> widgets.HTML:
    """Create custom progress bar with phase information."""
    progress_percent = min(100, max(0, progress * 100))
    
    return widgets.HTML(f"""
    <div style="margin: 8px 0;">
        <div style="font-size: 10px; color: #666; margin-bottom: 4px;">
            {phase} - {progress_percent:.1f}%
        </div>
        <div style="width: 100%; background-color: #f0f0f0; border-radius: 8px; height: 8px;">
            <div style="width: {progress_percent}%; background-color: {AUGMENT_COLORS['primary']}; 
                        height: 100%; border-radius: 8px; transition: width 0.3s ease;"></div>
        </div>
    </div>
    """)


@handle_ui_errors(error_component_title="Operation Summary Creation Error")
def create_operation_summary_widget() -> Dict[str, Any]:
    """
    Create a horizontally-optimized operation summary widget.
    
    Features:
    - 📊 Real-time operation status and progress in a compact horizontal layout
    - 📈 Dataset statistics and metrics displayed side by side
    - 🔄 Processing phases with visual progress indicators
    - ✅ Success/error status tracking with color-coded badges
    - 📝 Operation history and logs in a collapsible section
    
    Returns:
        Dictionary containing container, widgets, and metadata
    """
    
    # Create summary widgets with compact styling
    widgets_dict = {
        # Status indicators (top row)
        'operation_status': widgets.HTML(
            value="<div style='font-size: 11px; color: #666;'>Status: <span style='color: #28a745;'>Ready</span></div>",
            layout=widgets.Layout(width='100%', margin='2px 0')
        ),
        
        # Progress tracking (expanded to show more details)
        'operation_progress': widgets.HTML(
            value="<div style='font-size: 11px; color: #666;'>Progress: <span style='color: #007bff;'>0%</span></div>",
            layout=widgets.Layout(width='100%', margin='2px 0')
        ),
        
        # Current phase with icon
        'current_phase': widgets.HTML(
            value="<div style='font-size: 10px; color: #6c757d;'><i>Initializing...</i></div>",
            layout=widgets.Layout(width='100%', margin='2px 0')
        ),
        
        # Statistics display (more compact)
        'dataset_stats': widgets.HTML(
            value="""
            <div style='font-size: 10px; color: #333; padding: 8px; 
                        background: #f8f9fa; border-radius: 6px; margin: 2px 0;
                        border-left: 3px solid #6f42c1;'>
                <div style='font-weight: 600; color: #6f42c1; margin-bottom: 4px;'>
                    <i class='fas fa-database'></i> Dataset
                </div>
                <div style='display: flex; justify-content: space-between;'>
                    <div>Original: <b>--</b></div>
                    <div>Target: <b>--</b></div>
                    <div>Classes: <b>--</b></div>
                </div>
            </div>
            """,
            layout=widgets.Layout(width='100%', margin='4px 0')
        ),
        
        # Operation metrics (more compact)
        'operation_metrics': widgets.HTML(
            value="""
            <div style='font-size: 10px; color: #333; padding: 8px; 
                        background: #e7f5ff; border-radius: 6px; margin: 2px 0;
                        border-left: 3px solid #0d6efd;'>
                <div style='font-weight: 600; color: #0d6efd; margin-bottom: 4px;'>
                    <i class='fas fa-tachometer-alt'></i> Metrics
                </div>
                <div style='display: flex; justify-content: space-between;'>
                    <div>Time: <b>--</b></div>
                    <div>Processed: <b>--</b></div>
                    <div>Success: <b>--</b></div>
                </div>
            </div>
            """,
            layout=widgets.Layout(width='100%', margin='4px 0')
        ),
        
        # Recent activity log (more compact)
        'activity_log': widgets.HTML(
            value='''
            <div style='font-size: 10px; color: #333; padding: 8px; 
                        background: #f8f9fa; border-radius: 6px; margin: 2px 0;
                        border-left: 3px solid #fd7e14;'>
                <div style='font-weight: 600; color: #fd7e14; margin-bottom: 4px;'>
                    <i class='fas fa-history'></i> Recent Activity
                </div>
                <div style='font-family: monospace; font-size: 9px; line-height: 1.3;'>
                    Ready for augmentation operation...
                </div>
            </div>
            ''',
            layout=widgets.Layout(width='100%', margin='4px 0')
        )
    }
    
    # Create main container with horizontal layout
    left_column = widgets.VBox([
        # Status section
        widgets.HTML(f"""
        <div style='color: {AUGMENT_COLORS["primary"]}; margin: 4px 0 8px 0; 
                    font-size: 12px; font-weight: 600;'>
            <i class='fas fa-tasks'></i> Operation Summary
        </div>
        """),
        widgets_dict['operation_status'],
        widgets_dict['operation_progress'],
        widgets_dict['current_phase'],
        widgets.HTML("<div style='height: 10px;'></div>")  # Spacer
    ], layout=widgets.Layout(
        width='30%',
        padding='8px',
        border_right='1px solid #eee',
        min_width='200px'
    ))
    
    middle_column = widgets.VBox([
        # Dataset stats and metrics
        widgets.HTML(f"""
        <div style='color: {AUGMENT_COLORS["info"]}; margin: 4px 0 8px 0; 
                    font-size: 12px; font-weight: 600;'>
            <i class='fas fa-chart-bar'></i> Metrics
        </div>
        """),
        widgets_dict['dataset_stats'],
        widgets_dict['operation_metrics']
    ], layout=widgets.Layout(
        width='35%',
        padding='8px',
        border_right='1px solid #eee',
        min_width='200px'
    ))
    
    right_column = widgets.VBox([
        # Activity log
        widgets.HTML(f"""
        <div style='color: {AUGMENT_COLORS["success"]}; margin: 4px 0 8px 0; 
                    font-size: 12px; font-weight: 600;'>
            <i class='fas fa-history'></i> Activity Log
        </div>
        """),
        widgets_dict['activity_log']
    ], layout=widgets.Layout(
        width='35%',
        padding='8px',
        min_width='250px',
        overflow_y='auto'
    ))
    
    # Create main container with horizontal layout
    container = widgets.HBox(
        [left_column, middle_column, right_column],
        layout=widgets.Layout(
            width='100%',
            padding='0',
            display='flex',
            flex_flow='row wrap',
            align_items='flex-start',
            justify_content='space-between',
            border='1px solid #e0e0e0',
            borderRadius='8px',
            overflow='hidden',
            margin='0 0 10px 0'
        )
    )
    
    # Summary update methods
    def update_status(status: str, message: str):
        """Update operation status with color coding."""
        status_colors = {
            'success': '#28a745',
            'error': '#dc3545',
            'warning': '#ffc107',
            'info': '#17a2b8',
            'pending': '#6c757d'
        }
        color = status_colors.get(status.lower(), '#6c757d')
        widgets_dict['operation_status'].value = f"""
        <div style='font-size: 11px;'>
            Status: <span style='color: {color}; font-weight: 500;'>{message}</span>
        </div>
        """
    
    def update_progress(progress: float, phase: str = "Processing"):
        """Update operation progress with visual indicator."""
        progress_percent = min(100, max(0, int(progress * 100)))
        widgets_dict['operation_progress'].value = f"""
        <div style='font-size: 11px;'>
            Progress: <span style='color: #0d6efd;'>{progress_percent}%</span>
            <div style='width: 100%; height: 6px; background: #e9ecef; border-radius: 3px; margin-top: 4px; overflow: hidden;'>
                <div style='width: {progress_percent}%; height: 100%; background: #0d6efd; transition: width 0.3s;'></div>
            </div>
        </div>
        """
        widgets_dict['current_phase'].value = f"""
        <div style='font-size: 10px; color: #6c757d;'>
            <i class='fas fa-sync-alt fa-spin' style='margin-right: 4px;'></i>
            {phase}
        </div>
        """
    
    def update_dataset_stats(original: int, target: int, classes: int):
        """Update dataset statistics in a compact format."""
        widgets_dict['dataset_stats'].value = f"""
        <div style='font-size: 10px; color: #333; padding: 8px; 
                    background: #f8f9fa; border-radius: 6px; margin: 2px 0;
                    border-left: 3px solid #6f42c1;'>
            <div style='font-weight: 600; color: #6f42c1; margin-bottom: 4px;'>
                <i class='fas fa-database'></i> Dataset
            </div>
            <div style='display: flex; justify-content: space-between;'>
                <div>Original: <b>{original:,}</b></div>
                <div>Target: <b>{target:,}</b></div>
                <div>Classes: <b>{classes}</b></div>
            </div>
        </div>
        """
    
    def update_operation_metrics(time_elapsed: str, processed: int, success_rate: float):
        """Update operation metrics with visual indicators."""
        success_color = '#28a745' if success_rate >= 90 else '#ffc107' if success_rate >= 70 else '#dc3545'
        widgets_dict['operation_metrics'].value = f"""
        <div style='font-size: 10px; color: #333; padding: 8px; 
                    background: #e7f5ff; border-radius: 6px; margin: 2px 0;
                    border-left: 3px solid #0d6efd;'>
            <div style='font-weight: 600; color: #0d6efd; margin-bottom: 4px;'>
                <i class='fas fa-tachometer-alt'></i> Metrics
            </div>
            <div style='display: flex; justify-content: space-between;'>
                <div>Time: <b>{time_elapsed}</b></div>
                <div>Processed: <b>{processed:,}</b></div>
                <div>Success: <b style='color: {success_color};'>{success_rate:.1f}%</b></div>
            </div>
        </div>
        """
    
    def log_activity(message: str, level: str = 'info') -> None:
        """Add a message to the activity log with timestamp and styling.
        
        Args:
            message: The message to log
            level: Log level ('info', 'success', 'warning', 'error')
        """
        import datetime
        from IPython.display import display, HTML
        
        level_colors = {
            'info': '#17a2b8',
            'success': '#28a745',
            'warning': '#ffc107',
            'error': '#dc3545'
        }
        
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        color = level_colors.get(level.lower(), '#6c757d')
        
        # Get current log and limit to last 5 entries
        current_log = widgets_dict['activity_log'].value
        entries = current_log.split('<div class="log-entry">')
        entries = [e for e in entries if e.strip()]
        entries = entries[-5:]  # Keep only last 5 entries
        
        # Add new entry
        new_entry = f"""
        <div class="log-entry" style="margin-bottom: 4px; border-left: 2px solid {color}; padding-left: 6px;">
            <div style="display: flex; justify-content: space-between; font-size: 9px; color: #6c757d;">
                <span>[{timestamp}]</span>
                <span style="color: {color}; font-weight: 500; text-transform: uppercase;">{level}</span>
            </div>
            <div style="font-size: 9px; color: #333; line-height: 1.3;">
                {message}
            </div>
        </div>
        """
        
        entries.append(new_entry)
        widgets_dict['activity_log'].value = "".join(entries)
    
    # Add Font Awesome for icons if not already loaded
    display(HTML('''
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
    <style>
        .log-entry {
            transition: all 0.3s ease;
            opacity: 0.9;
        }
        .log-entry:hover {
            opacity: 1;
            background-color: #f8f9fa;
        }
    </style>
    '''))
    
    # Initialize activity log with empty content
    widgets_dict['activity_log'].value = ''
    
    # Convert the container to HTML string
    html_content = ""
    for child in container.children:
        if hasattr(child, 'value'):
            html_content += child.value if child.value else ""
    
    # Create a simple HTML widget with the content
    html_widget = widgets.HTML(value=html_content)
    
    return {
        'container': html_widget,
        'widgets': widgets_dict,
        'update_status': update_status,
        'update_progress': update_progress,
        'update_dataset_stats': update_dataset_stats,
        'update_operation_metrics': update_operation_metrics,
        'log_activity': log_activity,
        'refresh': lambda: None,  # Add empty refresh method for compatibility
        'form_type': 'operation_summary',
        'component_version': '2.0.0',
        'is_summary_container': True,
        'real_time_updates': True
    }