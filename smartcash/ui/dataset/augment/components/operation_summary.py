"""
File: smartcash/ui/dataset/augment/components/operation_summary.py
Description: Operation summary widget - NEW summary_container component

This is the new summary_container component requested in the refactoring.
It provides operation status, statistics, and real-time feedback.
"""

import ipywidgets as widgets
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
    Create operation summary widget - NEW summary_container component.
    
    Features:
    - 📊 Real-time operation status and progress
    - 📈 Dataset statistics and metrics
    - 🔄 Processing phases with visual feedback
    - ✅ Success/error status tracking
    - 📝 Operation history and logs
    
    Returns:
        Dictionary containing container, widgets, and metadata
    """
    
    # Create summary widgets
    widgets_dict = {
        # Status indicators
        'operation_status': widgets.HTML(
            value=_create_status_badge('pending', 'Ready')._dom_classes,
            layout=widgets.Layout(width='100%', margin='4px 0')
        ),
        
        # Progress tracking
        'operation_progress': widgets.HTML(
            value=_create_progress_bar(0.0, 'Waiting for operation...')._dom_classes,
            layout=widgets.Layout(width='100%', margin='4px 0')
        ),
        
        # Current phase
        'current_phase': widgets.HTML(
            value="<div style='font-size: 11px; color: #666;'>Phase: Initialization</div>",
            layout=widgets.Layout(width='100%', margin='2px 0')
        ),
        
        # Statistics display
        'dataset_stats': widgets.HTML(
            value="""
            <div style='font-size: 10px; color: #333; padding: 6px; 
                        background: #f8f9fa; border-radius: 4px; margin: 4px 0;'>
                <strong>Dataset Statistics:</strong><br>
                • Original images: --<br>
                • Target images: --<br>
                • Classes detected: --
            </div>
            """,
            layout=widgets.Layout(width='100%')
        ),
        
        # Operation metrics
        'operation_metrics': widgets.HTML(
            value="""
            <div style='font-size: 10px; color: #333; padding: 6px; 
                        background: #f0f8ff; border-radius: 4px; margin: 4px 0;'>
                <strong>Operation Metrics:</strong><br>
                • Processing time: --<br>
                • Images processed: --<br>
                • Success rate: --
            </div>
            """,
            layout=widgets.Layout(width='100%')
        ),
        
        # Recent activity log
        'activity_log': widgets.Textarea(
            value='Ready for augmentation operation...',
            description='Activity:',
            disabled=True,
            layout=widgets.Layout(width='100%', height='60px'),
            style={'description_width': '60px'}
        )
    }
    
    # Create section headers
    main_header = widgets.HTML(f"""
    <h6 style='color: {AUGMENT_COLORS["primary"]}; margin: 6px 0; font-size: 12px; font-weight: 600;'>
        📊 Operation Summary
    </h6>
    """)
    
    status_header = widgets.HTML(f"""
    <h6 style='color: {AUGMENT_COLORS["info"]}; margin: 8px 0 4px 0; font-size: 11px; font-weight: 600;'>
        🔄 Current Status
    </h6>
    """)
    
    metrics_header = widgets.HTML(f"""
    <h6 style='color: {AUGMENT_COLORS["success"]}; margin: 8px 0 4px 0; font-size: 11px; font-weight: 600;'>
        📈 Statistics
    </h6>
    """)
    
    # Create container with organized layout
    container = widgets.VBox([
        main_header,
        
        # Status section
        status_header,
        widgets_dict['operation_status'],
        widgets_dict['operation_progress'],
        widgets_dict['current_phase'],
        
        # Metrics section
        metrics_header,
        widgets_dict['dataset_stats'],
        widgets_dict['operation_metrics'],
        
        # Activity log
        widgets_dict['activity_log']
        
    ], layout=widgets.Layout(
        width='100%',
        padding='10px',
        display='flex',
        flex_flow='column',
        align_items='stretch',
        gap='4px'
    ))
    
    # Summary update methods
    def update_status(status: str, message: str):
        """Update operation status."""
        widgets_dict['operation_status'].value = _create_status_badge(status, message)._dom_classes
    
    def update_progress(progress: float, phase: str = "Processing"):
        """Update operation progress."""
        widgets_dict['operation_progress'].value = _create_progress_bar(progress, phase)._dom_classes
        widgets_dict['current_phase'].value = f"<div style='font-size: 11px; color: #666;'>Phase: {phase}</div>"
    
    def update_dataset_stats(original: int, target: int, classes: int):
        """Update dataset statistics."""
        widgets_dict['dataset_stats'].value = f"""
        <div style='font-size: 10px; color: #333; padding: 6px; 
                    background: #f8f9fa; border-radius: 4px; margin: 4px 0;'>
            <strong>Dataset Statistics:</strong><br>
            • Original images: {original:,}<br>
            • Target images: {target:,}<br>
            • Classes detected: {classes}
        </div>
        """
    
    def update_operation_metrics(time_elapsed: str, processed: int, success_rate: float):
        """Update operation metrics."""
        widgets_dict['operation_metrics'].value = f"""
        <div style='font-size: 10px; color: #333; padding: 6px; 
                    background: #f0f8ff; border-radius: 4px; margin: 4px 0;'>
            <strong>Operation Metrics:</strong><br>
            • Processing time: {time_elapsed}<br>
            • Images processed: {processed:,}<br>
            • Success rate: {success_rate:.1f}%
        </div>
        """
    
    def add_activity(message: str):
        """Add activity to log."""
        current_log = widgets_dict['activity_log'].value
        new_log = f"{message}\n{current_log}"
        # Keep only last 5 entries
        log_lines = new_log.split('\n')[:5]
        widgets_dict['activity_log'].value = '\n'.join(log_lines)
    
    return {
        'container': container,
        'widgets': widgets_dict,
        
        # Update methods
        'update_methods': {
            'status': update_status,
            'progress': update_progress,
            'dataset_stats': update_dataset_stats,
            'operation_metrics': update_operation_metrics,
            'activity': add_activity
        },
        
        # Status tracking
        'status_types': ['pending', 'processing', 'success', 'error', 'warning'],
        'progress_phases': list(PROGRESS_PHASES.keys()),
        
        # Form metadata
        'form_type': 'operation_summary',
        'component_version': '2.0.0',
        'is_summary_container': True,  # NEW: Mark as summary container
        'real_time_updates': True
    }