"""
File: smartcash/ui/dataset/preprocess/components/operation_summary.py
Description: Operation summary component for preprocessing module
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional
from smartcash.ui.dataset.preprocess.constants import BANKNOTE_CLASSES


def create_operation_summary(config: Optional[Dict[str, Any]] = None) -> widgets.VBox:
    """
    Create operation summary component.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        VBox widget with operation summary
    """
    config = config or {}
    preprocessing_config = config.get('preprocessing', {})
    data_config = config.get('data', {})
    
    # Extract configuration details
    preset = preprocessing_config.get('normalization', {}).get('preset', 'yolov5s')
    target_splits = preprocessing_config.get('target_splits', ['train', 'valid'])
    batch_size = preprocessing_config.get('batch_size', 32)
    validation_enabled = preprocessing_config.get('validation', {}).get('enabled', False)
    cleanup_target = preprocessing_config.get('cleanup_target', 'preprocessed')
    
    # Create summary content
    summary_html = f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 8px; padding: 15px; margin: 10px 0; color: white;">
        <h4 style="margin: 0 0 10px 0; color: white;">📊 Konfigurasi Preprocessing</h4>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 10px;">
            <div>
                <h5 style="margin: 0 0 5px 0; color: #f0f8ff;">🎨 Normalisasi</h5>
                <ul style="margin: 5px 0; padding-left: 20px; font-size: 13px;">
                    <li><strong>Preset:</strong> {preset}</li>
                    <li><strong>Target Size:</strong> {'640x640' if preset == 'yolov5s' else '832x832' if preset == 'yolov5l' else '1024x1024' if preset == 'yolov5x' else '640x640'}</li>
                    <li><strong>Format:</strong> YOLO-compatible NPY + TXT</li>
                </ul>
            </div>
            
            <div>
                <h5 style="margin: 0 0 5px 0; color: #f0f8ff;">⚡ Processing</h5>
                <ul style="margin: 5px 0; padding-left: 20px; font-size: 13px;">
                    <li><strong>Target Splits:</strong> {', '.join(target_splits)}</li>
                    <li><strong>Batch Size:</strong> {batch_size}</li>
                    <li><strong>Validasi:</strong> {'Aktif' if validation_enabled else 'Minimal'}</li>
                </ul>
            </div>
        </div>
        
        <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.3);">
            <h5 style="margin: 0 0 5px 0; color: #f0f8ff;">🎯 Classes yang Didukung</h5>
            <div style="display: flex; flex-wrap: wrap; gap: 5px; font-size: 12px;">
                {_create_class_badges()}
            </div>
        </div>
        
        <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.3);">
            <p style="margin: 0; font-size: 12px; opacity: 0.9;">
                💾 <strong>Data Directory:</strong> {data_config.get('dir', 'data')}<br>
                📁 <strong>Output Directory:</strong> {data_config.get('preprocessed_dir', 'data/preprocessed')}<br>
                🗑️ <strong>Cleanup Target:</strong> {cleanup_target}
            </p>
        </div>
    </div>
    """
    
    summary_widget = widgets.HTML(value=summary_html)
    
    # Create container with update method
    container = widgets.VBox([summary_widget])
    
    def update_summary(new_config: Dict[str, Any]) -> None:
        """Update summary with new configuration."""
        updated_html = _generate_summary_html(new_config)
        summary_widget.value = updated_html
    
    # Attach update method to container
    container.update_summary = update_summary
    container.summary_widget = summary_widget
    
    return container


def _create_class_badges() -> str:
    """Create HTML badges for banknote classes."""
    badges = []
    for class_id, class_info in BANKNOTE_CLASSES.items():
        display = class_info['display']
        color = _get_class_color(class_id)
        badge = f"""
        <span style="background-color: {color}; color: white; padding: 2px 6px; 
                     border-radius: 12px; font-size: 11px; white-space: nowrap;">
            {display}
        </span>
        """
        badges.append(badge)
    
    return ''.join(badges)


def _get_class_color(class_id: int) -> str:
    """Get color for banknote class."""
    colors = [
        '#FF6B6B',  # Rp1K - Red
        '#4ECDC4',  # Rp2K - Teal
        '#45B7D1',  # Rp5K - Blue
        '#96CEB4',  # Rp10K - Green
        '#FFEAA7',  # Rp20K - Yellow
        '#DDA0DD',  # Rp50K - Plum
        '#FFB347'   # Rp100K - Orange
    ]
    return colors[class_id % len(colors)]


def _generate_summary_html(config: Dict[str, Any]) -> str:
    """Generate updated summary HTML."""
    preprocessing_config = config.get('preprocessing', {})
    data_config = config.get('data', {})
    
    preset = preprocessing_config.get('normalization', {}).get('preset', 'yolov5s')
    target_splits = preprocessing_config.get('target_splits', ['train', 'valid'])
    batch_size = preprocessing_config.get('batch_size', 32)
    validation_enabled = preprocessing_config.get('validation', {}).get('enabled', False)
    cleanup_target = preprocessing_config.get('cleanup_target', 'preprocessed')
    
    return f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 8px; padding: 15px; margin: 10px 0; color: white;">
        <h4 style="margin: 0 0 10px 0; color: white;">📊 Konfigurasi Preprocessing</h4>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 10px;">
            <div>
                <h5 style="margin: 0 0 5px 0; color: #f0f8ff;">🎨 Normalisasi</h5>
                <ul style="margin: 5px 0; padding-left: 20px; font-size: 13px;">
                    <li><strong>Preset:</strong> {preset}</li>
                    <li><strong>Target Size:</strong> {'640x640' if preset == 'yolov5s' else '832x832' if preset == 'yolov5l' else '1024x1024' if preset == 'yolov5x' else '640x640'}</li>
                    <li><strong>Format:</strong> YOLO-compatible NPY + TXT</li>
                </ul>
            </div>
            
            <div>
                <h5 style="margin: 0 0 5px 0; color: #f0f8ff;">⚡ Processing</h5>
                <ul style="margin: 5px 0; padding-left: 20px; font-size: 13px;">
                    <li><strong>Target Splits:</strong> {', '.join(target_splits)}</li>
                    <li><strong>Batch Size:</strong> {batch_size}</li>
                    <li><strong>Validasi:</strong> {'Aktif' if validation_enabled else 'Minimal'}</li>
                </ul>
            </div>
        </div>
        
        <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.3);">
            <h5 style="margin: 0 0 5px 0; color: #f0f8ff;">🎯 Classes yang Didukung</h5>
            <div style="display: flex; flex-wrap: wrap; gap: 5px; font-size: 12px;">
                {_create_class_badges()}
            </div>
        </div>
        
        <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.3);">
            <p style="margin: 0; font-size: 12px; opacity: 0.9;">
                💾 <strong>Data Directory:</strong> {data_config.get('dir', 'data')}<br>
                📁 <strong>Output Directory:</strong> {data_config.get('preprocessed_dir', 'data/preprocessed')}<br>
                🗑️ <strong>Cleanup Target:</strong> {cleanup_target}
            </p>
        </div>
    </div>
    """


def create_operation_results_summary(results: Optional[Dict[str, Any]] = None) -> widgets.VBox:
    """
    Create operation results summary.
    
    Args:
        results: Operation results dictionary
        
    Returns:
        VBox widget with results summary
    """
    if not results:
        results = {}
    
    operation = results.get('operation', 'unknown')
    success = results.get('success', False)
    
    # Create different summaries based on operation type
    if operation == 'preprocess':
        return _create_preprocess_results_summary(results)
    elif operation == 'check':
        return _create_check_results_summary(results)
    elif operation == 'cleanup':
        return _create_cleanup_results_summary(results)
    else:
        return _create_generic_results_summary(results)


def _create_preprocess_results_summary(results: Dict[str, Any]) -> widgets.VBox:
    """Create preprocessing results summary."""
    stats = results.get('stats', {})
    config_info = results.get('configuration', {})
    
    total_files = stats.get('total_files', 0)
    processed_files = stats.get('processed_files', 0)
    preset = config_info.get('normalization_preset', 'default')
    target_splits = config_info.get('target_splits', [])
    
    html_content = f"""
    <div style="background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 6px; padding: 12px; margin: 10px 0;">
        <h5 style="color: #155724; margin: 0 0 8px 0;">✅ Preprocessing Completed</h5>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
            <div>
                <strong>📊 Processing Stats:</strong><br>
                • Total files: {total_files}<br>
                • Processed: {processed_files}<br>
                • Success rate: {(processed_files/total_files*100):.1f}% if {total_files} > 0 else 'N/A'
            </div>
            <div>
                <strong>⚙️ Configuration:</strong><br>
                • Preset: {preset}<br>
                • Splits: {', '.join(target_splits)}<br>
                • Format: YOLO NPY + TXT
            </div>
        </div>
    </div>
    """
    
    return widgets.VBox([widgets.HTML(value=html_content)])


def _create_check_results_summary(results: Dict[str, Any]) -> widgets.VBox:
    """Create check results summary."""
    service_ready = results.get('service_ready', False)
    file_stats = results.get('file_statistics', {})
    
    total_raw = sum(stats.get('raw_images', 0) for stats in file_stats.values())
    total_preprocessed = sum(stats.get('preprocessed_files', 0) for stats in file_stats.values())
    
    status_color = '#d4edda' if service_ready else '#fff3cd'
    status_border = '#c3e6cb' if service_ready else '#ffeaa7'
    status_text = '#155724' if service_ready else '#856404'
    status_icon = '✅' if service_ready else '⚠️'
    status_msg = 'Dataset Ready' if service_ready else 'Dataset Not Ready'
    
    html_content = f"""
    <div style="background-color: {status_color}; border: 1px solid {status_border}; border-radius: 6px; padding: 12px; margin: 10px 0;">
        <h5 style="color: {status_text}; margin: 0 0 8px 0;">{status_icon} {status_msg}</h5>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
            <div>
                <strong>📊 File Statistics:</strong><br>
                • Raw images: {total_raw}<br>
                • Preprocessed: {total_preprocessed}<br>
                • Ready for processing: {'Yes' if service_ready else 'No'}
            </div>
            <div>
                <strong>📁 By Split:</strong><br>
                {_format_split_stats(file_stats)}
            </div>
        </div>
    </div>
    """
    
    return widgets.VBox([widgets.HTML(value=html_content)])


def _create_cleanup_results_summary(results: Dict[str, Any]) -> widgets.VBox:
    """Create cleanup results summary."""
    files_removed = results.get('files_removed', 0)
    cleanup_target = results.get('cleanup_target', 'preprocessed')
    affected_splits = results.get('affected_splits', [])
    
    html_content = f"""
    <div style="background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 6px; padding: 12px; margin: 10px 0;">
        <h5 style="color: #721c24; margin: 0 0 8px 0;">🗑️ Cleanup Completed</h5>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
            <div>
                <strong>📊 Cleanup Stats:</strong><br>
                • Files removed: {files_removed}<br>
                • Target: {cleanup_target}<br>
                • Status: {'Success' if files_removed >= 0 else 'Failed'}
            </div>
            <div>
                <strong>📁 Affected Splits:</strong><br>
                • Splits: {', '.join(affected_splits) if affected_splits else 'None'}<br>
                • Operation: Cleanup {cleanup_target} files
            </div>
        </div>
    </div>
    """
    
    return widgets.VBox([widgets.HTML(value=html_content)])


def _create_generic_results_summary(results: Dict[str, Any]) -> widgets.VBox:
    """Create generic results summary."""
    success = results.get('success', False)
    message = results.get('message', 'Operation completed')
    operation = results.get('operation', 'unknown')
    
    status_color = '#d4edda' if success else '#f8d7da'
    status_border = '#c3e6cb' if success else '#f5c6cb'
    status_text = '#155724' if success else '#721c24'
    status_icon = '✅' if success else '❌'
    
    html_content = f"""
    <div style="background-color: {status_color}; border: 1px solid {status_border}; border-radius: 6px; padding: 12px; margin: 10px 0;">
        <h5 style="color: {status_text}; margin: 0 0 8px 0;">{status_icon} {operation.title()} Results</h5>
        <p style="margin: 0; color: {status_text};">{message}</p>
    </div>
    """
    
    return widgets.VBox([widgets.HTML(value=html_content)])


def _format_split_stats(file_stats: Dict[str, Any]) -> str:
    """Format split statistics for display."""
    if not file_stats:
        return "No data available"
    
    lines = []
    for split, stats in file_stats.items():
        raw_count = stats.get('raw_images', 0)
        processed_count = stats.get('preprocessed_files', 0)
        lines.append(f"• {split}: {raw_count} raw, {processed_count} processed")
    
    return '<br>'.join(lines) if lines else "No splits found"