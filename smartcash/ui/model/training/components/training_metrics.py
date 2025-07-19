"""
File: smartcash/ui/model/training/components/training_metrics.py
Description: Training metrics visualization and utilities for the training UI.
"""

from typing import Dict, Any


def generate_metrics_table_html(metrics: Dict[str, float]) -> str:
    """Generate HTML table for training metrics results.
    
    Args:
        metrics: Dictionary containing metric names and their values
        
    Returns:
        HTML string containing the formatted metrics table
    """
    return f"""
    <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
        <h4 style="margin-top: 0; color: #495057; text-align: center;">🏆 Training Results & Performance Metrics</h4>
        
        <table style="width: 100%; border-collapse: collapse; margin-top: 15px;">
            <thead>
                <tr style="background: #007bff; color: white;">
                    <th style="padding: 12px; text-align: left; border: 1px solid #ddd;">Metric</th>
                    <th style="padding: 12px; text-align: center; border: 1px solid #ddd;">Value</th>
                    <th style="padding: 12px; text-align: center; border: 1px solid #ddd;">Quality</th>
                </tr>
            </thead>
            <tbody>
                <tr style="background: white;">
                    <td style="padding: 10px; border: 1px solid #ddd; font-weight: 500;">📊 mAP@0.5</td>
                    <td style="padding: 10px; border: 1px solid #ddd; text-align: center; font-weight: 600; color: #007bff;">
                        {metrics.get('mAP@0.5', metrics.get('val_map50', 0.0)):.3f}
                    </td>
                    <td style="padding: 10px; border: 1px solid #ddd; text-align: center;">
                        {get_quality_indicator(metrics.get('mAP@0.5', metrics.get('val_map50', 0.0)), 'map')}
                    </td>
                </tr>
                <tr style="background: #f8f9fa;">
                    <td style="padding: 10px; border: 1px solid #ddd; font-weight: 500;">📈 mAP@0.75</td>
                    <td style="padding: 10px; border: 1px solid #ddd; text-align: center; font-weight: 600; color: #007bff;">
                        {metrics.get('mAP@0.75', metrics.get('val_map75', 0.0)):.3f}
                    </td>
                    <td style="padding: 10px; border: 1px solid #ddd; text-align: center;">
                        {get_quality_indicator(metrics.get('mAP@0.75', metrics.get('val_map75', 0.0)), 'map')}
                    </td>
                </tr>
                <tr style="background: white;">
                    <td style="padding: 10px; border: 1px solid #ddd; font-weight: 500;">🎯 Precision</td>
                    <td style="padding: 10px; border: 1px solid #ddd; text-align: center; font-weight: 600; color: #17a2b8;">
                        {metrics.get('precision', 0.0):.3f}
                    </td>
                    <td style="padding: 10px; border: 1px solid #ddd; text-align: center;">
                        {get_quality_indicator(metrics.get('precision', 0.0), 'precision')}
                    </td>
                </tr>
                <tr style="background: #f8f9fa;">
                    <td style="padding: 10px; border: 1px solid #ddd; font-weight: 500;">📋 Recall</td>
                    <td style="padding: 10px; border: 1px solid #ddd; text-align: center; font-weight: 600; color: #fd7e14;">
                        {metrics.get('recall', 0.0):.3f}
                    </td>
                    <td style="padding: 10px; border: 1px solid #ddd; text-align: center;">
                        {get_quality_indicator(metrics.get('recall', 0.0), 'recall')}
                    </td>
                </tr>
                <tr style="background: white;">
                    <td style="padding: 10px; border: 1px solid #ddd; font-weight: 500;">⚖️ F1-Score</td>
                    <td style="padding: 10px; border: 1px solid #ddd; text-align: center; font-weight: 600; color: #6f42c1;">
                        {metrics.get('f1_score', 0.0):.3f}
                    </td>
                    <td style="padding: 10px; border: 1px solid #ddd; text-align: center;">
                        {get_quality_indicator(metrics.get('f1_score', 0.0), 'f1')}
                    </td>
                </tr>
                <tr style="background: #f8f9fa;">
                    <td style="padding: 10px; border: 1px solid #ddd; font-weight: 500;">📉 Train Loss</td>
                    <td style="padding: 10px; border: 1px solid #ddd; text-align: center; font-weight: 600; color: #dc3545;">
                        {metrics.get('train_loss', 0.0):.4f}
                    </td>
                    <td style="padding: 10px; border: 1px solid #ddd; text-align: center;">
                        {get_quality_indicator(metrics.get('train_loss', 0.0), 'loss')}
                    </td>
                </tr>
                <tr style="background: white;">
                    <td style="padding: 10px; border: 1px solid #ddd; font-weight: 500;">📉 Val Loss</td>
                    <td style="padding: 10px; border: 1px solid #ddd; text-align: center; font-weight: 600; color: #dc3545;">
                        {metrics.get('val_loss', 0.0):.4f}
                    </td>
                    <td style="padding: 10px; border: 1px solid #ddd; text-align: center;">
                        {get_quality_indicator(metrics.get('val_loss', 0.0), 'loss')}
                    </td>
                </tr>
                <tr style="background: #f8f9fa;">
                    <td style="padding: 10px; border: 1px solid #ddd; font-weight: 500;">🕐 Epoch</td>
                    <td style="padding: 10px; border: 1px solid #ddd; text-align: center; font-weight: 600; color: #6c757d;">
                        {int(metrics.get('epoch', 0))}
                    </td>
                    <td style="padding: 10px; border: 1px solid #ddd; text-align: center;">
                        <span style="color: #007bff;">📊</span>
                    </td>
                </tr>
            </tbody>
        </table>
        
        <div style="margin-top: 15px; padding: 10px; background: white; border-radius: 4px; border-left: 4px solid #007bff;">
            <small style="color: #6c757d;">
                <strong>Overall Grade:</strong> {_calculate_overall_grade(metrics)} | 
                <strong>Model Size:</strong> {metrics.get('model_size_mb', 'N/A')} MB | 
                <strong>Inference Time:</strong> {metrics.get('inference_time_ms', 'N/A')} ms
            </small>
        </div>
    </div>
    """


def get_initial_metrics_html() -> str:
    """Get initial/placeholder HTML for metrics display."""
    return """
    <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; color: #6c757d;">
        <h4 style="margin-top: 0;">📊 Training Results</h4>
        <p>Training metrics will appear here after training completion or validation.</p>
        <div style="padding: 15px; background: white; border-radius: 4px; margin-top: 15px;">
            <div style="display: inline-block; margin: 0 15px;">
                <div style="font-size: 24px;">🎯</div>
                <div style="font-size: 12px;">Accuracy</div>
            </div>
            <div style="display: inline-block; margin: 0 15px;">
                <div style="font-size: 24px;">📊</div>
                <div style="font-size: 12px;">mAP</div>
            </div>
            <div style="display: inline-block; margin: 0 15px;">
                <div style="font-size: 24px;">📈</div>
                <div style="font-size: 12px;">Loss</div>
            </div>
            <div style="display: inline-block; margin: 0 15px;">
                <div style="font-size: 24px;">⚖️</div>
                <div style="font-size: 12px;">F1-Score</div>
            </div>
        </div>
        <small style="color: #adb5bd; margin-top: 10px; display: block;">
            Comprehensive performance metrics and model evaluation results
        </small>
    </div>
    """


def get_quality_indicator(value: float, metric_type: str) -> str:
    """Get quality indicator emoji and text for a metric value.
    
    Args:
        value: The metric value
        metric_type: Type of metric ('map', 'accuracy', 'precision', 'recall', 'f1', 'loss')
        
    Returns:
        HTML string with quality indicator
    """
    if metric_type == 'loss':
        # For loss, lower is better
        if value < 0.1:
            return '<span style="color: #28a745;">🟢 Excellent</span>'
        elif value < 0.3:
            return '<span style="color: #ffc107;">🟡 Good</span>'
        elif value < 0.5:
            return '<span style="color: #fd7e14;">🟠 Fair</span>'
        else:
            return '<span style="color: #dc3545;">🔴 Poor</span>'
    else:
        # For other metrics, higher is better
        if value >= 0.9:
            return '<span style="color: #28a745;">🟢 Excellent</span>'
        elif value >= 0.8:
            return '<span style="color: #20c997;">🟢 Very Good</span>'
        elif value >= 0.7:
            return '<span style="color: #ffc107;">🟡 Good</span>'
        elif value >= 0.6:
            return '<span style="color: #fd7e14;">🟠 Fair</span>'
        elif value >= 0.5:
            return '<span style="color: #dc3545;">🔴 Poor</span>'
        else:
            return '<span style="color: #6c757d;">⚫ Very Poor</span>'


def _calculate_overall_grade(metrics: Dict[str, Any]) -> str:
    """Calculate overall performance grade based on key metrics."""
    try:
        # Get key metrics with fallbacks
        map50 = metrics.get('mAP@0.5', metrics.get('val_map50', 0.0))
        precision = metrics.get('precision', 0.0)
        recall = metrics.get('recall', 0.0)
        f1_score = metrics.get('f1_score', 0.0)
        
        # Calculate weighted average (mAP has higher weight)
        weights = {'map50': 0.4, 'precision': 0.2, 'recall': 0.2, 'f1': 0.2}
        values = [map50, precision, recall, f1_score]
        
        if all(v == 0.0 for v in values):
            return "Not Available"
        
        weighted_score = (
            weights['map50'] * map50 +
            weights['precision'] * precision +
            weights['recall'] * recall +
            weights['f1'] * f1_score
        )
        
        # Convert to letter grade
        if weighted_score >= 0.9:
            return "A+ (Outstanding)"
        elif weighted_score >= 0.85:
            return "A (Excellent)"
        elif weighted_score >= 0.8:
            return "B+ (Very Good)"
        elif weighted_score >= 0.75:
            return "B (Good)"
        elif weighted_score >= 0.7:
            return "C+ (Above Average)"
        elif weighted_score >= 0.65:
            return "C (Average)"
        elif weighted_score >= 0.6:
            return "D+ (Below Average)"
        elif weighted_score >= 0.5:
            return "D (Poor)"
        else:
            return "F (Failing)"
            
    except Exception:
        return "Unable to Calculate"


def format_metric_value(value: Any, metric_type: str) -> str:
    """Format metric value for display.
    
    Args:
        value: The metric value
        metric_type: Type of metric for appropriate formatting
        
    Returns:
        Formatted string
    """
    try:
        if isinstance(value, (int, float)):
            if metric_type in ['map', 'accuracy', 'precision', 'recall', 'f1']:
                return f"{value:.3f}"
            elif metric_type == 'loss':
                return f"{value:.4f}"
            elif metric_type == 'epoch':
                return str(int(value))
            elif metric_type == 'time':
                return f"{value:.2f}s"
            else:
                return f"{value:.3f}"
        else:
            return str(value)
    except Exception:
        return "N/A"


def create_metrics_summary(metrics: Dict[str, Any]) -> Dict[str, str]:
    """Create a summary of key metrics for quick display.
    
    Args:
        metrics: Dictionary of metrics
        
    Returns:
        Dictionary with formatted summary metrics
    """
    return {
        'mAP@0.5': format_metric_value(metrics.get('mAP@0.5', metrics.get('val_map50', 0.0)), 'map'),
        'mAP@0.75': format_metric_value(metrics.get('mAP@0.75', metrics.get('val_map75', 0.0)), 'map'),
        'Precision': format_metric_value(metrics.get('precision', 0.0), 'precision'),
        'Recall': format_metric_value(metrics.get('recall', 0.0), 'recall'),
        'F1-Score': format_metric_value(metrics.get('f1_score', 0.0), 'f1'),
        'Train Loss': format_metric_value(metrics.get('train_loss', 0.0), 'loss'),
        'Val Loss': format_metric_value(metrics.get('val_loss', 0.0), 'loss'),
        'Overall Grade': _calculate_overall_grade(metrics)
    }


# Backward compatibility aliases
_get_quality_indicator = get_quality_indicator