"""
File: smartcash/ui/model/train/components/training_metrics.py
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
        <h4 style="margin-top: 0; color: #495057; text-align: center;">🏆 Final Training Results</h4>
        
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
                        {metrics.get('val_map50', 0.0):.3f}
                    </td>
                    <td style="padding: 10px; border: 1px solid #ddd; text-align: center;">
                        {_get_quality_indicator(metrics.get('val_map50', 0.0), 'map')}
                    </td>
                </tr>
                <tr style="background: #f8f9fa;">
                    <td style="padding: 10px; border: 1px solid #ddd; font-weight: 500;">📈 mAP@0.75</td>
                    <td style="padding: 10px; border: 1px solid #ddd; text-align: center; font-weight: 600; color: #007bff;">
                        {metrics.get('val_map75', 0.0):.3f}
                    </td>
                    <td style="padding: 10px; border: 1px solid #ddd; text-align: center;">
                        {_get_quality_indicator(metrics.get('val_map75', 0.0), 'map')}
                    </td>
                </tr>
                <tr style="background: white;">
                    <td style="padding: 10px; border: 1px solid #ddd; font-weight: 500;">🎯 Accuracy</td>
                    <td style="padding: 10px; border: 1px solid #ddd; text-align: center; font-weight: 600; color: #28a745;">
                        {metrics.get('accuracy', 0.0):.3f}
                    </td>
                    <td style="padding: 10px; border: 1px solid #ddd; text-align: center;">
                        {_get_quality_indicator(metrics.get('accuracy', 0.0), 'accuracy')}
                    </td>
                </tr>
                <tr style="background: #f8f9fa;">
                    <td style="padding: 10px; border: 1px solid #ddd; font-weight: 500;">🔍 Precision</td>
                    <td style="padding: 10px; border: 1px solid #ddd; text-align: center; font-weight: 600; color: #17a2b8;">
                        {metrics.get('precision', 0.0):.3f}
                    </td>
                    <td style="padding: 10px; border: 1px solid #ddd; text-align: center;">
                        {_get_quality_indicator(metrics.get('precision', 0.0), 'precision')}
                    </td>
                </tr>
                <tr style="background: white;">
                    <td style="padding: 10px; border: 1px solid #ddd; font-weight: 500;">📋 Recall</td>
                    <td style="padding: 10px; border: 1px solid #ddd; text-align: center; font-weight: 600; color: #fd7e14;">
                        {metrics.get('recall', 0.0):.3f}
                    </td>
                    <td style="padding: 10px; border: 1px solid #ddd; text-align: center;">
                        {_get_quality_indicator(metrics.get('recall', 0.0), 'recall')}
                    </td>
                </tr>
                <tr style="background: #f8f9fa;">
                    <td style="padding: 10px; border: 1px solid #ddd; font-weight: 500;">⚖️ F1-Score</td>
                    <td style="padding: 10px; border: 1px solid #ddd; text-align: center; font-weight: 600; color: #6f42c1;">
                        {metrics.get('f1_score', 0.0):.3f}
                    </td>
                    <td style="padding: 10px; border: 1px solid #ddd; text-align: center;">
                        {_get_quality_indicator(metrics.get('f1_score', 0.0), 'f1')}
                    </td>
                </tr>
            </tbody>
        </table>
        
        <div style="margin-top: 15px; padding: 10px; background: #e3f2fd; border-radius: 4px;">
            <small style="color: #1976d2;">
                <strong>💡 Quality Indicators:</strong> 
                🟢 Excellent (>0.8) | 🟡 Good (0.6-0.8) | 🟠 Fair (0.4-0.6) | 🔴 Poor (<0.4)
            </small>
        </div>
    </div>
    """


def get_quality_indicator(value: float, metric_type: str) -> str:
    """Get quality indicator emoji based on metric value and type.
    
    Args:
        value: The metric value (0.0 to 1.0)
        metric_type: Type of the metric (e.g., 'accuracy', 'precision', 'recall', 'f1', 'map')
        
    Returns:
        String with emoji and quality description
    """
    if value >= 0.8:
        return "🟢 Excellent"
    elif value >= 0.6:
        return "🟡 Good"
    elif value >= 0.4:
        return "🟠 Fair"
    else:
        return "🔴 Poor"


def get_initial_metrics_html() -> str:
    """Get initial HTML for metrics panel when no training has completed.
    
    Returns:
        HTML string for the initial metrics panel state
    """
    return """
    <div style="padding: 20px; text-align: center; color: #666;">
        <h4 style="margin-top: 0;">🎯 Training Metrics</h4>
        <p>Training metrics will appear here after completion</p>
        <div style="margin-top: 15px; padding: 10px; background: #f8f9fa; border-radius: 4px;">
            <small>Expected metrics: mAP, Accuracy, Precision, Recall, F1-Score</small>
        </div>
    </div>
    """


# For backward compatibility
_generate_metrics_table_html = generate_metrics_table_html
_get_quality_indicator = get_quality_indicator
_get_initial_metrics_html = get_initial_metrics_html
