"""
File: smartcash/ui/model/training/components/training_metrics.py
Description: Training metrics visualization and utilities for the training UI.
"""

from typing import Dict, Any


def generate_metrics_table_html(metrics: Dict[str, float]) -> str:
    """Generate HTML table for training metrics results with intelligent layer support.
    
    Args:
        metrics: Dictionary containing metric names and their values (pre-filtered by intelligent layer detection)
        
    Returns:
        HTML string containing the formatted metrics table
    """
    # Extract key metrics with fallbacks and calculate accuracy if not present
    # Note: metrics have already been filtered by intelligent layer detection
    accuracy = metrics.get('val_accuracy', metrics.get('accuracy', 0.0))
    precision = metrics.get('val_precision', metrics.get('precision', 0.0))
    recall = metrics.get('val_recall', metrics.get('recall', 0.0))
    f1_score = metrics.get('val_f1', metrics.get('f1_score', metrics.get('f1', 0.0)))
    map_50 = metrics.get('val_map50', metrics.get('mAP@0.5', 0.0))
    map_75 = metrics.get('val_map75', metrics.get('mAP@0.75', 0.0))
    train_loss = metrics.get('train_loss', 0.0)
    val_loss = metrics.get('val_loss', 0.0)
    
    # Calculate average mAP if not present
    avg_map = (map_50 + map_75) / 2 if (map_50 > 0 or map_75 > 0) else metrics.get('mAP', 0.0)
    
    return f"""
    <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
        <h4 style="margin-top: 0; color: #495057; text-align: center;">📊 Training Results & Performance Metrics</h4>
        
        <!-- Primary Metrics Cards -->
        <div style="display: flex; flex-wrap: wrap; gap: 15px; margin: 20px 0; justify-content: space-between;">
            <div style="flex: 1; min-width: 200px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 16px; border-radius: 8px; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.15);">
                <div style="font-size: 28px; margin-bottom: 8px;">🎯</div>
                <div style="font-size: 14px; opacity: 0.9; margin-bottom: 4px;">Accuracy</div>
                <div style="font-size: 24px; font-weight: 700;">{accuracy:.1%}</div>
                <div style="font-size: 11px; opacity: 0.8; margin-top: 4px;">{get_quality_indicator(accuracy, 'accuracy').split('>')[1].split('<')[0] if '>' in get_quality_indicator(accuracy, 'accuracy') else 'N/A'}</div>
            </div>
            
            <div style="flex: 1; min-width: 200px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 16px; border-radius: 8px; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.15);">
                <div style="font-size: 28px; margin-bottom: 8px;">🎯</div>
                <div style="font-size: 14px; opacity: 0.9; margin-bottom: 4px;">Precision</div>
                <div style="font-size: 24px; font-weight: 700;">{precision:.3f}</div>
                <div style="font-size: 11px; opacity: 0.8; margin-top: 4px;">{get_quality_indicator(precision, 'precision').split('>')[1].split('<')[0] if '>' in get_quality_indicator(precision, 'precision') else 'N/A'}</div>
            </div>
            
            <div style="flex: 1; min-width: 200px; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 16px; border-radius: 8px; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.15);">
                <div style="font-size: 28px; margin-bottom: 8px;">📋</div>
                <div style="font-size: 14px; opacity: 0.9; margin-bottom: 4px;">Recall</div>
                <div style="font-size: 24px; font-weight: 700;">{recall:.3f}</div>
                <div style="font-size: 11px; opacity: 0.8; margin-top: 4px;">{get_quality_indicator(recall, 'recall').split('>')[1].split('<')[0] if '>' in get_quality_indicator(recall, 'recall') else 'N/A'}</div>
            </div>
        </div>
        
        <div style="display: flex; flex-wrap: wrap; gap: 15px; margin: 20px 0; justify-content: space-between;">
            <div style="flex: 1; min-width: 200px; background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); color: #333; padding: 16px; border-radius: 8px; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.15);">
                <div style="font-size: 28px; margin-bottom: 8px;">⚖️</div>
                <div style="font-size: 14px; opacity: 0.8; margin-bottom: 4px;">F1-Score</div>
                <div style="font-size: 24px; font-weight: 700;">{f1_score:.3f}</div>
                <div style="font-size: 11px; opacity: 0.7; margin-top: 4px;">{get_quality_indicator(f1_score, 'f1').split('>')[1].split('<')[0] if '>' in get_quality_indicator(f1_score, 'f1') else 'N/A'}</div>
            </div>
            
            <div style="flex: 1; min-width: 200px; background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); color: #333; padding: 16px; border-radius: 8px; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.15);">
                <div style="font-size: 28px; margin-bottom: 8px;">📊</div>
                <div style="font-size: 14px; opacity: 0.8; margin-bottom: 4px;">mAP</div>
                <div style="font-size: 24px; font-weight: 700;">{avg_map:.3f}</div>
                <div style="font-size: 11px; opacity: 0.7; margin-top: 4px;">{get_quality_indicator(avg_map, 'map').split('>')[1].split('<')[0] if '>' in get_quality_indicator(avg_map, 'map') else 'N/A'}</div>
            </div>
            
            <div style="flex: 1; min-width: 200px; background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); color: #333; padding: 16px; border-radius: 8px; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.15);">
                <div style="font-size: 28px; margin-bottom: 8px;">📉</div>
                <div style="font-size: 14px; opacity: 0.8; margin-bottom: 4px;">Loss</div>
                <div style="font-size: 24px; font-weight: 700;">{val_loss if val_loss > 0 else train_loss:.4f}</div>
                <div style="font-size: 11px; opacity: 0.7; margin-top: 4px;">{get_quality_indicator(val_loss if val_loss > 0 else train_loss, 'loss').split('>')[1].split('<')[0] if '>' in get_quality_indicator(val_loss if val_loss > 0 else train_loss, 'loss') else 'N/A'}</div>
            </div>
        </div>
        
        <!-- Detailed Metrics Table -->
        <div style="margin-top: 25px;">
            <h5 style="color: #495057; margin-bottom: 15px; border-bottom: 2px solid #dee2e6; padding-bottom: 8px;">📋 Detailed Performance Breakdown</h5>
            <table style="width: 100%; border-collapse: collapse; background: white; border-radius: 6px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <thead>
                    <tr style="background: #007bff; color: white;">
                        <th style="padding: 12px; text-align: left; font-weight: 600;">Metric</th>
                        <th style="padding: 12px; text-align: center; font-weight: 600;">Value</th>
                        <th style="padding: 12px; text-align: center; font-weight: 600;">Quality</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="background: #f8f9fa;">
                        <td style="padding: 10px; border: 1px solid #dee2e6; font-weight: 500;">📊 mAP@0.5</td>
                        <td style="padding: 10px; border: 1px solid #dee2e6; text-align: center; font-weight: 600; color: #007bff;">{map_50:.3f}</td>
                        <td style="padding: 10px; border: 1px solid #dee2e6; text-align: center;">{get_quality_indicator(map_50, 'map')}</td>
                    </tr>
                    <tr style="background: white;">
                        <td style="padding: 10px; border: 1px solid #dee2e6; font-weight: 500;">📈 mAP@0.75</td>
                        <td style="padding: 10px; border: 1px solid #dee2e6; text-align: center; font-weight: 600; color: #007bff;">{map_75:.3f}</td>
                        <td style="padding: 10px; border: 1px solid #dee2e6; text-align: center;">{get_quality_indicator(map_75, 'map')}</td>
                    </tr>
                    <tr style="background: #f8f9fa;">
                        <td style="padding: 10px; border: 1px solid #dee2e6; font-weight: 500;">📉 Train Loss</td>
                        <td style="padding: 10px; border: 1px solid #dee2e6; text-align: center; font-weight: 600; color: #dc3545;">{train_loss:.4f}</td>
                        <td style="padding: 10px; border: 1px solid #dee2e6; text-align: center;">{get_quality_indicator(train_loss, 'loss')}</td>
                    </tr>
                    <tr style="background: white;">
                        <td style="padding: 10px; border: 1px solid #dee2e6; font-weight: 500;">📉 Val Loss</td>
                        <td style="padding: 10px; border: 1px solid #dee2e6; text-align: center; font-weight: 600; color: #dc3545;">{val_loss:.4f}</td>
                        <td style="padding: 10px; border: 1px solid #dee2e6; text-align: center;">{get_quality_indicator(val_loss, 'loss')}</td>
                    </tr>
                    <tr style="background: #f8f9fa;">
                        <td style="padding: 10px; border: 1px solid #dee2e6; font-weight: 500;">🕐 Epoch</td>
                        <td style="padding: 10px; border: 1px solid #dee2e6; text-align: center; font-weight: 600; color: #6c757d;">{int(metrics.get('epoch', 0))}</td>
                        <td style="padding: 10px; border: 1px solid #dee2e6; text-align: center;"><span style="color: #007bff;">📊</span></td>
                    </tr>
                </tbody>
            </table>
        </div>
        
        <div style="margin-top: 20px; padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 8px; color: white; text-align: center; box-shadow: 0 3px 6px rgba(0,0,0,0.16);">
            <div style="font-size: 18px; font-weight: 600; margin-bottom: 8px;">🏆 Overall Performance Grade</div>
            <div style="font-size: 24px; font-weight: 700; margin-bottom: 8px;">{_calculate_overall_grade(metrics)}</div>
            <div style="font-size: 12px; opacity: 0.9;">
                <strong>Model Size:</strong> {metrics.get('model_size_mb', 'N/A')} MB | 
                <strong>Inference Time:</strong> {metrics.get('inference_time_ms', 'N/A')} ms
            </div>
        </div>
    </div>
    """


def get_initial_metrics_html() -> str:
    """Get initial/placeholder HTML for metrics display."""
    return """
    <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
        <h4 style="margin-top: 0; color: #495057; text-align: center;">📊 Training Results & Performance Metrics</h4>
        <p style="text-align: center; color: #6c757d; margin-bottom: 25px;">Metrics will be displayed here after training completion or validation.</p>
        
        <!-- Primary Metrics Cards Placeholder -->
        <div style="display: flex; flex-wrap: wrap; gap: 15px; margin: 20px 0; justify-content: space-between;">
            <div style="flex: 1; min-width: 200px; background: linear-gradient(135deg, #e9ecef 0%, #ced4da 100%); color: #6c757d; padding: 16px; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div style="font-size: 28px; margin-bottom: 8px;">🎯</div>
                <div style="font-size: 14px; opacity: 0.8; margin-bottom: 4px;">Accuracy</div>
                <div style="font-size: 24px; font-weight: 700;">--.--%</div>
                <div style="font-size: 11px; opacity: 0.6; margin-top: 4px;">Awaiting Results</div>
            </div>
            
            <div style="flex: 1; min-width: 200px; background: linear-gradient(135deg, #e9ecef 0%, #ced4da 100%); color: #6c757d; padding: 16px; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div style="font-size: 28px; margin-bottom: 8px;">🎯</div>
                <div style="font-size: 14px; opacity: 0.8; margin-bottom: 4px;">Precision</div>
                <div style="font-size: 24px; font-weight: 700;">-.---</div>
                <div style="font-size: 11px; opacity: 0.6; margin-top: 4px;">Awaiting Results</div>
            </div>
            
            <div style="flex: 1; min-width: 200px; background: linear-gradient(135deg, #e9ecef 0%, #ced4da 100%); color: #6c757d; padding: 16px; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div style="font-size: 28px; margin-bottom: 8px;">📋</div>
                <div style="font-size: 14px; opacity: 0.8; margin-bottom: 4px;">Recall</div>
                <div style="font-size: 24px; font-weight: 700;">-.---</div>
                <div style="font-size: 11px; opacity: 0.6; margin-top: 4px;">Awaiting Results</div>
            </div>
        </div>
        
        <div style="display: flex; flex-wrap: wrap; gap: 15px; margin: 20px 0; justify-content: space-between;">
            <div style="flex: 1; min-width: 200px; background: linear-gradient(135deg, #e9ecef 0%, #ced4da 100%); color: #6c757d; padding: 16px; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div style="font-size: 28px; margin-bottom: 8px;">⚖️</div>
                <div style="font-size: 14px; opacity: 0.8; margin-bottom: 4px;">F1-Score</div>
                <div style="font-size: 24px; font-weight: 700;">-.---</div>
                <div style="font-size: 11px; opacity: 0.6; margin-top: 4px;">Awaiting Results</div>
            </div>
            
            <div style="flex: 1; min-width: 200px; background: linear-gradient(135deg, #e9ecef 0%, #ced4da 100%); color: #6c757d; padding: 16px; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div style="font-size: 28px; margin-bottom: 8px;">📊</div>
                <div style="font-size: 14px; opacity: 0.8; margin-bottom: 4px;">mAP</div>
                <div style="font-size: 24px; font-weight: 700;">-.---</div>
                <div style="font-size: 11px; opacity: 0.6; margin-top: 4px;">Awaiting Results</div>
            </div>
            
            <div style="flex: 1; min-width: 200px; background: linear-gradient(135deg, #e9ecef 0%, #ced4da 100%); color: #6c757d; padding: 16px; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div style="font-size: 28px; margin-bottom: 8px;">📉</div>
                <div style="font-size: 14px; opacity: 0.8; margin-bottom: 4px;">Loss</div>
                <div style="font-size: 24px; font-weight: 700;">-.----</div>
                <div style="font-size: 11px; opacity: 0.6; margin-top: 4px;">Awaiting Results</div>
            </div>
        </div>
        
        <div style="margin-top: 25px; padding: 15px; background: linear-gradient(135deg, #e9ecef 0%, #ced4da 100%); border-radius: 8px; color: #6c757d; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="font-size: 16px; font-weight: 600; margin-bottom: 8px;">🏆 Performance Evaluation</div>
            <div style="font-size: 14px; opacity: 0.8;">Start training or run validation to see comprehensive performance metrics</div>
            <div style="font-size: 12px; opacity: 0.7; margin-top: 8px;">
                Accuracy • Precision • Recall • F1-Score • mAP • Loss
            </div>
        </div>
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