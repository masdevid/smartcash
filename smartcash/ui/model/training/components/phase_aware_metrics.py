"""
File: smartcash/ui/model/training/components/phase_aware_metrics.py
Description: Phase-aware training metrics visualization with two-column layout for Phase 1 and Phase 2.
"""

from typing import Dict, Any, Optional, Tuple
from smartcash.model.training.utils.metrics_utils import filter_phase_relevant_metrics


def generate_phase_aware_metrics_html(
    metrics: Dict[str, float], 
    current_phase: int,
    training_mode: str = "two_phase"
) -> str:
    """Generate HTML for phase-aware metrics display with two-column layout.
    
    Args:
        metrics: Complete metrics dictionary from training
        current_phase: Current training phase (1 or 2)
        training_mode: Training mode ('two_phase' or 'single_phase')
        
    Returns:
        HTML string containing the phase-aware metrics display
    """
    # Filter metrics for current phase
    filtered_metrics = filter_phase_relevant_metrics(metrics, current_phase)
    
    if training_mode == "single_phase":
        return _generate_single_phase_metrics_html(filtered_metrics)
    else:
        return _generate_two_phase_metrics_html(metrics, current_phase, filtered_metrics)


def _generate_two_phase_metrics_html(
    complete_metrics: Dict[str, float],
    current_phase: int,
    filtered_metrics: Dict[str, float]
) -> str:
    """Generate two-column metrics display for two-phase training."""
    
    # Get Phase 1 and Phase 2 metrics
    phase_1_metrics = filter_phase_relevant_metrics(complete_metrics, 1)
    phase_2_metrics = complete_metrics  # Phase 2 shows all metrics
    
    # Determine active phase styling
    phase_1_active = current_phase == 1
    phase_2_active = current_phase == 2
    
    return f"""
    <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
        <h4 style="margin-top: 0; color: #495057; text-align: center;">📊 Phase-Aware Training Metrics</h4>
        
        <!-- Phase Indicator -->
        <div style="display: flex; justify-content: center; margin-bottom: 20px; gap: 10px;">
            <div style="padding: 8px 16px; border-radius: 20px; font-weight: 600; {'background: #007bff; color: white;' if phase_1_active else 'background: #e9ecef; color: #6c757d;'}">
                Phase 1: Layer Training
            </div>
            <div style="padding: 8px 16px; border-radius: 20px; font-weight: 600; {'background: #28a745; color: white;' if phase_2_active else 'background: #e9ecef; color: #6c757d;'}">
                Phase 2: Fine-tuning
            </div>
        </div>
        
        <!-- Two-Column Layout -->
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;">
            
            <!-- Phase 1 Column -->
            <div style="background: {'#e3f2fd' if phase_1_active else '#f8f9fa'}; border: {'2px solid #007bff' if phase_1_active else '1px solid #dee2e6'}; border-radius: 8px; padding: 15px;">
                <h5 style="margin-top: 0; color: #007bff; text-align: center; margin-bottom: 15px;">
                    🎯 Phase 1 Metrics
                    {'<span style="font-size: 12px; color: #28a745; font-weight: normal;"> (ACTIVE)</span>' if phase_1_active else ''}
                </h5>
                
                {_generate_phase_metrics_cards(phase_1_metrics, is_phase_1=True)}
            </div>
            
            <!-- Phase 2 Column -->
            <div style="background: {'#e8f5e8' if phase_2_active else '#f8f9fa'}; border: {'2px solid #28a745' if phase_2_active else '1px solid #dee2e6'}; border-radius: 8px; padding: 15px;">
                <h5 style="margin-top: 0; color: #28a745; text-align: center; margin-bottom: 15px;">
                    🚀 Phase 2 Metrics
                    {'<span style="font-size: 12px; color: #28a745; font-weight: normal;"> (ACTIVE)</span>' if phase_2_active else ''}
                </h5>
                
                {_generate_phase_metrics_cards(phase_2_metrics, is_phase_1=False)}
            </div>
        </div>
        
        <!-- Current Active Metrics Summary -->
        <div style="background: linear-gradient(135deg, {'#007bff' if phase_1_active else '#28a745'} 0%, {'#0056b3' if phase_1_active else '#1e7e34'} 100%); color: white; padding: 15px; border-radius: 8px; text-align: center;">
            <div style="font-size: 16px; font-weight: 600; margin-bottom: 8px;">
                📈 Current Focus: Phase {current_phase}
            </div>
            <div style="font-size: 14px; opacity: 0.9;">
                {_get_phase_description(current_phase)}
            </div>
            <div style="font-size: 12px; opacity: 0.8; margin-top: 8px;">
                {_get_active_metrics_summary(filtered_metrics)}
            </div>
        </div>
    </div>
    """


def _generate_single_phase_metrics_html(metrics: Dict[str, float]) -> str:
    """Generate metrics display for single-phase training."""
    
    return f"""
    <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
        <h4 style="margin-top: 0; color: #495057; text-align: center;">📊 Single-Phase Training Metrics</h4>
        
        <!-- Single Phase Indicator -->
        <div style="display: flex; justify-content: center; margin-bottom: 20px;">
            <div style="padding: 8px 16px; border-radius: 20px; font-weight: 600; background: #6f42c1; color: white;">
                Single Phase: Complete Training
            </div>
        </div>
        
        <!-- Metrics Display -->
        <div style="background: #fff; border: 2px solid #6f42c1; border-radius: 8px; padding: 20px;">
            {_generate_phase_metrics_cards(metrics, is_phase_1=False)}
        </div>
        
        <!-- Summary -->
        <div style="background: linear-gradient(135deg, #6f42c1 0%, #5a2d91 100%); color: white; padding: 15px; border-radius: 8px; text-align: center; margin-top: 15px;">
            <div style="font-size: 16px; font-weight: 600; margin-bottom: 8px;">
                🎯 Single-Phase Training Active
            </div>
            <div style="font-size: 14px; opacity: 0.9;">
                Training all layers simultaneously with full model optimization
            </div>
            <div style="font-size: 12px; opacity: 0.8; margin-top: 8px;">
                {_get_active_metrics_summary(metrics)}
            </div>
        </div>
    </div>
    """


def _generate_phase_metrics_cards(metrics: Dict[str, float], is_phase_1: bool) -> str:
    """Generate metric cards for a specific phase."""
    
    if is_phase_1:
        # Phase 1: Focus on core training metrics and layer_1 metrics
        core_metrics = [
            ('train_loss', '📉 Train Loss', 'loss'),
            ('val_loss', '📉 Val Loss', 'loss'),
        ]
        
        layer_metrics = [
            ('layer_1_accuracy', '🎯 Layer 1 Accuracy', 'accuracy'),
            ('layer_1_precision', '🎯 Layer 1 Precision', 'precision'),
            ('layer_1_recall', '📋 Layer 1 Recall', 'recall'),
            ('layer_1_f1', '⚖️ Layer 1 F1', 'f1'),
        ]
        
        all_metrics = core_metrics + layer_metrics
        card_color = '#007bff'
        
    else:
        # Phase 2: Show all validation metrics
        all_metrics = [
            ('train_loss', '📉 Train Loss', 'loss'),
            ('val_loss', '📉 Val Loss', 'loss'),
            ('val_map50', '📊 mAP@0.5', 'map'),
            ('val_precision', '🎯 Precision', 'precision'),
            ('val_recall', '📋 Recall', 'recall'),
            ('val_f1', '⚖️ F1-Score', 'f1'),
            ('val_accuracy', '🎯 Accuracy', 'accuracy'),
        ]
        card_color = '#28a745'
    
    cards_html = ""
    
    # Generate cards in rows of 2
    for i in range(0, len(all_metrics), 2):
        cards_html += '<div style="display: flex; gap: 10px; margin-bottom: 10px;">'
        
        for j in range(2):
            if i + j < len(all_metrics):
                metric_key, metric_label, metric_type = all_metrics[i + j]
                value = metrics.get(metric_key, 0.0)
                quality = _get_quality_indicator(value, metric_type)
                
                cards_html += f"""
                <div style="flex: 1; background: white; border: 1px solid #dee2e6; border-radius: 6px; padding: 12px; text-align: center;">
                    <div style="font-size: 12px; color: {card_color}; font-weight: 600; margin-bottom: 4px;">{metric_label}</div>
                    <div style="font-size: 18px; font-weight: 700; color: #333; margin-bottom: 2px;">{_format_metric_value(value, metric_type)}</div>
                    <div style="font-size: 10px;">{quality}</div>
                </div>
                """
            else:
                # Empty placeholder for alignment
                cards_html += '<div style="flex: 1;"></div>'
        
        cards_html += '</div>'
    
    return cards_html


def _get_phase_description(phase: int) -> str:
    """Get description for the current phase."""
    if phase == 1:
        return "Training Layer 1 (banknote detection) while Layer 2+3 are frozen. Focus on layer-specific metrics."
    elif phase == 2:
        return "Fine-tuning all layers with full model optimization. All validation metrics are relevant."
    else:
        return "Single-phase training with complete model optimization."


def _get_active_metrics_summary(metrics: Dict[str, float]) -> str:
    """Get summary of active metrics for display."""
    active_count = len([v for v in metrics.values() if isinstance(v, (int, float)) and v != 0])
    return f"Monitoring {active_count} active metrics for optimal training progress"


def _get_quality_indicator(value: float, metric_type: str) -> str:
    """Get quality indicator for a metric value."""
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


def _format_metric_value(value: float, metric_type: str) -> str:
    """Format metric value for display."""
    try:
        if metric_type in ['accuracy', 'precision', 'recall', 'f1', 'map']:
            return f"{value:.3f}" if value > 0 else "0.000"
        elif metric_type == 'loss':
            return f"{value:.4f}" if value > 0 else "0.0000"
        else:
            return f"{value:.3f}" if value > 0 else "0.000"
    except (ValueError, TypeError):
        return "N/A"


def get_initial_phase_aware_metrics_html(training_mode: str = "two_phase") -> str:
    """Get initial/placeholder HTML for phase-aware metrics display."""
    
    if training_mode == "single_phase":
        return """
        <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
            <h4 style="margin-top: 0; color: #495057; text-align: center;">📊 Single-Phase Training Metrics</h4>
            <p style="text-align: center; color: #6c757d; margin-bottom: 25px;">Metrics will be displayed here during training.</p>
            
            <div style="display: flex; justify-content: center; margin-bottom: 20px;">
                <div style="padding: 8px 16px; border-radius: 20px; font-weight: 600; background: #e9ecef; color: #6c757d;">
                    Single Phase: Awaiting Training Start
                </div>
            </div>
            
            <div style="background: #fff; border: 2px dashed #ced4da; border-radius: 8px; padding: 30px; text-align: center; color: #6c757d;">
                <div style="font-size: 24px; margin-bottom: 10px;">⏳</div>
                <div style="font-size: 16px; font-weight: 600; margin-bottom: 8px;">Ready to Start Training</div>
                <div style="font-size: 14px;">All metrics will appear here once training begins</div>
            </div>
        </div>
        """
    
    else:
        return """
        <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
            <h4 style="margin-top: 0; color: #495057; text-align: center;">📊 Phase-Aware Training Metrics</h4>
            <p style="text-align: center; color: #6c757d; margin-bottom: 25px;">Two-phase metrics will be displayed here during training.</p>
            
            <!-- Phase Indicator -->
            <div style="display: flex; justify-content: center; margin-bottom: 20px; gap: 10px;">
                <div style="padding: 8px 16px; border-radius: 20px; font-weight: 600; background: #e9ecef; color: #6c757d;">
                    Phase 1: Awaiting Start
                </div>
                <div style="padding: 8px 16px; border-radius: 20px; font-weight: 600; background: #e9ecef; color: #6c757d;">
                    Phase 2: Pending
                </div>
            </div>
            
            <!-- Two-Column Layout Placeholder -->
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;">
                
                <!-- Phase 1 Placeholder -->
                <div style="background: #f8f9fa; border: 1px dashed #ced4da; border-radius: 8px; padding: 20px; text-align: center;">
                    <h5 style="margin-top: 0; color: #007bff; margin-bottom: 15px;">🎯 Phase 1 Metrics</h5>
                    <div style="color: #6c757d; font-size: 14px;">
                        <div style="margin-bottom: 8px;">• Train/Val Loss</div>
                        <div style="margin-bottom: 8px;">• Layer 1 Accuracy</div>
                        <div style="margin-bottom: 8px;">• Layer 1 Precision/Recall</div>
                        <div>• Layer 1 F1-Score</div>
                    </div>
                </div>
                
                <!-- Phase 2 Placeholder -->
                <div style="background: #f8f9fa; border: 1px dashed #ced4da; border-radius: 8px; padding: 20px; text-align: center;">
                    <h5 style="margin-top: 0; color: #28a745; margin-bottom: 15px;">🚀 Phase 2 Metrics</h5>
                    <div style="color: #6c757d; font-size: 14px;">
                        <div style="margin-bottom: 8px;">• All Phase 1 Metrics</div>
                        <div style="margin-bottom: 8px;">• Validation mAP</div>
                        <div style="margin-bottom: 8px;">• Overall Precision/Recall</div>
                        <div>• Model Performance</div>
                    </div>
                </div>
            </div>
            
            <!-- Summary Placeholder -->
            <div style="background: #e9ecef; color: #6c757d; padding: 15px; border-radius: 8px; text-align: center;">
                <div style="font-size: 16px; font-weight: 600; margin-bottom: 8px;">⏳ Ready for Two-Phase Training</div>
                <div style="font-size: 14px;">Start training to see phase-specific metrics and progress</div>
            </div>
        </div>
        """


def create_live_metrics_update(
    metrics: Dict[str, float],
    current_phase: int,
    epoch: int,
    training_mode: str = "two_phase"
) -> Dict[str, Any]:
    """Create live metrics update data for UI components.
    
    Args:
        metrics: Current metrics from training
        current_phase: Current training phase
        epoch: Current epoch number
        training_mode: Training mode
        
    Returns:
        Dictionary with update data for UI
    """
    # Filter metrics for current phase
    filtered_metrics = filter_phase_relevant_metrics(metrics, current_phase)
    
    return {
        'html': generate_phase_aware_metrics_html(metrics, current_phase, training_mode),
        'metrics': filtered_metrics,
        'phase': current_phase,
        'epoch': epoch,
        'training_mode': training_mode,
        'active_metric_count': len([v for v in filtered_metrics.values() if isinstance(v, (int, float)) and v != 0]),
        'phase_description': _get_phase_description(current_phase),
        'timestamp': None  # Can be added if needed
    }