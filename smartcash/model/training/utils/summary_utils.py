#!/usr/bin/env python3
"""
Summary generation utilities for the unified training pipeline.

This module provides functions for generating markdown summaries and other
summary-related functionality to keep the main pipeline code clean and focused.
"""

import time
from typing import Dict, Any, Optional
from smartcash.common.logger import get_logger

logger = get_logger(__name__)


def generate_markdown_summary(
    config: Optional[Dict[str, Any]] = None,
    phase_results: Optional[Dict[str, Dict[str, Any]]] = None,
    training_session_id: Optional[str] = None,
    training_start_time: Optional[float] = None,
    visualization_results: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate markdown summary for UI display.
    
    Args:
        config: Training configuration dictionary
        phase_results: Dictionary of phase results
        training_session_id: Unique training session identifier
        training_start_time: Training start timestamp
        
    Returns:
        Markdown formatted summary string
    """
    try:
        if not isinstance(config, (dict, type(None))):
            logger.warning(f"Error generating markdown summary: 'config' must be a dictionary or None, but got {type(config).__name__}")
            return "# Training Summary\n\nError generating summary due to invalid configuration data."
        
        if not isinstance(phase_results, (dict, type(None))):
            logger.warning(f"Error generating markdown summary: 'phase_results' must be a dictionary or None, but got {type(phase_results).__name__}")
            return "# Training Summary\n\nError generating summary due to invalid phase results data."

        if not isinstance(visualization_results, (dict, type(None))):
            logger.warning(f"Error generating markdown summary: 'visualization_results' must be a dictionary or None, but got {type(visualization_results).__name__}")
            return "# Training Summary\n\nError generating summary due to invalid visualization results data."

        summary = ["# SmartCash Training Summary\n"]
        
        # Configuration section
        if config:
            summary.append("## Configuration")
            summary.append(f"- **Backbone**: {config.get('model', {}).get('backbone', 'N/A')}")
            summary.append(f"- **Training Phases**: {len(config.get('training_phases', {}))}")
            
            if 'training' in config and isinstance(config['training'], dict):
                training = config['training']
                summary.append(f"- **Loss Type**: {training.get('loss', {}).get('type', 'N/A')}")
                summary.append(f"- **Batch Size**: {training.get('data', {}).get('batch_size', 'Auto')}")
                summary.append(f"- **Learning Rate**: {training.get('learning_rate', 'N/A')}")
                
                if 'early_stopping' in training and isinstance(training['early_stopping'], dict):
                    es = training['early_stopping']
                    summary.append(f"- **Early Stopping**: {'Enabled' if es.get('enabled') else 'Disabled'}")
                    if es.get('enabled'):
                        summary.append(f"  - Patience: {es.get('patience', 'N/A')} epochs")
                        summary.append(f"  - Metric: {es.get('metric', 'N/A')}")
                        summary.append(f"  - Mode: {es.get('mode', 'N/A')}")
            
            summary.append("")
        
        # Phase results section
        if phase_results:
            summary.append("## Phase Results")
            for phase_name, result in phase_results.items():
                if isinstance(result, dict):
                    status = "✅ Success" if result.get('success') else "❌ Failed"
                    duration = result.get('duration', 0)
                    summary.append(f"- **{phase_name.replace('_', ' ').title()}**: {status} ({duration:.1f}s)")
            summary.append("")
        
        # Training session info section
        if training_session_id:
            summary.append("## Session Information")
            summary.append(f"- **Session ID**: `{training_session_id}`")
            if training_start_time:
                total_duration = time.time() - training_start_time
                summary.append(f"- **Total Duration**: {total_duration:.1f} seconds")
            summary.append("")

        # Visualization results section
        if visualization_results:
            summary.append("## Visualization Results")
            if visualization_results.get('dashboard_path'):
                summary.append(f"- **Comprehensive Dashboard**: {visualization_results['dashboard_path']}")
            
            if isinstance(visualization_results.get('currency_plots'), dict):
                summary.append("- **Currency Analysis Plots**:")
                for plot_name, plot_path in visualization_results['currency_plots'].items():
                    summary.append(f"  - {plot_name.replace('_', ' ').title()}: {plot_path}")
            
            if isinstance(visualization_results.get('layer_plots'), dict):
                summary.append("- **Layer Analysis Plots**:")
                for plot_name, plot_path in visualization_results['layer_plots'].items():
                    summary.append(f"  - {plot_name.replace('_', ' ').title()}: {plot_path}")
            
            if isinstance(visualization_results.get('class_plots'), dict):
                summary.append("- **Class Analysis Plots**:")
                for plot_name, plot_path in visualization_results['class_plots'].items():
                    summary.append(f"  - {plot_name.replace('_', ' ').title()}: {plot_path}")
            summary.append("")

        return "\n".join(summary)
        
    except Exception as e:
        logger.warning(f"Error generating markdown summary: {e}")
        return "# Training Summary\n\nError generating summary."

def generate_training_report(
    pipeline_summary: Dict[str, Any],
    final_training_result: Dict[str, Any],
    visualization_result: Dict[str, Any],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate comprehensive training report with all results.
    
    Args:
        pipeline_summary: Summary from progress tracker
        final_training_result: Final training phase results
        visualization_result: Visualization generation results
        config: Training configuration
        
    Returns:
        Comprehensive training report dictionary
    """
    try:
        report = {
            'timestamp': time.time(),
            'pipeline_summary': pipeline_summary,
            'training_results': final_training_result,
            'visualization_results': visualization_result,
            'configuration': config
        }
        
        # Add derived metrics
        if final_training_result.get('success'):
            best_metrics = final_training_result.get('best_metrics', {})
            report['performance_summary'] = {
                'best_map50': best_metrics.get('val_map50', 0),
                'final_train_loss': best_metrics.get('train_loss', 0),
                'final_val_loss': best_metrics.get('val_loss', 0),
                'epochs_completed': final_training_result.get('epochs_completed', 0),
                'early_stopped': best_metrics.get('early_stopped', False)
            }
            
            # Calculate average layer performance
            layer_accuracies = []
            layer_f1_scores = []
            for layer in ['layer_1', 'layer_2', 'layer_3']:
                acc = best_metrics.get(f'{layer}_accuracy', 0)
                f1 = best_metrics.get(f'{layer}_f1', 0)
                if acc > 0:
                    layer_accuracies.append(acc)
                if f1 > 0:
                    layer_f1_scores.append(f1)
            
            if layer_accuracies:
                report['performance_summary']['avg_accuracy'] = sum(layer_accuracies) / len(layer_accuracies)
            if layer_f1_scores:
                report['performance_summary']['avg_f1'] = sum(layer_f1_scores) / len(layer_f1_scores)
        
        # Add visualization summary
        if visualization_result.get('success'):
            report['visualization_summary'] = {
                'charts_generated': visualization_result.get('charts_count', 0),
                'session_id': visualization_result.get('session_id'),
                'visualization_directory': visualization_result.get('visualization_directory')
            }
        
        return report
        
    except Exception as e:
        logger.error(f"Error generating training report: {e}")
        return {
            'error': str(e),
            'timestamp': time.time(),
            'pipeline_summary': pipeline_summary
        }


def format_training_summary_for_display(report: Dict[str, Any]) -> str:
    """
    Format training report for console/UI display.
    
    Args:
        report: Training report from generate_training_report
        
    Returns:
        Formatted string for display
    """
    try:
        lines = []
        lines.append("=" * 60)
        lines.append("🎉 Training Summary")
        lines.append("=" * 60)
        
        # Performance summary
        if 'performance_summary' in report:
            perf = report['performance_summary']
            lines.append(f"📊 Performance Results:")
            lines.append(f"   • Best mAP@0.5: {perf.get('best_map50', 0):.4f}")
            lines.append(f"   • Final train loss: {perf.get('final_train_loss', 0):.4f}")
            lines.append(f"   • Final val loss: {perf.get('final_val_loss', 0):.4f}")
            lines.append(f"   • Epochs completed: {perf.get('epochs_completed', 0)}")
            
            if perf.get('avg_accuracy'):
                lines.append(f"   • Average accuracy: {perf['avg_accuracy']:.4f}")
            if perf.get('avg_f1'):
                lines.append(f"   • Average F1 score: {perf['avg_f1']:.4f}")
            
            if perf.get('early_stopped'):
                lines.append(f"   • Early stopped: Yes")
            
            lines.append("")
        
        # Pipeline summary
        if 'pipeline_summary' in report:
            pipeline = report['pipeline_summary']
            lines.append(f"⏱️ Pipeline Summary:")
            lines.append(f"   • Total duration: {pipeline.get('total_duration', 0):.1f}s")
            lines.append(f"   • Phases completed: {pipeline.get('phases_completed', 0)}")
            lines.append(f"   • Success: {pipeline.get('success', False)}")
            lines.append("")
        
        # Visualization summary
        if 'visualization_summary' in report:
            viz = report['visualization_summary']
            lines.append(f"📊 Visualization Summary:")
            lines.append(f"   • Charts generated: {viz.get('charts_generated', 0)}")
            lines.append(f"   • Session ID: {viz.get('session_id', 'N/A')}")
            if viz.get('visualization_directory'):
                lines.append(f"   • Charts location: {viz['visualization_directory']}")
            lines.append("")
        
        lines.append("=" * 60)
        return "\n".join(lines)
        
    except Exception as e:
        logger.error(f"Error formatting training summary: {e}")
        return f"Error formatting summary: {e}"