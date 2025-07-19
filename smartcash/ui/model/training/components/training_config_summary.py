"""
File: smartcash/ui/model/training/components/training_config_summary.py
Description: Configuration summary component for training UI.
"""

import html
import ipywidgets as widgets
from typing import Dict, Any
from smartcash.ui.logger import get_module_logger


def create_config_summary(config: Dict[str, Any]) -> widgets.Widget:
    """Create configuration summary panel.
    
    Args:
        config: Training configuration dictionary
        
    Returns:
        Widget containing the configuration summary
    """
    logger = get_module_logger("smartcash.ui.model.training.components")
    
    try:
        if not isinstance(config, dict):
            raise ValueError(f"Expected config to be a dictionary, got {type(config).__name__}")
            
        logger.debug(f"Creating config summary with config keys: {list(config.keys())}")
        
        training_config = config.get('training', {})
        model_selection = config.get('model_selection', {})
        data_config = config.get('data', {})
        monitoring_config = config.get('monitoring', {})
        
        # Helper function to safely get nested config values
        def get_nested_value(d, keys, default=None):
            if not isinstance(d, dict):
                return default
                
            for key in keys:
                if isinstance(d, dict) and key in d:
                    d = d[key]
                else:
                    return default
            return d
        
        # Helper function to format boolean values
        def format_bool(value):
            return "✅ Enabled" if value else "❌ Disabled"
        
        # Model configuration summary
        model_info = f"""
        <div style="background: #e3f2fd; padding: 12px; border-radius: 6px; margin-bottom: 10px;">
            <h5 style="margin: 0 0 8px 0; color: #1976d2;">🏗️ Model Configuration</h5>
            <div style="font-size: 13px; color: #424242;">
                <strong>Source:</strong> {html.escape(str(model_selection.get('source', 'backbone')))} |
                <strong>Backbone:</strong> {html.escape(str(model_selection.get('backbone_type', 'Not Selected')))} |
                <strong>Classes:</strong> {model_selection.get('num_classes', 7)} |
                <strong>Input Size:</strong> {model_selection.get('input_size', 640)}px
            </div>
            {f'<div style="font-size: 12px; color: #666; margin-top: 4px;"><strong>Checkpoint:</strong> {html.escape(str(model_selection.get("checkpoint_path", "None")))}</div>' if model_selection.get('checkpoint_path') else ''}
        </div>
        """
        
        # Training configuration summary
        training_info = f"""
        <div style="background: #f3e5f5; padding: 12px; border-radius: 6px; margin-bottom: 10px;">
            <h5 style="margin: 0 0 8px 0; color: #7b1fa2;">🚀 Training Parameters</h5>
            <div style="font-size: 13px; color: #424242;">
                <strong>Epochs:</strong> {training_config.get('epochs', 100)} |
                <strong>Batch Size:</strong> {training_config.get('batch_size', 16)} |
                <strong>Learning Rate:</strong> {training_config.get('learning_rate', 0.001):.4f} |
                <strong>Optimizer:</strong> {html.escape(str(training_config.get('optimizer', 'adam')).title())}
            </div>
            <div style="font-size: 12px; color: #666; margin-top: 4px;">
                <strong>Scheduler:</strong> {html.escape(str(training_config.get('scheduler', 'cosine')).title())} |
                <strong>Weight Decay:</strong> {training_config.get('weight_decay', 0.0005):.4f} |
                <strong>Mixed Precision:</strong> {format_bool(training_config.get('mixed_precision', True))}
            </div>
        </div>
        """
        
        # Data configuration summary
        data_info = f"""
        <div style="background: #e8f5e8; padding: 12px; border-radius: 6px; margin-bottom: 10px;">
            <h5 style="margin: 0 0 8px 0; color: #388e3c;">📊 Data Configuration</h5>
            <div style="font-size: 13px; color: #424242;">
                <strong>Train Split:</strong> {data_config.get('train_split', 0.8):.1%} |
                <strong>Val Split:</strong> {data_config.get('val_split', 0.2):.1%} |
                <strong>Workers:</strong> {data_config.get('workers', 4)} |
                <strong>Augmentation:</strong> {format_bool(data_config.get('augmentation', {}).get('enabled', True))}
            </div>
        </div>
        """
        
        # Early stopping and monitoring configuration
        early_stopping = training_config.get('early_stopping', {})
        monitoring_info = f"""
        <div style="background: #fff3e0; padding: 12px; border-radius: 6px; margin-bottom: 10px;">
            <h5 style="margin: 0 0 8px 0; color: #f57c00;">⚙️ Advanced Settings</h5>
            <div style="font-size: 13px; color: #424242;">
                <strong>Early Stopping:</strong> {format_bool(early_stopping.get('enabled', True))} |
                <strong>Patience:</strong> {early_stopping.get('patience', 15)} epochs |
                <strong>Save Period:</strong> {training_config.get('save_period', 10)} epochs
            </div>
            <div style="font-size: 12px; color: #666; margin-top: 4px;">
                <strong>Primary Metric:</strong> {monitoring_config.get('primary_metric', 'mAP@0.5')} |
                <strong>Charts:</strong> {format_bool(config.get('charts', {}).get('enabled', True))}
            </div>
        </div>
        """
        
        # Output configuration
        output_config = config.get('output', {})
        output_info = f"""
        <div style="background: #fce4ec; padding: 12px; border-radius: 6px; margin-bottom: 10px;">
            <h5 style="margin: 0 0 8px 0; color: #c2185b;">💾 Output Settings</h5>
            <div style="font-size: 13px; color: #424242;">
                <strong>Save Directory:</strong> {html.escape(str(output_config.get('save_dir', 'runs/training')))} |
                <strong>Name:</strong> {html.escape(str(output_config.get('name', 'smartcash_training')))}
            </div>
            <div style="font-size: 12px; color: #666; margin-top: 4px;">
                <strong>Save TXT:</strong> {format_bool(output_config.get('save_txt', True))} |
                <strong>Save Config:</strong> {format_bool(output_config.get('save_conf', True))}
            </div>
        </div>
        """
        
        # Device configuration
        device_config = config.get('device', {})
        device_info = f"""
        <div style="background: #e1f5fe; padding: 12px; border-radius: 6px; margin-bottom: 10px;">
            <h5 style="margin: 0 0 8px 0; color: #0277bd;">🖥️ Device Settings</h5>
            <div style="font-size: 13px; color: #424242;">
                <strong>Auto Detect:</strong> {format_bool(device_config.get('auto_detect', True))} |
                <strong>Preferred:</strong> {html.escape(str(device_config.get('preferred', 'auto')).upper())}
            </div>
        </div>
        """
        
        # Create complete summary HTML
        summary_html = f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 100%; overflow-x: auto;">
            <h4 style="margin-top: 0; color: #333; text-align: center;">📋 Training Configuration Summary</h4>
            {model_info}
            {training_info}
            {data_info}
            {monitoring_info}
            {output_info}
            {device_info}
            <div style="text-align: center; padding: 8px; background: #f5f5f5; border-radius: 4px; margin-top: 10px;">
                <small style="color: #666;">
                    Configuration updated: {_get_current_timestamp()} | 
                    Total Sections: {len([x for x in [training_config, model_selection, data_config] if x])}
                </small>
            </div>
        </div>
        """
        
        # Create widget
        summary_widget = widgets.HTML(value=summary_html)
        
        # Add update method
        def update_summary(new_config: Dict[str, Any]):
            """Update summary with new configuration."""
            try:
                updated_widget = create_config_summary(new_config)
                summary_widget.value = updated_widget.value
                logger.debug("✅ Configuration summary updated")
            except Exception as e:
                logger.error(f"Failed to update config summary: {e}")
        
        summary_widget.update_summary = update_summary
        
        logger.debug("✅ Configuration summary created successfully")
        return summary_widget
        
    except Exception as e:
        logger.error(f"Failed to create config summary: {e}")
        
        # Return error widget
        error_html = f"""
        <div style="background: #ffebee; border: 1px solid #f44336; padding: 15px; border-radius: 6px; color: #c62828;">
            <h5 style="margin: 0 0 8px 0;">❌ Configuration Summary Error</h5>
            <p style="margin: 0; font-size: 13px;">Failed to create configuration summary: {html.escape(str(e))}</p>
        </div>
        """
        return widgets.HTML(value=error_html)


def create_simple_config_summary(config: Dict[str, Any]) -> widgets.Widget:
    """Create a simplified configuration summary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Simplified summary widget
    """
    training_config = config.get('training', {})
    model_selection = config.get('model_selection', {})
    
    summary_html = f"""
    <div style="background: #f8f9fa; padding: 15px; border-radius: 6px; border-left: 4px solid #007bff;">
        <h5 style="margin: 0 0 10px 0; color: #495057;">📋 Quick Summary</h5>
        <div style="font-size: 13px; color: #6c757d;">
            <strong>Model:</strong> {model_selection.get('backbone_type', 'Not Selected')} | 
            <strong>Epochs:</strong> {training_config.get('epochs', 100)} | 
            <strong>Batch Size:</strong> {training_config.get('batch_size', 16)} | 
            <strong>LR:</strong> {training_config.get('learning_rate', 0.001):.4f}
        </div>
    </div>
    """
    
    return widgets.HTML(value=summary_html)


def _get_current_timestamp() -> str:
    """Get current timestamp for summary."""
    try:
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "Unknown"


def _format_config_value(value: Any, max_length: int = 30) -> str:
    """Format configuration value for display."""
    try:
        str_value = str(value)
        if len(str_value) > max_length:
            return str_value[:max_length-3] + "..."
        return str_value
    except Exception:
        return "N/A"


# For backward compatibility
create_configuration_summary = create_config_summary