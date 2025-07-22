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
        
        # Create 4-column card layout for configuration summary
        model_card = f"""
        <div style="flex: 1; background: #e3f2fd; padding: 16px; border-radius: 8px; margin: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); min-width: 200px;">
            <div style="text-align: center; margin-bottom: 12px;">
                <div style="font-size: 24px; margin-bottom: 4px;">🏗️</div>
                <h5 style="margin: 0; color: #1976d2; font-size: 14px; font-weight: 600;">Model Configuration</h5>
            </div>
            <div style="font-size: 12px; color: #424242; line-height: 1.4;">
                <div style="margin-bottom: 6px;"><strong>Backbone:</strong><br>{html.escape(str(model_selection.get('backbone_type', 'Not Selected')))}</div>
                <div style="margin-bottom: 6px;"><strong>Classes:</strong> {model_selection.get('num_classes', 7)}</div>
                <div style="margin-bottom: 6px;"><strong>Input Size:</strong> {model_selection.get('input_size', 640)}px</div>
                <div><strong>Source:</strong> {html.escape(str(model_selection.get('source', 'backbone')))}</div>
            </div>
        </div>
        """
        
        training_card = f"""
        <div style="flex: 1; background: #f3e5f5; padding: 16px; border-radius: 8px; margin: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); min-width: 200px;">
            <div style="text-align: center; margin-bottom: 12px;">
                <div style="font-size: 24px; margin-bottom: 4px;">🚀</div>
                <h5 style="margin: 0; color: #7b1fa2; font-size: 14px; font-weight: 600;">Training Parameters</h5>
            </div>
            <div style="font-size: 12px; color: #424242; line-height: 1.4;">
                <div style="margin-bottom: 6px;"><strong>Epochs:</strong> {training_config.get('epochs', 100)}</div>
                <div style="margin-bottom: 6px;"><strong>Batch Size:</strong> {training_config.get('batch_size', 16)}</div>
                <div style="margin-bottom: 6px;"><strong>Learning Rate:</strong> {training_config.get('learning_rate', 0.001):.4f}</div>
                <div><strong>Optimizer:</strong> {html.escape(str(training_config.get('optimizer', 'adam')).title())}</div>
            </div>
        </div>
        """
        
        # Display fixed data split (75/15/15)
        train_split = data_config.get('train_split', 0.75)
        val_split = data_config.get('val_split', 0.15)
        test_split = data_config.get('test_split', 0.15)
        
        data_card = f"""
        <div style="flex: 1; background: #e8f5e8; padding: 16px; border-radius: 8px; margin: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); min-width: 200px;">
            <div style="text-align: center; margin-bottom: 12px;">
                <div style="font-size: 24px; margin-bottom: 4px;">📊</div>
                <h5 style="margin: 0; color: #388e3c; font-size: 14px; font-weight: 600;">Data Configuration</h5>
            </div>
            <div style="font-size: 12px; color: #424242; line-height: 1.4;">
                <div style="margin-bottom: 6px;"><strong>Data Split (Fixed):</strong></div>
                <div style="margin-bottom: 4px; padding-left: 8px;">• Train: {train_split:.0%}</div>
                <div style="margin-bottom: 4px; padding-left: 8px;">• Valid: {val_split:.0%}</div>
                <div style="margin-bottom: 6px; padding-left: 8px;">• Test: {test_split:.0%}</div>
                <div style="margin-bottom: 6px;"><strong>Workers:</strong> {data_config.get('workers', 4)}</div>
                <div><strong>Augmentation:</strong> {format_bool(data_config.get('augmentation', {}).get('enabled', True))}</div>
            </div>
        </div>
        """
        
        monitoring_card = f"""
        <div style="flex: 1; background: #fff3e0; padding: 16px; border-radius: 8px; margin: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); min-width: 200px;">
            <div style="text-align: center; margin-bottom: 12px;">
                <div style="font-size: 24px; margin-bottom: 4px;">⚙️</div>
                <h5 style="margin: 0; color: #f57c00; font-size: 14px; font-weight: 600;">Advanced Settings</h5>
            </div>
            <div style="font-size: 12px; color: #424242; line-height: 1.4;">
                <div style="margin-bottom: 6px;"><strong>Early Stopping:</strong> {format_bool(training_config.get('early_stopping', {}).get('enabled', True))}</div>
                <div style="margin-bottom: 6px;"><strong>Patience:</strong> {training_config.get('early_stopping', {}).get('patience', 15)} epochs</div>
                <div style="margin-bottom: 6px;"><strong>Mixed Precision:</strong> {format_bool(training_config.get('mixed_precision', True))}</div>
                <div><strong>Save Period:</strong> {training_config.get('save_period', 10)} epochs</div>
            </div>
        </div>
        """
        
        # Create complete summary HTML with 4-column card layout
        summary_html = f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 100%; overflow-x: auto;">
            <h4 style="margin-top: 0; color: #333; text-align: center; margin-bottom: 20px;">📋 Training Configuration Summary</h4>
            <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: stretch; margin-bottom: 20px;">
                {model_card}
                {training_card}
                {data_card}
                {monitoring_card}
            </div>
            <div style="text-align: center; padding: 12px; background: #f8f9fa; border-radius: 6px; border: 1px solid #e9ecef;">
                <small style="color: #6c757d;">
                    <strong>Last Updated:</strong> {_get_current_timestamp()} | 
                    <strong>Device:</strong> {html.escape(str(config.get('device', {}).get('preferred', 'auto')).upper())} | 
                    <strong>Output Dir:</strong> {html.escape(str(config.get('output', {}).get('save_dir', 'runs/training')))}
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