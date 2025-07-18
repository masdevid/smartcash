"""
File: smartcash/ui/model/train/components/config_summary.py
Description: Configuration summary component for training UI.
"""

import html
import ipywidgets as widgets
from typing import Dict, Any
from smartcash.ui.logger import get_module_logger
from ..constants import generate_model_name


def create_config_summary(config: Dict[str, Any]) -> widgets.Widget:
    """Create configuration summary panel.
    
    Args:
        config: Training configuration dictionary
        
    Returns:
        Widget containing the configuration summary
    """
    logger = get_module_logger("smartcash.ui.model.train.components")
    
    try:
        if not isinstance(config, dict):
            raise ValueError(f"Expected config to be a dictionary, got {type(config).__name__}")
            
        logger.debug(f"Creating config summary with config keys: {list(config.keys())}")
        
        training_config = config.get('training', {})
        if not isinstance(training_config, dict):
            logger.warning(f"Expected training_config to be a dictionary, got {type(training_config).__name__}")
            training_config = {}
            
        backbone_integration = config.get('backbone_integration', {})
        if not isinstance(backbone_integration, dict):
            logger.warning(f"Expected backbone_integration to be a dictionary, got {type(backbone_integration).__name__}")
            backbone_integration = {}
        
        # Generate model name preview
        backbone_type = backbone_integration.get('backbone_type', 'efficientnet_b4')
        layer_mode = training_config.get('layer_mode', 'single')
        optimization_type = training_config.get('optimization_type', 'default')
        model_name = generate_model_name(backbone_type, layer_mode, optimization_type)
        
        # Create summary content
        summary_html = f"""
        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #007bff;">
            <h4 style="margin-top: 0; color: #495057;">🚀 Training Configuration Summary</h4>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 15px;">
                <div>
                    <h5 style="color: #6c757d; margin-bottom: 8px;">📋 Training Settings</h5>
                    <p><strong>Mode:</strong> {training_config.get('layer_mode', 'single').title()}</p>
                    <p><strong>Epochs:</strong> {training_config.get('epochs', 100)}</p>
                    <p><strong>Batch Size:</strong> {training_config.get('batch_size', 16)}</p>
                    <p><strong>Learning Rate:</strong> {training_config.get('learning_rate', 0.001)}</p>
                    <p><strong>Optimization:</strong> {training_config.get('optimization_type', 'default').title()}</p>
                </div>
                
                <div>
                    <h5 style="color: #6c757d; margin-bottom: 8px;">🧬 Model Information</h5>
                    <p><strong>Backbone:</strong> {backbone_type}</p>
                    <p><strong>Model Name:</strong> <code>{model_name}</code></p>
                    <p><strong>Mixed Precision:</strong> {'✅' if training_config.get('mixed_precision', True) else '❌'}</p>
                    <p><strong>Early Stopping:</strong> {'✅' if training_config.get('early_stopping', {{}}).get('enabled', True) else '❌'}</p>
                </div>
            </div>
            
            <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #dee2e6;">
                <h5 style="color: #6c757d; margin-bottom: 8px;">📊 Monitoring Features</h5>
                <div style="display: flex; gap: 20px;">
                    <span>📈 Live Loss Chart</span>
                    <span>📊 Live mAP Chart</span>
                    <span>🔄 Real-time Progress</span>
                    <span>💾 Auto-save Best Model</span>
                    <span>🏆 Final Metrics Table</span>
                </div>
            </div>
            
            <div style="margin-top: 15px; padding: 10px; background: #e3f2fd; border-radius: 4px;">
                <small style="color: #1976d2;">
                    <strong>📝 Note:</strong> Training will continue from backbone configuration. 
                    Best model will be saved as <code>{model_name}</code> and will overwrite any existing model with the same name.
                </small>
            </div>
        </div>
        """
        
        summary_container = widgets.HTML(
            value=summary_html,
            layout=widgets.Layout(width='100%', margin='10px 0')
        )
        
        logger.debug("✅ Configuration summary created successfully")
        return summary_container
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Failed to create configuration summary: {e}\n{error_details}")
        
        # Return a more detailed error message for debugging
        error_html = """
        <div style="background: #fff5f5; padding: 15px; border-radius: 8px; border-left: 4px solid #ff4d4f;">
            <h4 style="margin-top: 0; color: #cf1322;">⚠️ Configuration Summary Unavailable</h4>
            <p>Failed to generate configuration summary due to an error:</p>
            <pre style="background: #f8f8f8; padding: 10px; border-radius: 4px; overflow-x: auto;">
{error}
            </pre>
            <p>Please check the logs for more details.</p>
        </div>
        """.format(error=html.escape(str(e)))
        
        return widgets.HTML(error_html)


# For backward compatibility
_create_configuration_summary = create_config_summary
