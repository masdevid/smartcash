"""
File: smartcash/ui/model/train/components/training_charts.py
Description: Training visualization charts component for the training UI.
"""

import ipywidgets as widgets
from typing import Dict, Any
from smartcash.ui.logger import get_module_logger


def create_dual_charts_layout(chart_config: Dict[str, Any], ui_config: Dict[str, Any]) -> Dict[str, Any]:
    """Create dual live charts for loss and mAP monitoring.
    
    Args:
        chart_config: Configuration for the charts
        ui_config: UI configuration parameters
        
    Returns:
        Dictionary containing chart widgets and their update methods
    """
    logger = get_module_logger("smartcash.ui.model.train.components")
    
    try:
        # Create placeholder charts using simple HTML widgets
        loss_chart = widgets.HTML(
            value="""
            <div style="border: 2px solid #ff6b6b; padding: 20px; margin: 10px; border-radius: 8px; text-align: center;">
                <h4 style="color: #ff6b6b; margin-top: 0;">📈 Training Loss Chart</h4>
                <p>Real-time training and validation loss will be displayed here</p>
                <div style="background: #fff5f5; padding: 10px; border-radius: 4px; margin-top: 10px;">
                    <small>Metrics: Train Loss, Validation Loss</small>
                </div>
            </div>
            """,
            layout=widgets.Layout(width='100%', height='200px')
        )
        
        map_chart = widgets.HTML(
            value="""
            <div style="border: 2px solid #4ecdc4; padding: 20px; margin: 10px; border-radius: 8px; text-align: center;">
                <h4 style="color: #4ecdc4; margin-top: 0;">📊 mAP Performance Chart</h4>
                <p>Real-time mAP performance metrics will be displayed here</p>
                <div style="background: #f0fdfc; padding: 10px; border-radius: 4px; margin-top: 10px;">
                    <small>Metrics: mAP@0.5, mAP@0.75</small>
                </div>
            </div>
            """,
            layout=widgets.Layout(width='100%', height='200px')
        )
        
        # Add update methods to charts
        def update_loss_data(data):
            # Placeholder for live update functionality
            pass
        
        def update_map_data(data):
            # Placeholder for live update functionality  
            pass
            
        loss_chart.add_data = update_loss_data
        map_chart.add_data = update_map_data
        
        charts = {
            'loss_chart': loss_chart,
            'map_chart': map_chart
        }
        
        logger.debug("✅ Dual live charts created successfully")
        return charts
        
    except Exception as e:
        logger.error(f"Failed to create dual charts: {e}")
        raise


# For backward compatibility
_create_dual_charts_layout = create_dual_charts_layout
