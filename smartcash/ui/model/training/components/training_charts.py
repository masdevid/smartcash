"""
File: smartcash/ui/model/training/components/training_charts.py
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
    logger = get_module_logger("smartcash.ui.model.training.components")
    
    try:
        # Get chart configuration
        charts_config = chart_config.get('charts', {})
        loss_chart_config = charts_config.get('loss_chart', {})
        map_chart_config = charts_config.get('map_chart', {})
        
        # Create loss chart widget
        loss_chart = widgets.HTML(
            value=f"""
            <div style="border: 2px solid #ff6b6b; padding: 20px; margin: 10px; border-radius: 8px; text-align: center; height: 280px; background: linear-gradient(135deg, #fff5f5 0%, #ffffff 100%);">
                <h4 style="color: #ff6b6b; margin-top: 0; font-weight: bold;">📈 Training & Validation Loss</h4>
                <p style="color: #666; margin: 10px 0;">Real-time loss monitoring during training</p>
                <div style="background: #fff; padding: 15px; border-radius: 6px; margin-top: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <div style="display: flex; justify-content: space-around; margin-bottom: 10px;">
                        <div style="text-align: center;">
                            <div style="color: #ff6b6b; font-weight: bold; font-size: 14px;">Train Loss</div>
                            <div id="train-loss-value" style="color: #333; font-size: 18px; font-weight: bold;">-</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="color: #4ecdc4; font-weight: bold; font-size: 14px;">Val Loss</div>
                            <div id="val-loss-value" style="color: #333; font-size: 18px; font-weight: bold;">-</div>
                        </div>
                    </div>
                    <div style="color: #666; font-size: 12px;">
                        Epoch: <span id="loss-epoch-value">0</span> | 
                        Update Frequency: {loss_chart_config.get('update_frequency', 'epoch')}
                    </div>
                </div>
                <div style="margin-top: 10px; padding: 8px; background: #fff; border-radius: 4px; font-size: 11px; color: #888;">
                    Live chart will display training progress curves
                </div>
            </div>
            """,
            layout=widgets.Layout(width='48%', height='320px')
        )
        
        # Create mAP chart widget
        map_chart = widgets.HTML(
            value=f"""
            <div style="border: 2px solid #4ecdc4; padding: 20px; margin: 10px; border-radius: 8px; text-align: center; height: 280px; background: linear-gradient(135deg, #f0fdfc 0%, #ffffff 100%);">
                <h4 style="color: #4ecdc4; margin-top: 0; font-weight: bold;">📊 mAP Performance Metrics</h4>
                <p style="color: #666; margin: 10px 0;">Real-time performance evaluation</p>
                <div style="background: #fff; padding: 15px; border-radius: 6px; margin-top: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <div style="display: flex; justify-content: space-around; margin-bottom: 10px;">
                        <div style="text-align: center;">
                            <div style="color: #45b7d1; font-weight: bold; font-size: 14px;">mAP@0.5</div>
                            <div id="map50-value" style="color: #333; font-size: 18px; font-weight: bold;">-</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="color: #96ceb4; font-weight: bold; font-size: 14px;">mAP@0.75</div>
                            <div id="map75-value" style="color: #333; font-size: 18px; font-weight: bold;">-</div>
                        </div>
                    </div>
                    <div style="color: #666; font-size: 12px;">
                        Epoch: <span id="map-epoch-value">0</span> | 
                        Primary Metric: {chart_config.get('monitoring', {}).get('primary_metric', 'mAP@0.5')}
                    </div>
                </div>
                <div style="margin-top: 10px; padding: 8px; background: #fff; border-radius: 4px; font-size: 11px; color: #888;">
                    Live chart will display performance trends
                </div>
            </div>
            """,
            layout=widgets.Layout(width='48%', height='320px')
        )
        
        # Data storage for charts
        loss_data = {'epochs': [], 'train_loss': [], 'val_loss': []}
        map_data = {'epochs': [], 'map50': [], 'map75': []}
        
        # Add update methods to charts
        def update_loss_data(data: Dict[str, Any]):
            """Update loss chart with new training data (supports intelligent layer filtering)."""
            try:
                epoch = data.get('epoch', 0)
                train_loss = data.get('train_loss', 0.0)
                val_loss = data.get('val_loss', 0.0)
                
                # Update data storage
                if epoch not in loss_data['epochs']:
                    loss_data['epochs'].append(epoch)
                    loss_data['train_loss'].append(train_loss)
                    loss_data['val_loss'].append(val_loss)
                
                # Update HTML display
                updated_html = loss_chart.value
                updated_html = updated_html.replace(
                    'id="train-loss-value" style="color: #333; font-size: 18px; font-weight: bold;">-</div>',
                    f'id="train-loss-value" style="color: #333; font-size: 18px; font-weight: bold;">{train_loss:.4f}</div>'
                )
                updated_html = updated_html.replace(
                    'id="val-loss-value" style="color: #333; font-size: 18px; font-weight: bold;">-</div>',
                    f'id="val-loss-value" style="color: #333; font-size: 18px; font-weight: bold;">{val_loss:.4f}</div>'
                )
                updated_html = updated_html.replace(
                    'id="loss-epoch-value">0</span>',
                    f'id="loss-epoch-value">{epoch}</span>'
                )
                
                loss_chart.value = updated_html
                logger.debug(f"📈 Loss chart updated: epoch {epoch}, train_loss {train_loss:.4f}, val_loss {val_loss:.4f}")
                
            except Exception as e:
                logger.warning(f"Failed to update loss chart: {e}")
        
        def update_map_data(data: Dict[str, Any]):
            """Update mAP chart with new performance data."""
            try:
                epoch = data.get('epoch', 0)
                map50 = data.get('mAP@0.5', 0.0)
                map75 = data.get('mAP@0.75', 0.0)
                
                # Update data storage
                if epoch not in map_data['epochs']:
                    map_data['epochs'].append(epoch)
                    map_data['map50'].append(map50)
                    map_data['map75'].append(map75)
                
                # Update HTML display
                updated_html = map_chart.value
                updated_html = updated_html.replace(
                    'id="map50-value" style="color: #333; font-size: 18px; font-weight: bold;">-</div>',
                    f'id="map50-value" style="color: #333; font-size: 18px; font-weight: bold;">{map50:.3f}</div>'
                )
                updated_html = updated_html.replace(
                    'id="map75-value" style="color: #333; font-size: 18px; font-weight: bold;">-</div>',
                    f'id="map75-value" style="color: #333; font-size: 18px; font-weight: bold;">{map75:.3f}</div>'
                )
                updated_html = updated_html.replace(
                    'id="map-epoch-value">0</span>',
                    f'id="map-epoch-value">{epoch}</span>'
                )
                
                map_chart.value = updated_html
                logger.debug(f"📊 mAP chart updated: epoch {epoch}, mAP@0.5 {map50:.3f}, mAP@0.75 {map75:.3f}")
                
            except Exception as e:
                logger.warning(f"Failed to update mAP chart: {e}")
        
        def reset_charts():
            """Reset both charts to initial state."""
            try:
                loss_data.clear()
                loss_data.update({'epochs': [], 'train_loss': [], 'val_loss': []})
                map_data.clear()
                map_data.update({'epochs': [], 'map50': [], 'map75': []})
                
                # Reset HTML to initial state would require recreating the widgets
                logger.debug("📈 Charts reset to initial state")
                
            except Exception as e:
                logger.warning(f"Failed to reset charts: {e}")
        
        # Attach methods to chart widgets
        loss_chart.add_data = update_loss_data
        map_chart.add_data = update_map_data
        loss_chart.reset = reset_charts
        map_chart.reset = reset_charts
        
        # Store data for external access
        loss_chart.data = loss_data
        map_chart.data = map_data
        
        charts = {
            'loss_chart': loss_chart,
            'map_chart': map_chart,
            'update_loss': update_loss_data,
            'update_map': update_map_data,
            'reset_charts': reset_charts
        }
        
        logger.debug("✅ Dual live charts created successfully with update methods")
        return charts
        
    except Exception as e:
        logger.error(f"Failed to create dual charts: {e}")
        raise


def create_simple_chart_placeholder(title: str, metrics: list, color: str = "#333") -> widgets.Widget:
    """Create a simple chart placeholder widget.
    
    Args:
        title: Chart title
        metrics: List of metrics to display
        color: Primary color for the chart
        
    Returns:
        Chart placeholder widget
    """
    metrics_html = " | ".join(metrics)
    
    return widgets.HTML(
        value=f"""
        <div style="border: 2px solid {color}; padding: 15px; margin: 5px; border-radius: 6px; text-align: center;">
            <h5 style="color: {color}; margin-top: 0;">{title}</h5>
            <p style="color: #666; font-size: 12px;">Metrics: {metrics_html}</p>
            <div style="background: #f8f9fa; padding: 10px; border-radius: 4px; margin-top: 10px;">
                <small>Live updates will appear here during training</small>
            </div>
        </div>
        """,
        layout=widgets.Layout(width='100%', height='150px')
    )


# For backward compatibility
_create_dual_charts_layout = create_dual_charts_layout