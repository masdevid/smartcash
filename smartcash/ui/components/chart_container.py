"""
Reusable chart container component for displaying live metrics.
"""

from typing import Dict, Any, Optional, List, Union
import ipywidgets as widgets
from IPython.display import display, HTML
import json
import uuid

from smartcash.ui.components.base_component import BaseUIComponent


class ChartContainer(BaseUIComponent):
    """A reusable container for displaying live charts with metric data."""
    
    # Chart type options
    CHART_TYPES = {
        "line": "Line Chart",
        "bar": "Bar Chart", 
        "area": "Area Chart"
    }
    
    def __init__(self, 
                 component_name: str = "chart_container",
                 title: str = "Metrics Chart",
                 chart_type: str = "line",
                 columns: int = 1,
                 height: int = 400,
                 **kwargs):
        """Initialize the chart container.
        
        Args:
            component_name: Unique name for this component
            title: Title for the chart container
            chart_type: Type of chart (line, bar, area)
            columns: Number of chart columns (1 or 2)
            height: Height of the chart area in pixels
            **kwargs: Additional arguments to pass to BaseUIComponent
        """
        self._title = title
        self._chart_type = chart_type
        self._columns = min(max(columns, 1), 2)  # Limit to 1 or 2 columns
        self._height = height
        self._chart_data = {}
        self._chart_configs = {}
        
        # Initialize base class
        super().__init__(component_name, **kwargs)
    
    def _create_ui_components(self) -> None:
        """Create and initialize UI components."""
        # Create main container
        self._ui_components['main_container'] = widgets.VBox(
            layout=widgets.Layout(
                width='100%',
                border='1px solid #ddd',
                border_radius='8px',
                padding='15px',
                margin='10px 0'
            )
        )
        
        # Create title
        self._ui_components['title'] = widgets.HTML(
            f"<h3 style='margin: 0 0 15px 0; color: #2c3e50;'>{self._title}</h3>"
        )
        
        # Create chart type selector
        chart_options = list(self.CHART_TYPES.items())
        # Ensure default value is valid and find index
        if self._chart_type not in self.CHART_TYPES:
            self._chart_type = list(self.CHART_TYPES.keys())[0]
        
        # Find the index of the chart type in options
        default_index = 0
        for i, (key, _) in enumerate(chart_options):
            if key == self._chart_type:
                default_index = i
                break
        
        self._ui_components['chart_type_selector'] = widgets.Dropdown(
            options=chart_options,
            index=default_index,
            description='Chart Type:',
            style={'description_width': '80px'},
            layout=widgets.Layout(width='200px')
        )
        
        # Create charts container based on columns
        if self._columns == 1:
            self._create_single_column_layout()
        else:
            self._create_two_column_layout()
        
        # Create controls container
        controls_container = widgets.HBox([
            self._ui_components['chart_type_selector'],
            widgets.HTML("<div style='flex-grow: 1;'></div>"),  # Spacer
        ], layout=widgets.Layout(margin='0 0 15px 0'))
        
        # Assemble main container
        self._ui_components['main_container'].children = [
            self._ui_components['title'],
            controls_container,
            self._ui_components['charts_container']
        ]
        
        # Set up event handlers
        self._ui_components['chart_type_selector'].observe(
            self._on_chart_type_change, names='value'
        )
        
        # Set container reference
        self._ui_components['container'] = self._ui_components['main_container']
    
    def _create_single_column_layout(self) -> None:
        """Create single column chart layout."""
        self._ui_components['chart_1'] = self._create_chart_widget("chart_1")
        
        self._ui_components['charts_container'] = widgets.VBox([
            self._ui_components['chart_1']
        ], layout=widgets.Layout(width='100%'))
    
    def _create_two_column_layout(self) -> None:
        """Create two column chart layout."""
        self._ui_components['chart_1'] = self._create_chart_widget("chart_1", "Left Chart")
        self._ui_components['chart_2'] = self._create_chart_widget("chart_2", "Right Chart")
        
        # Create column containers
        left_column = widgets.VBox([
            self._ui_components['chart_1']
        ], layout=widgets.Layout(width='48%'))
        
        right_column = widgets.VBox([
            self._ui_components['chart_2']
        ], layout=widgets.Layout(width='48%'))
        
        self._ui_components['charts_container'] = widgets.HBox([
            left_column,
            widgets.HTML("<div style='width: 4%;'></div>"),  # Spacer
            right_column
        ], layout=widgets.Layout(width='100%'))
    
    def _create_chart_widget(self, chart_id: str, title: str = "Chart") -> widgets.Widget:
        """Create individual chart widget."""
        chart_html = f"""
        <div id="{chart_id}_{self.component_name}" style="
            width: 100%;
            height: {self._height}px;
            border: 1px solid #e1e5e9;
            border-radius: 6px;
            background: white;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        ">
            <div style="color: #6c757d; font-size: 18px; margin-bottom: 10px;">📊</div>
            <div style="color: #6c757d; font-size: 14px;">Waiting for data...</div>
            <div style="color: #6c757d; font-size: 12px; margin-top: 5px;">{title}</div>
        </div>
        
        <script>
        // Initialize chart data storage
        if (typeof window.chartData === 'undefined') {{
            window.chartData = {{}};
        }}
        window.chartData['{chart_id}_{self.component_name}'] = {{
            data: [],
            config: {{
                type: '{self._chart_type}',
                title: '{title}',
                color: '#4ecdc4'
            }}
        }};
        
        // Function to update chart
        window.updateChart_{chart_id}_{self.component_name} = function(data, config) {{
            const chartElement = document.getElementById('{chart_id}_{self.component_name}');
            if (!chartElement) return;
            
            // Store data
            window.chartData['{chart_id}_{self.component_name}'].data = data;
            window.chartData['{chart_id}_{self.component_name}'].config = config;
            
            // Simple visualization
            if (data && data.length > 0) {{
                const maxValue = Math.max(...data);
                const minValue = Math.min(...data);
                const range = maxValue - minValue || 1;
                
                let chartHtml = `
                <div style="padding: 10px; width: 100%; height: 100%;">
                    <div style="font-weight: bold; margin-bottom: 10px; color: #2c3e50;">
                        ${{config.title || 'Chart'}}
                    </div>
                    <div style="display: flex; align-items: end; height: {self._height - 80}px; gap: 2px;">
                `;
                
                data.slice(-20).forEach((value, index) => {{
                    const height = ((value - minValue) / range) * ({self._height - 100}) || 10;
                    const color = config.color || '#4ecdc4';
                    const barWidth = 100/Math.min(data.length, 20);
                    
                    if (config.type === 'line') {{
                        chartHtml += `
                        <div style="
                            width: ${{barWidth}}%;
                            height: ${{height}}px;
                            background: linear-gradient(to top, ${{color}}, transparent);
                            border-radius: 2px 2px 0 0;
                            margin-bottom: 2px;
                            position: relative;
                        ">
                            <div style="
                                position: absolute;
                                top: -2px;
                                left: 50%;
                                transform: translateX(-50%);
                                width: 4px;
                                height: 4px;
                                background: ${{color}};
                                border-radius: 50%;
                            "></div>
                        </div>`;
                    }} else {{
                        chartHtml += `
                        <div style="
                            width: ${{barWidth}}%;
                            height: ${{height}}px;
                            background: ${{color}};
                            border-radius: 2px 2px 0 0;
                        "></div>`;
                    }}
                }});
                
                chartHtml += `
                    </div>
                    <div style="margin-top: 10px; font-size: 12px; color: #6c757d;">
                        Latest: ${{data[data.length - 1].toFixed(4)}} | 
                        Max: ${{maxValue.toFixed(4)}} | 
                        Min: ${{minValue.toFixed(4)}}
                    </div>
                </div>`;
                
                chartElement.innerHTML = chartHtml;
            }}
        }};
        </script>
        """
        
        return widgets.HTML(chart_html)
    
    def _on_chart_type_change(self, change) -> None:
        """Handle chart type selection change."""
        # Get the key from the selected tuple
        selected_tuple = change['new']
        if isinstance(selected_tuple, tuple):
            self._chart_type = selected_tuple[0]
        else:
            self._chart_type = selected_tuple
        
        # Update all chart configurations
        for chart_id in ['chart_1', 'chart_2']:
            if chart_id in self._chart_configs:
                self._chart_configs[chart_id]['type'] = self._chart_type
                # Re-render chart with new type
                self.update_chart(
                    chart_id, 
                    self._chart_data.get(chart_id, []),
                    self._chart_configs[chart_id]
                )
    
    def update_chart(self, chart_id: str, data: List[float], 
                    config: Optional[Dict[str, Any]] = None) -> None:
        """Update chart with new data.
        
        Args:
            chart_id: ID of chart to update ('chart_1' or 'chart_2')
            data: List of numeric values to display
            config: Chart configuration (title, color, etc.)
        """
        if not self._initialized:
            self.initialize()
        
        # Validate chart_id
        if chart_id not in ['chart_1', 'chart_2']:
            return
        
        # Only update if chart exists (single column only has chart_1)
        if chart_id == 'chart_2' and self._columns == 1:
            return
        
        # Store data and config
        self._chart_data[chart_id] = data
        
        default_config = {
            'type': self._chart_type,
            'title': f'Chart {chart_id[-1]}',
            'color': '#4ecdc4' if chart_id == 'chart_1' else '#ff6b6b'
        }
        
        if config:
            default_config.update(config)
        
        self._chart_configs[chart_id] = default_config
        
        # Update chart via JavaScript
        js_code = f"""
        <script>
        if (typeof window.updateChart_{chart_id}_{self.component_name} === 'function') {{
            window.updateChart_{chart_id}_{self.component_name}(
                {json.dumps(data)}, 
                {json.dumps(default_config)}
            );
        }}
        </script>
        """
        
        display(HTML(js_code))
    
    def update_charts(self, chart_data: Dict[str, Dict[str, Any]]) -> None:
        """Update multiple charts at once.
        
        Args:
            chart_data: Dictionary with chart_id as key and dict with 'data' and 'config' as value
        """
        for chart_id, chart_info in chart_data.items():
            if chart_id in ['chart_1', 'chart_2']:
                self.update_chart(
                    chart_id,
                    chart_info.get('data', []),
                    chart_info.get('config', {})
                )
    
    def clear_charts(self) -> None:
        """Clear all chart data."""
        self._chart_data = {}
        self._chart_configs = {}
        
        # Reset charts to initial state
        for chart_id in ['chart_1', 'chart_2']:
            if chart_id == 'chart_2' and self._columns == 1:
                continue
            
            js_code = f"""
            <script>
            const chartElement = document.getElementById('{chart_id}_{self.component_name}');
            if (chartElement) {{
                chartElement.innerHTML = `
                <div style="color: #6c757d; font-size: 18px; margin-bottom: 10px;">📊</div>
                <div style="color: #6c757d; font-size: 14px;">Waiting for data...</div>
                <div style="color: #6c757d; font-size: 12px; margin-top: 5px;">Chart {chart_id[-1]}</div>
                `;
            }}
            </script>
            """
            display(HTML(js_code))
    
    def set_chart_config(self, chart_id: str, config: Dict[str, Any]) -> None:
        """Set configuration for a specific chart.
        
        Args:
            chart_id: ID of chart to configure
            config: Configuration dictionary
        """
        if chart_id in ['chart_1', 'chart_2']:
            self._chart_configs[chart_id] = config
    
    @property
    def container(self):
        """Get the main container widget."""
        if not self._initialized:
            self.initialize()
        return self._ui_components['container']


def create_chart_container(title: str = "Metrics Chart",
                          chart_type: str = "line",
                          columns: int = 1,
                          height: int = 400) -> ChartContainer:
    """Create a chart container component.
    
    Args:
        title: Title for the chart container
        chart_type: Type of chart (line, bar, area)
        columns: Number of chart columns (1 or 2)
        height: Height of the chart area in pixels
        
    Returns:
        ChartContainer instance
    """
    return ChartContainer(
        component_name=f"chart_container_{uuid.uuid4().hex[:8]}",
        title=title,
        chart_type=chart_type,
        columns=columns,
        height=height
    )