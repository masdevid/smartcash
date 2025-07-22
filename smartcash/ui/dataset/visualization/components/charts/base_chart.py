"""
file_path: smartcash/ui/dataset/visualization/components/charts/base_chart.py

Base chart component for reusable chart implementations.
Each chart type has its dedicated file for reusability and maintenance.
"""
from typing import Dict, Any, Optional, List, Union
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import pandas as pd

class BaseChart:
    """Base class for all chart components."""
    
    def __init__(self, chart_id: str, title: str, chart_type: str = "base"):
        """Initialize base chart.
        
        Args:
            chart_id: Unique identifier for the chart
            title: Chart title
            chart_type: Type of chart (for styling)
        """
        self.chart_id = chart_id
        self.title = title
        self.chart_type = chart_type
        self.data = None
        self.chart_config = {}
        
        # Create UI components
        self._create_widgets()
    
    def _create_widgets(self):
        """Create base widgets for the chart."""
        # Chart container with styling
        self.chart_container = widgets.Output(
            layout=widgets.Layout(
                width='100%',
                min_height='400px',
                border='1px solid #e1e4e8',
                border_radius='8px',
                padding='16px',
                margin='8px 0'
            )
        )
        
        # Chart header
        self.chart_header = widgets.HTML(
            value=f'<h4 style="margin: 0 0 16px 0; color: #2c3e50;">{self.title}</h4>'
        )
        
        # Main widget container
        self.main_widget = widgets.VBox([
            self.chart_header,
            self.chart_container
        ], layout=widgets.Layout(width='100%'))
    
    def update_data(self, data: Any, config: Optional[Dict[str, Any]] = None):
        """Update chart data and configuration.
        
        Args:
            data: Chart data
            config: Optional chart configuration
        """
        self.data = data
        if config:
            self.chart_config.update(config)
        self.render()
    
    def render(self):
        """Render the chart. Override in subclasses."""
        with self.chart_container:
            clear_output(wait=True)
            display(HTML(
                '<div style="text-align: center; padding: 40px; color: #6c757d;">'
                f'📊 {self.chart_type.title()} chart placeholder<br>'
                '<small>Override render() method in subclass</small>'
                '</div>'
            ))
    
    def get_widget(self) -> widgets.Widget:
        """Get the main chart widget."""
        return self.main_widget
    
    def set_title(self, title: str):
        """Update chart title."""
        self.title = title
        self.chart_header.value = f'<h4 style="margin: 0 0 16px 0; color: #2c3e50;">{title}</h4>'
    
    def show_error(self, message: str):
        """Show error message in chart container."""
        with self.chart_container:
            clear_output(wait=True)
            display(HTML(
                '<div style="text-align: center; padding: 40px; color: #dc3545;">'
                f'❌ Error: {message}'
                '</div>'
            ))
    
    def show_loading(self, message: str = "Loading chart..."):
        """Show loading indicator."""
        with self.chart_container:
            clear_output(wait=True)
            display(HTML(
                '<div style="text-align: center; padding: 40px; color: #6c757d;">'
                f'⏳ {message}'
                '</div>'
            ))
    
    def show_no_data(self, message: str = "No data available"):
        """Show no data message."""
        with self.chart_container:
            clear_output(wait=True)
            display(HTML(
                '<div style="text-align: center; padding: 40px; color: #6c757d;">'
                f'📭 {message}'
                '</div>'
            ))


class ChartWithTable(BaseChart):
    """Base class for charts that include data tables."""
    
    def __init__(self, chart_id: str, title: str, chart_type: str = "chart_with_table"):
        """Initialize chart with table component."""
        super().__init__(chart_id, title, chart_type)
        self.show_table = True
        self.table_data = None
        
        # Create table container
        self._create_table_widgets()
    
    def _create_table_widgets(self):
        """Create table-specific widgets."""
        # Table toggle button
        self.table_toggle = widgets.ToggleButton(
            value=True,
            description='Show Table',
            icon='table',
            layout=widgets.Layout(width='120px', margin='0 0 10px 0')
        )
        self.table_toggle.observe(self._on_table_toggle, names='value')
        
        # Table container
        self.table_container = widgets.Output(
            layout=widgets.Layout(
                width='100%',
                max_height='300px',
                overflow_y='auto',
                border='1px solid #e1e4e8',
                border_radius='4px',
                margin='16px 0 0 0'
            )
        )
        
        # Update main widget to include table
        self.main_widget = widgets.VBox([
            self.chart_header,
            self.table_toggle,
            self.chart_container,
            self.table_container
        ], layout=widgets.Layout(width='100%'))
    
    def _on_table_toggle(self, change):
        """Handle table toggle."""
        self.show_table = change['new']
        self.table_toggle.description = 'Show Table' if not self.show_table else 'Hide Table'
        
        if self.show_table:
            self.table_container.layout.display = 'block'
            if self.table_data is not None:
                self.render_table()
        else:
            self.table_container.layout.display = 'none'
    
    def render_table(self, data: Optional[pd.DataFrame] = None):
        """Render data table.
        
        Args:
            data: Optional DataFrame to display
        """
        if data is not None:
            self.table_data = data
        
        if self.table_data is None:
            return
        
        with self.table_container:
            clear_output(wait=True)
            try:
                # Style the dataframe for better display
                styled_df = self.table_data.style.set_table_styles([
                    {'selector': 'th', 'props': [
                        ('background-color', '#f8f9fa'),
                        ('color', '#495057'),
                        ('font-weight', 'bold'),
                        ('text-align', 'center'),
                        ('padding', '8px')
                    ]},
                    {'selector': 'td', 'props': [
                        ('text-align', 'center'),
                        ('padding', '6px'),
                        ('border-bottom', '1px solid #dee2e6')
                    ]},
                    {'selector': 'table', 'props': [
                        ('width', '100%'),
                        ('border-collapse', 'collapse'),
                        ('font-size', '14px')
                    ]}
                ])
                display(styled_df)
            except Exception as e:
                display(HTML(f'<div style="color: #dc3545; padding: 10px;">Error displaying table: {e}</div>'))


def create_chart_style_css():
    """Create CSS styles for charts."""
    return """
    <style>
    .chart-container {
        background: white;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .chart-header {
        font-size: 16px;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 2px solid #ecf0f1;
    }
    .chart-controls {
        display: flex;
        gap: 10px;
        margin-bottom: 16px;
        flex-wrap: wrap;
    }
    .data-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 16px;
        font-size: 14px;
    }
    .data-table th,
    .data-table td {
        padding: 8px 12px;
        text-align: left;
        border-bottom: 1px solid #dee2e6;
    }
    .data-table th {
        background-color: #f8f9fa;
        font-weight: 600;
        color: #495057;
    }
    .data-table tr:hover {
        background-color: #f8f9fa;
    }
    .chart-placeholder {
        text-align: center;
        padding: 60px 20px;
        color: #6c757d;
        background: #f8f9fa;
        border: 2px dashed #dee2e6;
        border-radius: 8px;
        margin: 16px 0;
    }
    .chart-error {
        text-align: center;
        padding: 40px 20px;
        color: #dc3545;
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 4px;
        margin: 16px 0;
    }
    </style>
    """