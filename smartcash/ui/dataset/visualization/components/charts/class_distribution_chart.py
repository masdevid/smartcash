"""
file_path: smartcash/ui/dataset/visualization/components/charts/class_distribution_chart.py

Class distribution chart component for visualization module.
Shows per layer and per class distribution with charts and tables.
"""
from typing import Dict, Any, Optional, List
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from .base_chart import ChartWithTable, create_chart_style_css

class ClassDistributionChart(ChartWithTable):
    """Chart component for displaying class distribution analysis."""
    
    def __init__(self):
        """Initialize class distribution chart."""
        super().__init__(
            chart_id="class_distribution",
            title="📊 Class Distribution Analysis",
            chart_type="class_distribution"
        )
        self.layer_data = {}
        self.class_data = {}
    
    def update_data(self, class_distribution_stats: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        """Update chart with class distribution statistics.
        
        Args:
            class_distribution_stats: Class distribution data from backend
            config: Optional chart configuration
        """
        try:
            if not class_distribution_stats.get('success', False):
                self.show_no_data("No class distribution data available")
                return
            
            # Extract layer and class data
            self.layer_data = class_distribution_stats.get('by_layer', {})
            self.main_banknotes = class_distribution_stats.get('main_banknotes', {})
            self.class_balance = class_distribution_stats.get('class_balance', {})
            
            # Process data for visualization
            self._process_distribution_data()
            
            # Render the chart
            self.render()
            
        except Exception as e:
            self.show_error(f"Error updating class distribution: {e}")
    
    def _process_distribution_data(self):
        """Process raw distribution data for charts and tables."""
        # Process layer data for table
        layer_rows = []
        for layer_name, layer_info in self.layer_data.items():
            layer_rows.append({
                'Layer': layer_name.replace('_', ' ').title(),
                'Total Objects': layer_info.get('total_objects', 0),
                'Active Classes': layer_info.get('active_classes', 0),
                'Avg Objects/Class': layer_info.get('avg_objects_per_class', 0)
            })
        
        self.layer_table_data = pd.DataFrame(layer_rows) if layer_rows else pd.DataFrame()
        
        # Process main banknotes data for table
        banknote_rows = []
        for class_id, banknote_info in self.main_banknotes.items():
            banknote_rows.append({
                'Class ID': class_id,
                'Count': banknote_info.get('count', 0),
                'Percentage': f"{banknote_info.get('percentage', 0):.1f}%",
                'Class Info': banknote_info.get('class_info', {}).get('name', f'Class {class_id}')
            })
        
        self.banknote_table_data = pd.DataFrame(banknote_rows) if banknote_rows else pd.DataFrame()
    
    def render(self):
        """Render the class distribution chart and analysis."""
        with self.chart_container:
            clear_output(wait=True)
            
            try:
                # Add CSS styles
                display(HTML(create_chart_style_css()))
                
                # Create analysis overview
                self._render_overview()
                
                # Create layer distribution visualization
                self._render_layer_distribution()
                
                # Create main banknotes analysis
                self._render_banknotes_analysis()
                
                # Create class balance information
                self._render_class_balance()
                
            except Exception as e:
                self.show_error(f"Error rendering chart: {e}")
        
        # Render table if enabled
        if self.show_table:
            self.render_table()
    
    def _render_overview(self):
        """Render distribution overview."""
        total_objects = sum(layer.get('total_objects', 0) for layer in self.layer_data.values())
        total_layers = len(self.layer_data)
        total_classes = sum(layer.get('active_classes', 0) for layer in self.layer_data.values())
        
        overview_html = f"""
        <div class="chart-container">
            <div class="chart-header">📊 Distribution Overview</div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin: 16px 0;">
                <div style="text-align: center; padding: 16px; background: #e3f2fd; border-radius: 8px;">
                    <div style="font-size: 24px; font-weight: bold; color: #1976d2;">{total_objects:,}</div>
                    <div style="color: #666; font-size: 14px;">Total Objects</div>
                </div>
                <div style="text-align: center; padding: 16px; background: #e8f5e8; border-radius: 8px;">
                    <div style="font-size: 24px; font-weight: bold; color: #388e3c;">{total_classes}</div>
                    <div style="color: #666; font-size: 14px;">Active Classes</div>
                </div>
                <div style="text-align: center; padding: 16px; background: #fff3e0; border-radius: 8px;">
                    <div style="font-size: 24px; font-weight: bold; color: #f57c00;">{total_layers}</div>
                    <div style="color: #666; font-size: 14px;">Data Layers</div>
                </div>
            </div>
        </div>
        """
        display(HTML(overview_html))
    
    def _render_layer_distribution(self):
        """Render layer-wise distribution visualization."""
        if not self.layer_data:
            return
        
        # Create horizontal bar chart using HTML/CSS
        max_objects = max(layer.get('total_objects', 0) for layer in self.layer_data.values()) or 1
        
        bars_html = ""
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
        
        for i, (layer_name, layer_info) in enumerate(self.layer_data.items()):
            objects = layer_info.get('total_objects', 0)
            classes = layer_info.get('active_classes', 0)
            width = (objects / max_objects) * 100
            color = colors[i % len(colors)]
            
            bars_html += f"""
            <div style="margin: 12px 0;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <span style="font-weight: 500; color: #2c3e50;">{layer_name.replace('_', ' ').title()}</span>
                    <span style="color: #666; font-size: 14px;">{objects:,} objects, {classes} classes</span>
                </div>
                <div style="background: #f8f9fa; height: 24px; border-radius: 12px; overflow: hidden;">
                    <div style="background: {color}; height: 100%; width: {width:.1f}%; border-radius: 12px; transition: width 0.3s ease;"></div>
                </div>
            </div>
            """
        
        layer_chart_html = f"""
        <div class="chart-container">
            <div class="chart-header">🏷️ Distribution by Layer</div>
            <div style="padding: 16px;">
                {bars_html}
            </div>
        </div>
        """
        display(HTML(layer_chart_html))
    
    def _render_banknotes_analysis(self):
        """Render main banknotes analysis."""
        if not self.main_banknotes:
            return
        
        # Create pie chart representation using HTML/CSS
        total_banknotes = sum(info.get('count', 0) for info in self.main_banknotes.values())
        
        if total_banknotes == 0:
            return
        
        banknote_items = ""
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24', '#f0932b', '#eb4d4b', '#6c5ce7']
        
        for i, (class_id, info) in enumerate(self.main_banknotes.items()):
            count = info.get('count', 0)
            percentage = info.get('percentage', 0)
            class_name = info.get('class_info', {}).get('name', f'Class {class_id}')
            color = colors[i % len(colors)]
            
            banknote_items += f"""
            <div style="display: flex; align-items: center; margin: 8px 0; padding: 8px; background: #f8f9fa; border-radius: 4px;">
                <div style="width: 16px; height: 16px; background: {color}; border-radius: 50%; margin-right: 12px;"></div>
                <div style="flex: 1;">
                    <div style="font-weight: 500; color: #2c3e50;">{class_name}</div>
                    <div style="font-size: 12px; color: #666;">Class ID: {class_id}</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-weight: bold; color: #2c3e50;">{count:,}</div>
                    <div style="font-size: 12px; color: #666;">{percentage:.1f}%</div>
                </div>
            </div>
            """
        
        banknotes_html = f"""
        <div class="chart-container">
            <div class="chart-header">💰 Main Banknotes Analysis</div>
            <div style="padding: 16px;">
                <div style="text-align: center; margin-bottom: 16px;">
                    <span style="font-size: 18px; font-weight: bold; color: #2c3e50;">Total: {total_banknotes:,} objects</span>
                </div>
                {banknote_items}
            </div>
        </div>
        """
        display(HTML(banknotes_html))
    
    def _render_class_balance(self):
        """Render class balance information."""
        if not self.class_balance:
            return
        
        balanced = self.class_balance.get('balanced', True)
        imbalance_ratio = self.class_balance.get('imbalance_ratio', 1.0)
        recommendation = self.class_balance.get('recommendation', 'Well balanced')
        
        balance_color = '#27ae60' if balanced else '#e74c3c'
        balance_icon = '✅' if balanced else '⚠️'
        balance_text = 'Balanced' if balanced else 'Imbalanced'
        
        balance_html = f"""
        <div class="chart-container">
            <div class="chart-header">⚖️ Class Balance Analysis</div>
            <div style="padding: 16px;">
                <div style="display: flex; align-items: center; margin-bottom: 16px;">
                    <div style="font-size: 24px; margin-right: 12px;">{balance_icon}</div>
                    <div>
                        <div style="font-size: 18px; font-weight: bold; color: {balance_color};">{balance_text}</div>
                        <div style="color: #666; font-size: 14px;">Imbalance Ratio: {imbalance_ratio:.2f}x</div>
                    </div>
                </div>
                <div style="background: #f8f9fa; padding: 12px; border-radius: 4px; border-left: 4px solid {balance_color};">
                    <strong>💡 Recommendation:</strong> {recommendation}
                </div>
            </div>
        </div>
        """
        display(HTML(balance_html))
    
    def render_table(self, data: Optional[pd.DataFrame] = None):
        """Render data tables for layer and banknote distribution."""
        with self.table_container:
            clear_output(wait=True)
            
            try:
                # Display layer distribution table
                if not self.layer_table_data.empty:
                    display(HTML('<h5 style="margin: 16px 0 8px 0; color: #2c3e50;">🏷️ Layer Distribution</h5>'))
                    styled_layer = self.layer_table_data.style.set_table_styles([
                        {'selector': 'th', 'props': [
                            ('background-color', '#3498db'),
                            ('color', 'white'),
                            ('font-weight', 'bold'),
                            ('text-align', 'center'),
                            ('padding', '10px')
                        ]},
                        {'selector': 'td', 'props': [
                            ('text-align', 'center'),
                            ('padding', '8px'),
                            ('border-bottom', '1px solid #dee2e6')
                        ]},
                        {'selector': 'table', 'props': [
                            ('width', '100%'),
                            ('border-collapse', 'collapse'),
                            ('margin-bottom', '20px')
                        ]}
                    ])
                    display(styled_layer)
                
                # Display banknote distribution table
                if not self.banknote_table_data.empty:
                    display(HTML('<h5 style="margin: 16px 0 8px 0; color: #2c3e50;">💰 Main Banknotes</h5>'))
                    styled_banknote = self.banknote_table_data.style.set_table_styles([
                        {'selector': 'th', 'props': [
                            ('background-color', '#e74c3c'),
                            ('color', 'white'),
                            ('font-weight', 'bold'),
                            ('text-align', 'center'),
                            ('padding', '10px')
                        ]},
                        {'selector': 'td', 'props': [
                            ('text-align', 'center'),
                            ('padding', '8px'),
                            ('border-bottom', '1px solid #dee2e6')
                        ]},
                        {'selector': 'table', 'props': [
                            ('width', '100%'),
                            ('border-collapse', 'collapse')
                        ]}
                    ])
                    display(styled_banknote)
                
            except Exception as e:
                display(HTML(f'<div style="color: #dc3545; padding: 10px;">Error displaying tables: {e}</div>'))


def create_class_distribution_chart() -> ClassDistributionChart:
    """Create a class distribution chart component.
    
    Returns:
        Configured class distribution chart ready for use
        
    Usage:
        >>> chart = create_class_distribution_chart()
        >>> display(chart.get_widget())
        >>> # Later update with data:
        >>> chart.update_data(class_distribution_stats)
    """
    return ClassDistributionChart()