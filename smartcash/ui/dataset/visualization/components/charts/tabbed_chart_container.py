"""
file_path: smartcash/ui/dataset/visualization/components/charts/tabbed_chart_container.py

Tabbed chart container for visualization module.
Provides tabbed interface for different chart types with horizontal table layouts.
"""
from typing import Dict, Any, Optional, List
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from .class_distribution_chart import create_class_distribution_chart
from .base_chart import create_chart_style_css

class TabbedChartContainer:
    """Tabbed container for organizing multiple chart types."""
    
    def __init__(self):
        """Initialize tabbed chart container."""
        self.charts = {}
        self.current_stats = None
        
        # Create UI components
        self._create_widgets()
        self._initialize_charts()
    
    def _create_widgets(self):
        """Create tabbed interface widgets."""
        # Tab configuration
        self.tab_configs = [
            {
                'title': '📊 Class Distribution',
                'key': 'class_distribution',
                'description': 'Per layer and per class distribution analysis'
            },
            {
                'title': '📈 Data Overview',
                'key': 'data_overview', 
                'description': 'Dataset statistics and file distribution'
            },
            {
                'title': '🎨 Processing Status',
                'key': 'processing_status',
                'description': 'Preprocessing and augmentation status'
            }
        ]
        
        # Create tab widget
        self.tab_widget = widgets.Tab()
        self.tab_contents = []
        
        # Create content for each tab
        for config in self.tab_configs:
            tab_content = widgets.Output(
                layout=widgets.Layout(
                    width='100%',
                    min_height='500px',
                    padding='16px'
                )
            )
            self.tab_contents.append(tab_content)
        
        # Set up tab widget
        self.tab_widget.children = self.tab_contents
        for i, config in enumerate(self.tab_configs):
            self.tab_widget.set_title(i, config['title'])
        
        # Add tab change handler
        self.tab_widget.observe(self._on_tab_change, names='selected_index')
        
        # Create main container with header
        self.header = widgets.HTML(
            value="""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 16px; border-radius: 8px 8px 0 0; margin-bottom: 0;">
                <h3 style="margin: 0; font-weight: 600;">📊 Data Analysis Dashboard</h3>
                <p style="margin: 8px 0 0 0; opacity: 0.9; font-size: 14px;">
                    Interactive charts and tables for comprehensive dataset analysis
                </p>
            </div>
            """
        )
        
        self.main_container = widgets.VBox([
            self.header,
            self.tab_widget
        ], layout=widgets.Layout(
            width='100%',
            border='1px solid #e1e4e8',
            border_radius='8px',
            overflow='hidden'
        ))
        
        # Initialize with empty state
        self._show_initial_state()
    
    def _initialize_charts(self):
        """Initialize chart components for each tab."""
        # Class Distribution Chart
        self.charts['class_distribution'] = create_class_distribution_chart()
        
        # Placeholder for other charts (will be implemented)
        self.charts['data_overview'] = None
        self.charts['processing_status'] = None
    
    def _on_tab_change(self, change):
        """Handle tab change events."""
        tab_index = change['new']
        if tab_index < len(self.tab_configs):
            tab_key = self.tab_configs[tab_index]['key']
            self._render_tab_content(tab_key, tab_index)
    
    def _show_initial_state(self):
        """Show initial empty state for all tabs."""
        for i, config in enumerate(self.tab_configs):
            with self.tab_contents[i]:
                clear_output(wait=True)
                display(HTML(
                    '<div style="text-align: center; padding: 60px 20px; color: #6c757d;">'
                    f'📊 {config["description"]}<br><br>'
                    '<small>Run the refresh operation to load chart data</small>'
                    '</div>'
                ))
    
    def _render_tab_content(self, tab_key: str, tab_index: int):
        """Render content for a specific tab.
        
        Args:
            tab_key: Key identifying the tab type
            tab_index: Index of the tab
        """
        with self.tab_contents[tab_index]:
            clear_output(wait=True)
            
            try:
                if tab_key == 'class_distribution' and self.charts['class_distribution']:
                    # Display class distribution chart
                    display(self.charts['class_distribution'].get_widget())
                    
                elif tab_key == 'data_overview':
                    self._render_data_overview()
                    
                elif tab_key == 'processing_status':
                    self._render_processing_status()
                    
                else:
                    # Show placeholder for unimplemented tabs
                    config = self.tab_configs[tab_index]
                    display(HTML(
                        '<div style="text-align: center; padding: 60px 20px; color: #6c757d;">'
                        f'🔧 {config["description"]}<br><br>'
                        '<small>Chart implementation in progress</small>'
                        '</div>'
                    ))
                    
            except Exception as e:
                display(HTML(
                    f'<div style="text-align: center; padding: 40px; color: #dc3545;">'
                    f'❌ Error rendering {tab_key}: {e}'
                    '</div>'
                ))
    
    def _render_data_overview(self):
        """Render data overview tab content."""
        if not self.current_stats:
            display(HTML(
                '<div style="text-align: center; padding: 60px; color: #6c757d;">'
                '📊 Data overview will be displayed here<br>'
                '<small>No data available - run refresh to load statistics</small>'
                '</div>'
            ))
            return
        
        # Extract dataset statistics
        dataset_stats = self.current_stats.get('dataset_stats', {})
        if not dataset_stats.get('success', False):
            display(HTML(
                '<div style="text-align: center; padding: 60px; color: #6c757d;">'
                '📊 No dataset statistics available'
                '</div>'
            ))
            return
        
        overview = dataset_stats.get('overview', {})
        file_types = dataset_stats.get('file_types', {})
        by_split = dataset_stats.get('by_split', {})
        
        # Create overview HTML
        overview_html = f"""
        <div style="padding: 20px;">
            <h4 style="color: #2c3e50; margin-bottom: 20px;">📂 Dataset File Overview</h4>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 16px; margin-bottom: 30px;">
                <div style="background: #e3f2fd; padding: 20px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 28px; font-weight: bold; color: #1976d2;">{overview.get('total_files', 0):,}</div>
                    <div style="color: #666; margin-top: 8px;">Total Files</div>
                </div>
                <div style="background: #e8f5e8; padding: 20px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 28px; font-weight: bold; color: #388e3c;">{file_types.get('raw_images', {}).get('count', 0):,}</div>
                    <div style="color: #666; margin-top: 8px;">Raw Images</div>
                </div>
                <div style="background: #fff3e0; padding: 20px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 28px; font-weight: bold; color: #f57c00;">{file_types.get('preprocessed_npy', {}).get('count', 0):,}</div>
                    <div style="color: #666; margin-top: 8px;">Preprocessed Files</div>
                </div>
                <div style="background: #fce4ec; padding: 20px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 28px; font-weight: bold; color: #c2185b;">{file_types.get('augmented_npy', {}).get('count', 0):,}</div>
                    <div style="color: #666; margin-top: 8px;">Augmented Files</div>
                </div>
            </div>
            
            <h5 style="color: #2c3e50; margin: 30px 0 16px 0;">📋 Split Distribution</h5>
            <div style="overflow-x: auto;">
                <table style="width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <thead>
                        <tr style="background: #f8f9fa;">
                            <th style="padding: 12px; text-align: left; font-weight: 600; color: #495057;">Split</th>
                            <th style="padding: 12px; text-align: center; font-weight: 600; color: #495057;">Raw Files</th>
                            <th style="padding: 12px; text-align: center; font-weight: 600; color: #495057;">Preprocessed</th>
                            <th style="padding: 12px; text-align: center; font-weight: 600; color: #495057;">Augmented</th>
                            <th style="padding: 12px; text-align: center; font-weight: 600; color: #495057;">Total</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        for split_name, split_data in by_split.items():
            raw_count = split_data.get('raw', 0)
            preprocessed_count = split_data.get('preprocessed', 0)  
            augmented_count = split_data.get('augmented', 0)
            total_count = split_data.get('total_files', raw_count)
            
            overview_html += f"""
                        <tr style="border-bottom: 1px solid #dee2e6;">
                            <td style="padding: 12px; font-weight: 500; color: #2c3e50;">{split_name.title()}</td>
                            <td style="padding: 12px; text-align: center;">{raw_count:,}</td>
                            <td style="padding: 12px; text-align: center;">{preprocessed_count:,}</td>
                            <td style="padding: 12px; text-align: center;">{augmented_count:,}</td>
                            <td style="padding: 12px; text-align: center; font-weight: 500;">{total_count:,}</td>
                        </tr>
            """
        
        overview_html += """
                    </tbody>
                </table>
            </div>
        </div>
        """
        
        display(HTML(overview_html))
    
    def _render_processing_status(self):
        """Render processing status tab content."""
        if not self.current_stats:
            display(HTML(
                '<div style="text-align: center; padding: 60px; color: #6c757d;">'
                '🎨 Processing status will be displayed here<br>'
                '<small>No data available - run refresh to load statistics</small>'
                '</div>'
            ))
            return
        
        # Extract augmentation statistics
        aug_stats = self.current_stats.get('augmentation_stats', {})
        dataset_stats = self.current_stats.get('dataset_stats', {})
        
        # Create processing status HTML
        status_html = """
        <div style="padding: 20px;">
            <h4 style="color: #2c3e50; margin-bottom: 20px;">🎨 Data Processing Status</h4>
        """
        
        if aug_stats.get('success', False):
            by_split = aug_stats.get('by_split', {})
            
            status_html += """
            <div style="margin-bottom: 30px;">
                <h5 style="color: #2c3e50; margin-bottom: 16px;">📊 Augmentation Progress</h5>
                <div style="overflow-x: auto;">
                    <table style="width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        <thead>
                            <tr style="background: #f8f9fa;">
                                <th style="padding: 12px; text-align: left; font-weight: 600; color: #495057;">Split</th>
                                <th style="padding: 12px; text-align: center; font-weight: 600; color: #495057;">Augmented Files</th>
                                <th style="padding: 12px; text-align: center; font-weight: 600; color: #495057;">File Size (MB)</th>
                                <th style="padding: 12px; text-align: center; font-weight: 600; color: #495057;">Status</th>
                            </tr>
                        </thead>
                        <tbody>
            """
            
            for split_name, split_aug_data in by_split.items():
                file_count = split_aug_data.get('file_count', 0)
                file_size = split_aug_data.get('total_size_mb', 0)
                status = '✅ Complete' if file_count > 0 else '⏳ Pending'
                status_color = '#27ae60' if file_count > 0 else '#f39c12'
                
                status_html += f"""
                            <tr style="border-bottom: 1px solid #dee2e6;">
                                <td style="padding: 12px; font-weight: 500; color: #2c3e50;">{split_name.title()}</td>
                                <td style="padding: 12px; text-align: center;">{file_count:,}</td>
                                <td style="padding: 12px; text-align: center;">{file_size:.1f}</td>
                                <td style="padding: 12px; text-align: center; color: {status_color}; font-weight: 500;">{status}</td>
                            </tr>
                """
            
            status_html += """
                        </tbody>
                    </table>
                </div>
            </div>
            """
        else:
            status_html += """
            <div style="text-align: center; padding: 40px; background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; margin-bottom: 20px;">
                <div style="color: #856404; font-size: 16px;">⚠️ No augmentation data available</div>
                <div style="color: #856404; font-size: 14px; margin-top: 8px;">Run the augmentation process to see status here</div>
            </div>
            """
        
        status_html += "</div>"
        display(HTML(status_html))
    
    def update_charts(self, comprehensive_stats: Dict[str, Any]):
        """Update all charts with comprehensive statistics.
        
        Args:
            comprehensive_stats: Complete statistics from refresh operation
        """
        try:
            self.current_stats = comprehensive_stats
            
            # Update class distribution chart
            if self.charts['class_distribution']:
                class_dist = comprehensive_stats.get('class_distribution', {})
                self.charts['class_distribution'].update_data(class_dist)
            
            # Re-render current tab if it's not class distribution
            current_tab = self.tab_widget.selected_index
            if current_tab < len(self.tab_configs):
                tab_key = self.tab_configs[current_tab]['key']
                if tab_key != 'class_distribution':
                    self._render_tab_content(tab_key, current_tab)
            
        except Exception as e:
            # Show error in current tab
            current_tab = self.tab_widget.selected_index
            with self.tab_contents[current_tab]:
                clear_output(wait=True)
                display(HTML(
                    f'<div style="text-align: center; padding: 40px; color: #dc3545;">'
                    f'❌ Error updating charts: {e}'
                    '</div>'
                ))
    
    def get_widget(self) -> widgets.Widget:
        """Get the main tabbed container widget."""
        return self.main_container
    
    def get_chart(self, chart_key: str):
        """Get a specific chart by key.
        
        Args:
            chart_key: Key identifying the chart
            
        Returns:
            Chart instance or None if not found
        """
        return self.charts.get(chart_key)


def create_tabbed_chart_container() -> TabbedChartContainer:
    """Create a tabbed chart container component.
    
    Returns:
        Configured tabbed chart container ready for use
        
    Usage:
        >>> container = create_tabbed_chart_container()
        >>> display(container.get_widget())
        >>> # Later update with data:
        >>> container.update_charts(comprehensive_stats)
    """
    return TabbedChartContainer()