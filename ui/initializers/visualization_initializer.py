"""
File: smartcash/ui/initializers/visualization_initializer.py
Deskripsi: VisualizationInitializer untuk cell visualisasi data tanpa config management
"""

from typing import Dict, Any, Optional, Callable, Union, List
import ipywidgets as widgets
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from smartcash.ui.utils.fallback_utils import create_error_ui, try_operation_safe

class VisualizationInitializer:
    """Initializer khusus untuk visualisasi data dengan matplotlib/plotly integration"""
    
    def __init__(self, module_name: str, title: Optional[str] = None, 
                 figure_size: tuple = (10, 6), style: str = 'seaborn-v0_8'):
        self.module_name = module_name
        self.title = title or f"{module_name.replace('_', ' ').title()} Visualization"
        self.figure_size = figure_size
        self.style = style
        self.plots = {}
        self.callbacks = []
        
        # Setup matplotlib style
        try:
            plt.style.use(style)
        except Exception:
            plt.style.use('default')
    
    def initialize(self, plot_fn: Callable = None, data: Any = None, 
                  interactive: bool = False, **kwargs) -> Any:
        """Initialize visualization cell dengan plot function"""
        try:
            # Create header
            header = self._create_viz_header()
            
            # Create plot output
            plot_output = widgets.Output()
            
            # Create controls jika interactive
            controls = self._create_controls(**kwargs) if interactive else None
            
            # Execute plot function
            if plot_fn and callable(plot_fn):
                with plot_output:
                    try_operation_safe(
                        lambda: self._execute_plot(plot_fn, data, **kwargs),
                        on_error=lambda e: display(HTML(f"<p style='color:red'>Plot error: {str(e)}</p>"))
                    )
            
            # Combine components
            components = [header, plot_output]
            if controls:
                components.insert(1, controls)
            
            container = widgets.VBox(components, layout=widgets.Layout(width='100%'))
            
            # Result metadata
            result = {
                'ui': container,
                'plot_output': plot_output,
                'header': header,
                'controls': controls,
                'module_name': self.module_name,
                'plots': self.plots,
                'visualization_type': 'matplotlib' if not interactive else 'interactive'
            }
            
            # Trigger callbacks
            [try_operation_safe(lambda cb=cb: cb(result)) for cb in self.callbacks]
            
            return container
            
        except Exception as e:
            return create_error_ui(f"Visualization error: {str(e)}", self.module_name)
    
    def _create_viz_header(self) -> widgets.HTML:
        """Create visualization header dengan styling"""
        return widgets.HTML(f"""
        <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
            <h3 style="margin: 0; font-weight: 600;">📊 {self.title}</h3>
        </div>
        """)
    
    def _create_controls(self, **kwargs) -> widgets.HBox:
        """Create interactive controls untuk plot"""
        controls = []
        
        # Common controls
        if 'plot_type' in kwargs:
            plot_types = kwargs.get('plot_types', ['line', 'bar', 'scatter', 'hist'])
            plot_selector = widgets.Dropdown(
                options=plot_types,
                value=kwargs.get('plot_type', plot_types[0]),
                description='Plot Type:'
            )
            controls.append(plot_selector)
        
        if 'columns' in kwargs:
            columns = kwargs.get('columns', [])
            if columns:
                column_selector = widgets.SelectMultiple(
                    options=columns,
                    value=columns[:2] if len(columns) >= 2 else columns,
                    description='Columns:'
                )
                controls.append(column_selector)
        
        # Refresh button
        refresh_btn = widgets.Button(
            description='🔄 Refresh',
            button_style='primary',
            layout=widgets.Layout(width='auto')
        )
        controls.append(refresh_btn)
        
        return widgets.HBox(controls, layout=widgets.Layout(margin='0 0 10px 0')) if controls else None
    
    def _execute_plot(self, plot_fn: Callable, data: Any = None, **kwargs) -> None:
        """Execute plot function dengan error handling"""
        # Setup figure
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Execute plot function
        if data is not None:
            result = plot_fn(data, ax=ax, **kwargs)
        else:
            result = plot_fn(ax=ax, **kwargs)
        
        # Store plot reference
        plot_id = f"plot_{len(self.plots)}"
        self.plots[plot_id] = {'figure': fig, 'axes': ax, 'result': result}
        
        # Display plot
        plt.tight_layout()
        plt.show()
    
    def add_plot(self, plot_name: str, plot_fn: Callable, data: Any = None, **kwargs) -> None:
        """Add plot ke collection dengan one-liner"""
        self.plots[plot_name] = {'function': plot_fn, 'data': data, 'kwargs': kwargs}
    
    def clear_plots(self) -> None:
        """Clear all plots dengan one-liner"""
        plt.close('all'), self.plots.clear()
    
    # Callback management
    add_callback = lambda self, cb: self.callbacks.append(cb) if cb not in self.callbacks else None
    remove_callback = lambda self, cb: self.callbacks.remove(cb) if cb in self.callbacks else None


# Factory functions untuk common visualization types
def create_data_viz_cell(module_name: str, data: pd.DataFrame, plot_type: str = 'auto',
                        columns: List[str] = None, title: str = None, **kwargs) -> Any:
    """Factory untuk data visualization cell"""
    def plot_fn(df, ax=None, **plot_kwargs):
        if plot_type == 'auto':
            # Auto-detect plot type berdasarkan data
            if len(df.select_dtypes(include=['number']).columns) >= 2:
                return df.plot(kind='scatter', x=df.columns[0], y=df.columns[1], ax=ax)
            else:
                return df.plot(kind='hist', ax=ax)
        else:
            return df.plot(kind=plot_type, ax=ax, **plot_kwargs)
    
    viz = VisualizationInitializer(module_name, title)
    return viz.initialize(plot_fn, data, columns=columns or list(data.columns), **kwargs)

def create_custom_viz_cell(module_name: str, plot_function: Callable,
                          data: Any = None, title: str = None, interactive: bool = False, **kwargs) -> Any:
    """Factory untuk custom visualization dengan plot function"""
    viz = VisualizationInitializer(module_name, title)
    return viz.initialize(plot_function, data, interactive=interactive, **kwargs)

def create_matplotlib_cell(module_name: str, figure_fn: Callable,
                          figure_size: tuple = (10, 6), title: str = None, **kwargs) -> Any:
    """Factory untuk matplotlib figure cell"""
    viz = VisualizationInitializer(module_name, title, figure_size)
    return viz.initialize(figure_fn, **kwargs)

def create_seaborn_cell(module_name: str, data: pd.DataFrame, plot_type: str,
                       x: str = None, y: str = None, title: str = None, **kwargs) -> Any:
    """Factory untuk seaborn plots"""
    def seaborn_plot(df, ax=None, **plot_kwargs):
        plot_func = getattr(sns, plot_type, sns.scatterplot)
        return plot_func(data=df, x=x, y=y, ax=ax, **plot_kwargs)
    
    viz = VisualizationInitializer(module_name, title)
    return viz.initialize(seaborn_plot, data, **kwargs)

# One-liner utilities untuk quick plots
create_line_plot = lambda module, data, x, y, **kw: create_seaborn_cell(module, data, 'lineplot', x, y, **kw)
create_scatter_plot = lambda module, data, x, y, **kw: create_seaborn_cell(module, data, 'scatterplot', x, y, **kw)
create_hist_plot = lambda module, data, column, **kw: create_seaborn_cell(module, data, 'histplot', x=column, **kw)
create_box_plot = lambda module, data, x, y, **kw: create_seaborn_cell(module, data, 'boxplot', x, y, **kw)

def create_distribution_viz(module_name: str, data: pd.DataFrame, columns: List[str] = None, 
                           title: str = None) -> Any:
    """Create distribution visualization untuk multiple columns"""
    def dist_plot(df, ax=None):
        cols = columns or df.select_dtypes(include=['number']).columns[:4]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        for i, col in enumerate(cols[:4]):
            if i < len(axes):
                df[col].hist(ax=axes[i], bins=20, alpha=0.7)
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_ylabel('Frequency')
        
        plt.tight_layout()
        return fig
    
    viz = VisualizationInitializer(module_name, title or f"{module_name} Distributions")
    return viz.initialize(dist_plot, data)

def create_correlation_viz(module_name: str, data: pd.DataFrame, title: str = None) -> Any:
    """Create correlation heatmap visualization"""
    def corr_plot(df, ax=None):
        corr_matrix = df.select_dtypes(include=['number']).corr()
        return sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
    
    viz = VisualizationInitializer(module_name, title or f"{module_name} Correlations")
    return viz.initialize(corr_plot, data)