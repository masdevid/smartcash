"""
Indicator Panel - A shared component for displaying environment status.

Provides a horizontal panel that shows environment status with colored indicators.
"""
from typing import Optional
import ipywidgets as widgets
from IPython.display import display


class IndicatorPanel:
    """A horizontal panel for displaying environment status with colored indicators.
    
    Features:
    - Shows environment status with colored bullet (🟢 Colab, 🔵 Local, 🟡 Unknown)
    - Displays config path
    - Uses space-between layout
    - Responsive design
    """
    
    def __init__(
        self,
        environment: str,
        config_path: str,
    ):
        """Initialize the environment indicator panel.
        
        Args:
            environment: The environment type ('colab', 'local', or other)
            config_path: Path to the config file being used
        """
        self.environment = environment.lower()
        self.config_path = config_path
        
        # Define environment indicators and styles with gradient colors
        self._env_styles = {
            'colab': {
                'emoji': '🟢',
                'name': 'Colab',
                'bg_gradient': 'linear-gradient(135deg, #e6f7e6 0%, #f0f9f0 50%, #e6f7e6 100%)',
                'border_color': '#34c65e',  # Green
                'text_color': '#1a5c1a',  # Dark green
                'badge_gradient': 'linear-gradient(135deg, #e6f7e6 0%, #d4f0d4 100%)',
                'badge_border': '#34c65e40'  # Green with opacity
            },
            'local': {
                'emoji': '🔵',
                'name': 'Local',
                'bg_gradient': 'linear-gradient(135deg, #e6f0ff 0%, #f0f6ff 50%, #e6f0ff 100%)',
                'border_color': '#4d79ff',  # Blue
                'text_color': '#1a2e5c',  # Dark blue
                'badge_gradient': 'linear-gradient(135deg, #e6f0ff 0%, #cce0ff 100%)',
                'badge_border': '#4d79ff40'  # Blue with opacity
            }
        }
        # Default style for unknown environments
        self._default_style = {
            'emoji': '🟡',
            'name': self.environment.capitalize(),
            'bg_gradient': 'linear-gradient(135deg, #fff8e6 0%, #fffcf0 50%, #fff8e6 100%)',
            'border_color': '#e6b800',  # Yellow
            'text_color': '#5c4d1a',  # Dark yellow/brown
            'badge_gradient': 'linear-gradient(135deg, #fff8e6 0%, #fff0cc 100%)',
            'badge_border': '#e6b84040'  # Yellow with opacity
        }
        
        # Initialize the widget
        self._create_widget()
    
    def _get_environment_style(self) -> dict:
        """Get the style dictionary for the current environment."""
        return self._env_styles.get(self.environment, self._default_style)
        
    def _get_environment_indicator(self) -> str:
        """Get the formatted environment indicator with emoji."""
        style = self._get_environment_style()
        return f"{style['emoji']} {style['name']}"
    
    def _create_widget(self) -> None:
        """Create the indicator panel widget."""
        # Get environment style
        style = self._get_environment_style()
        
        # Create environment indicator with gradient styling
        env_indicator = widgets.HTML(
            value=(
                f'<div style="'
                f'font-size: 14px; '
                f'font-weight: 600; '
                f'color: {style["text_color"]}; '
                f'padding: 4px 12px; '
                f'border-radius: 4px; '
                f'background: {style["badge_gradient"]}; '
                f'border: 1px solid {style["badge_border"]}; '
                f'box-shadow: 0 1px 2px rgba(0,0,0,0.05); '
                f'margin: 0; '
                f'display: inline-block;'
                f'">'
                f'{self._get_environment_indicator()}'
                '</div>'
            ),
            layout=widgets.Layout(
                width='auto',
                flex='0 0 auto'
            )
        )
        
        # Create config path indicator with subtle gradient
        config_indicator = widgets.HTML(
            value=(
                f'<div style="'
                f'font-size: 13px; '
                f'font-family: monospace; '
                f'color: {style["text_color"]}cc; '
                f'padding: 4px 12px; '
                f'border-radius: 4px; '
                f'background: rgba(255,255,255,0.5); '
                f'border: 1px solid {style["border_color"]}20; '
                f'box-shadow: inset 0 1px 2px rgba(0,0,0,0.05); '
                f'margin: 0; '
                f'display: inline-block; '
                f'text-align: right;'
                f'">'
                f'Config: {self.config_path}'
                '</div>'
            ),
            layout=widgets.Layout(
                width='auto',
                flex='1 1 auto',
                text_align='right'
            )
        )
        
        # Create the main container using HBox for proper layout
        self.panel = widgets.HBox(
            children=[env_indicator, config_indicator],
            layout=widgets.Layout(
                width='100%',
                justify_content='space-between',
                align_items='center',
                padding='6px 12px',
                margin='2px 0',
                border=f'1px solid {style["border_color"]}40',
                border_radius='6px',
                background=style["bg_gradient"],
                box_shadow='0 1px 3px rgba(0,0,0,0.05)'
            )
        )
    
    def update(
        self,
        environment: Optional[str] = None,
        config_path: Optional[str] = None
    ) -> None:
        """Update the indicator panel properties.
        
        Args:
            environment: New environment value
            config_path: New config path
        """
        if environment is not None:
            self.environment = environment.lower()
        if config_path is not None:
            self.config_path = config_path
            
        # Recreate the widget with updated properties
        self._create_widget()
    
    def show(self) -> widgets.HBox:
        """Display the indicator panel.
        
        Returns:
            The HBox widget containing the indicator panel
        """
        return self.panel
    
    def _ipython_display_(self):
        """IPython display integration."""
        return display(self.show())


def create_indicator_panel(
    environment: str,
    config_path: str,
) -> IndicatorPanel:
    """Create a new environment indicator panel.
    
    Args:
        environment: The environment type ('colab', 'local', or other)
        config_path: Path to the config file being used
        
    Returns:
        An IndicatorPanel instance
    """
    return IndicatorPanel(
        environment=environment,
        config_path=config_path
    )
