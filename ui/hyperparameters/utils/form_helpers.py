"""
File: smartcash/ui/hyperparameters/utils/form_helpers.py
Deskripsi: Helper functions untuk form widgets hyperparameters dengan one-liner style
"""

from typing import Dict, Any, List, Tuple
import ipywidgets as widgets


def create_slider_widget(value: float, min_val: float, max_val: float, step: float, 
                        description: str, readout_format: str = '.3f') -> widgets.FloatSlider:
    """Create float slider dengan standard layout dan style"""
    return widgets.FloatSlider(
        value=value, min=min_val, max=max_val, step=step, description=description,
        readout_format=readout_format, style={'description_width': '140px'},
        layout=widgets.Layout(width='100%', margin='2px 0')
    )


def create_int_slider_widget(value: int, min_val: int, max_val: int, 
                           description: str, step: int = 1) -> widgets.IntSlider:
    """Create int slider dengan standard layout dan style"""
    return widgets.IntSlider(
        value=value, min=min_val, max=max_val, step=step, description=description,
        style={'description_width': '140px'}, layout=widgets.Layout(width='100%', margin='2px 0')
    )


def create_dropdown_widget(value: str, options: List[str], description: str) -> widgets.Dropdown:
    """Create dropdown dengan standard layout dan style"""
    return widgets.Dropdown(
        value=value, options=options, description=description,
        style={'description_width': '140px'}, layout=widgets.Layout(width='100%', margin='2px 0')
    )


def create_checkbox_widget(value: bool, description: str) -> widgets.Checkbox:
    """Create checkbox dengan standard layout dan style"""
    return widgets.Checkbox(
        value=value, description=description,
        style={'description_width': 'initial'}, layout=widgets.Layout(width='100%', margin='2px 0')
    )


def create_section_card(title: str, widgets_list: List[widgets.Widget], 
                       border_color: str = '#ddd') -> widgets.VBox:
    """Create section card dengan title dan widgets"""
    return widgets.VBox([
        widgets.HTML(f"<h5 style='margin: 0 0 10px 0; color: #333; border-bottom: 2px solid {border_color}; padding-bottom: 5px;'>{title}</h5>"),
        *widgets_list
    ], layout=widgets.Layout(
        padding='15px', border=f'1px solid {border_color}', 
        border_radius='8px', margin='5px', width='48%', min_width='300px'
    ))


def create_summary_cards_widget() -> widgets.HTML:
    """Create always visible summary cards widget untuk menampilkan config summary"""
    return widgets.HTML(
        value="""
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 8px 0;">
            <div style="background: #f5f5f5; padding: 8px; border-radius: 6px; border-left: 3px solid #ccc;">
                <h6 style="margin: 0 0 4px 0; color: #666; font-size: 12px;">📊 Training</h6>
                <p style="margin: 1px 0; font-size: 11px; color: #888;">Belum dikonfigurasi</p>
            </div>
            <div style="background: #f5f5f5; padding: 8px; border-radius: 6px; border-left: 3px solid #ccc;">
                <h6 style="margin: 0 0 4px 0; color: #666; font-size: 12px;">⚙️ Optimizer</h6>
                <p style="margin: 1px 0; font-size: 11px; color: #888;">Belum dikonfigurasi</p>
            </div>
            <div style="background: #f5f5f5; padding: 8px; border-radius: 6px; border-left: 3px solid #ccc;">
                <h6 style="margin: 0 0 4px 0; color: #666; font-size: 12px;">📈 Scheduler</h6>
                <p style="margin: 1px 0; font-size: 11px; color: #888;">Belum dikonfigurasi</p>
            </div>
            <div style="background: #f5f5f5; padding: 8px; border-radius: 6px; border-left: 3px solid #ccc;">
                <h6 style="margin: 0 0 4px 0; color: #666; font-size: 12px;">🛑 Early Stop</h6>
                <p style="margin: 1px 0; font-size: 11px; color: #888;">Belum dikonfigurasi</p>
            </div>
        </div>
        """,
        layout=widgets.Layout(width='100%', margin='5px 0')
    )


def get_form_widget_mappings() -> List[Tuple[str, str, str, Any]]:
    """Return mapping configuration untuk form widgets - (widget_key, config_section, config_key, default_value)"""
    return [
        # Training parameters
        ('epochs_slider', 'training', 'epochs', 100),
        ('batch_size_slider', 'training', 'batch_size', 16),
        ('learning_rate_slider', 'training', 'learning_rate', 0.01),
        ('image_size_slider', 'training', 'image_size', 640),
        ('mixed_precision_checkbox', 'training', 'mixed_precision', True),
        ('gradient_accumulation_slider', 'training', 'gradient_accumulation', 1),
        ('gradient_clipping_slider', 'training', 'gradient_clipping', 1.0),
        
        # Optimizer parameters
        ('optimizer_dropdown', 'optimizer', 'type', 'SGD'),
        ('weight_decay_slider', 'optimizer', 'weight_decay', 0.0005),
        ('momentum_slider', 'optimizer', 'momentum', 0.937),
        
        # Scheduler parameters
        ('scheduler_dropdown', 'scheduler', 'type', 'cosine'),
        ('warmup_epochs_slider', 'scheduler', 'warmup_epochs', 3),
        
        # Loss parameters
        ('box_loss_gain_slider', 'loss', 'box_loss_gain', 0.05),
        ('cls_loss_gain_slider', 'loss', 'cls_loss_gain', 0.5),
        ('obj_loss_gain_slider', 'loss', 'obj_loss_gain', 1.0),
        
        # Early stopping parameters
        ('early_stopping_checkbox', 'early_stopping', 'enabled', True),
        ('patience_slider', 'early_stopping', 'patience', 15),
        ('min_delta_slider', 'early_stopping', 'min_delta', 0.001),
        
        # Checkpoint parameters
        ('save_best_checkbox', 'checkpoint', 'save_best', True),
        ('checkpoint_metric_dropdown', 'checkpoint', 'metric', 'mAP_0.5')
    ]


def create_responsive_grid_layout(cards: List[widgets.Widget]) -> widgets.HBox:
    """Create responsive grid layout untuk cards dengan flexbox"""
    return widgets.HBox(
        cards,
        layout=widgets.Layout(
            width='100%', display='flex', flex_flow='row wrap',
            justify_content='space-between', align_items='stretch',
            gap='10px', overflow='hidden'
        )
    )