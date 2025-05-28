"""  
File: smartcash/ui/training_config/training_strategy/components/training_strategy_layout.py
Deskripsi: Layout arrangement untuk training strategy config cell yang DRY
"""

from typing import Dict, Any
import ipywidgets as widgets
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.header_utils import create_header
from smartcash.ui.components.tab_factory import create_tab_widget


def create_training_strategy_layout(form_components: Dict[str, Any]) -> Dict[str, Any]:
    """Create layout dengan reusable components dan tab organization"""
    
    # One-liner section creators
    create_section = lambda title, *widgets_list: widgets.VBox([widgets.HTML(f"<b>{title}</b>"), *widgets_list], layout=widgets.Layout(margin='10px 0'))
    
    # Organize widgets into logical sections
    main_params_section = create_section(
        "Parameter Utama",
        form_components['enabled_checkbox'],
        form_components['batch_size_slider'], 
        form_components['epochs_slider'],
        form_components['learning_rate_slider']
    )
    
    optimizer_section = create_section(
        "Optimizer",
        form_components['optimizer_dropdown'],
        form_components['weight_decay_slider'],
        form_components['momentum_slider']
    )
    
    scheduler_section = create_section(
        "Scheduler",
        form_components['scheduler_checkbox'],
        form_components['scheduler_dropdown'],
        form_components['warmup_epochs_slider'],
        form_components['min_lr_slider']
    )
    
    stopping_section = create_section(
        "Early Stopping",
        form_components['early_stopping_checkbox'],
        form_components['patience_slider'],
        form_components['min_delta_slider']
    )
    
    checkpoint_section = create_section(
        "Checkpoint",
        form_components['checkpoint_checkbox'],
        form_components['save_best_only_checkbox'],
        form_components['save_freq_slider']
    )
    
    utils_section = create_section(
        "Utilitas Training",
        form_components['experiment_name'],
        form_components['checkpoint_dir'],
        form_components['tensorboard'],
        form_components['log_metrics_every'],
        form_components['visualize_batch_every'],
        form_components['gradient_clipping'],
        form_components['mixed_precision'],
        form_components['layer_mode']
    )
    
    validation_section = create_section(
        "Validasi & Evaluasi",
        form_components['validation_frequency'],
        form_components['iou_threshold'],
        form_components['conf_threshold']
    )
    
    multiscale_section = create_section(
        "Multi-scale Training",
        form_components['multi_scale']
    )
    
    # Create tabs dengan logical grouping
    tab_items = [
        ('Parameter', widgets.VBox([main_params_section, optimizer_section], )),
        ('Training', widgets.VBox([scheduler_section, stopping_section, checkpoint_section])),
        ('Utilitas', utils_section),
        ('Validasi', widgets.VBox([validation_section, multiscale_section]))
    ]
    
    tabs = create_tab_widget(tab_items)
    
    # Create header
    header = create_header(
        title="Konfigurasi Strategi Pelatihan",
        description="Pengaturan strategi pelatihan untuk model deteksi mata uang",
        icon=ICONS.get('training', '🏋️')
    )
    
    # Create main container dengan required components
    main_container = widgets.VBox([
        header,
        form_components['status_panel'],
        tabs,
        form_components['button_container']
    ], layout=widgets.Layout(width='100%', padding='10px'))
    
    # Return components dengan required keys untuk ConfigCellInitializer
    return {
        'main_container': main_container,
        'save_button': form_components['save_button'],
        'reset_button': form_components['reset_button'],
        'status_panel': form_components['status_panel'],
        'tabs': tabs,
        'header': header,
        **form_components  # Include all form components
    }