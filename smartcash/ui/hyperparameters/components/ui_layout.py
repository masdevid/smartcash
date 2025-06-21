"""
File: smartcash/ui/hyperparameters/components/ui_layout.py
Deskripsi: Layout arrangement untuk hyperparameters dengan pola backbone
"""

from typing import Dict, Any, List, Optional
import ipywidgets as widgets
from smartcash.ui.utils.header_utils import create_header
from smartcash.ui.utils.layout_utils import (
    create_responsive_container,
    create_responsive_two_column
)
from smartcash.common.logger import get_logger

logger = get_logger(__name__)


def create_hyperparameters_layout(components: Dict[str, Any]) -> Dict[str, Any]:
    """Create layout untuk hyperparameters dengan pola backbone
    
    Args:
        components: Dictionary berisi komponen UI yang sudah dibuat
        
    Returns:
        Dictionary berisi layout dan komponen UI
    """
    try:
        # Buat header
        header = create_header(
            title="⚙️ Konfigurasi Hyperparameters",
            description="Atur parameter pelatihan untuk model deteksi mata uang",
            icon="⚙️"
        )
        
        # Buat section untuk setiap grup parameter
        training_section = _create_training_section(components)
        optimizer_section = _create_optimizer_section(components)
        scheduler_section = _create_scheduler_section(components)
        loss_section = _create_loss_section(components)
        early_stopping_section = _create_early_stopping_section(components)
        
        # Gabungkan semua section
        main_content = widgets.VBox([
            training_section,
            widgets.HTML("<hr style='margin: 15px 0; border: 1px dashed #ddd;'>"),
            optimizer_section,
            scheduler_section,
            loss_section,
            early_stopping_section
        ], layout=widgets.Layout(width='100%'))
        
        # Buat container utama
        main_container = widgets.VBox([
            header,
            components.get('status_panel', widgets.HTML('')),
            widgets.HTML("<hr style='margin: 15px 0; border: 1px dashed #ddd;'>"),
            main_content,
            widgets.HTML("<hr style='margin: 15px 0; border: 1px dashed #ddd;'>"),
            components.get('save_reset_container', widgets.HBox())
        ], layout=widgets.Layout(width='100%'))
        
        return {
            'main_container': main_container,
            'ui': main_container,
            'header': header,
            'main_content': main_content,
            **components
        }
        
    except Exception as e:
        logger.error(f"❌ Error creating hyperparameters layout: {e}")
        raise


def _create_training_section(components: Dict[str, Any]) -> widgets.Widget:
    """Buat section untuk parameter training"""
    training_widgets = [
        components.get('epochs'),
        components.get('batch_size'),
        components.get('learning_rate'),
        components.get('image_size')
    ]
    
    return create_responsive_container([
        widgets.HTML("<h4 style='color: #2c3e50; margin: 0 0 10px 0;'>📊 Parameter Training</h4>"),
        *[w for w in training_widgets if w is not None]
    ], padding='10px', style={'border-left': '4px solid #2196f3'})


def _create_optimizer_section(components: Dict[str, Any]) -> widgets.Widget:
    """Buat section untuk optimizer"""
    optimizer_widgets = [
        components.get('optimizer_type'),
        components.get('weight_decay'),
        components.get('momentum')
    ]
    
    return create_responsive_container([
        widgets.HTML("<h4 style='color: #2c3e50; margin: 10px 0;'>⚡ Optimizer</h4>"),
        *[w for w in optimizer_widgets if w is not None]
    ], padding='10px', style={'border-left': '4px solid #9c27b0'})


def _create_scheduler_section(components: Dict[str, Any]) -> widgets.Widget:
    """Buat section untuk scheduler"""
    scheduler_widgets = [
        components.get('scheduler_type'),
        components.get('warmup_epochs'),
        components.get('min_lr')
    ]
    
    return create_responsive_container([
        widgets.HTML("<h4 style='color: #2c3e50; margin: 10px 0;'>📈 Scheduler</h4>"),
        *[w for w in scheduler_widgets if w is not None]
    ], padding='10px', style={'border-left': '4px solid #4caf50'})


def _create_loss_section(components: Dict[str, Any]) -> widgets.Widget:
    """Buat section untuk loss weights"""
    loss_widgets = [
        components.get('box_loss_gain'),
        components.get('cls_loss_gain'),
        components.get('obj_loss_gain')
    ]
    
    return create_responsive_container([
        widgets.HTML("<h4 style='color: #2c3e50; margin: 10px 0;'>🎯 Loss Weights</h4>"),
        *[w for w in loss_widgets if w is not None]
    ], padding='10px', style={'border-left': '4px solid #ff9800'})


def _create_early_stopping_section(components: Dict[str, Any]) -> widgets.Widget:
    """Buat section untuk early stopping"""
    early_stop_widgets = [
        components.get('early_stopping_enabled'),
        components.get('patience'),
        components.get('min_delta')
    ]
    
    return create_responsive_container([
        widgets.HTML("<h4 style='color: #2c3e50; margin: 10px 0;'>⏱️ Early Stopping</h4>"),
        *[w for w in early_stop_widgets if w is not None]
    ], padding='10px', style={'border-left': '4px solid #f44336'})