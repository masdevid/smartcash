# File: smartcash/ui/utils/responsive_styling.py
# Deskripsi: Utility untuk responsive design dan mobile-friendly layouts

from typing import Dict, Any
import ipywidgets as widgets

def get_responsive_css() -> str:
    """CSS untuk responsive design 📱"""
    return """
    <style>
    .hyperparams-container {
        width: 100%;
        max-width: 1200px;
        margin: 0 auto;
        padding: 10px;
    }
    
    .param-group {
        margin-bottom: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .param-group:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    
    .param-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 15px;
        padding: 15px;
    }
    
    .widget-slider, .widget-dropdown, .widget-checkbox {
        width: 100% !important;
        min-width: 260px;
    }
    
    @media (max-width: 768px) {
        .param-grid {
            grid-template-columns: 1fr;
            gap: 10px;
            padding: 10px;
        }
        
        .hyperparams-container {
            padding: 5px;
        }
        
        .param-group {
            margin-bottom: 15px;
        }
    }
    
    @media (max-width: 480px) {
        .widget-slider, .widget-dropdown, .widget-checkbox {
            min-width: 240px;
        }
    }
    </style>
    """


def create_group_container(title: str, color: str, bg_gradient: str, 
                          children: list) -> widgets.VBox:
    """Buat container group dengan styling konsisten 🎨"""
    
    header = widgets.HTML(
        value=f"<h3 style='margin: 0 0 15px 0; color: {color}; font-weight: bold; font-size: 16px;'>{title}</h3>"
    )
    
    container = widgets.VBox([
        header,
        widgets.VBox(children, layout=widgets.Layout(
            display='flex',
            flex_direction='column',
            gap='10px'
        ))
    ], layout=widgets.Layout(
        border=f'2px solid {color}',
        border_radius='12px',
        padding='20px',
        margin='10px 0',
        width='100%',
        background=bg_gradient
    ))
    
    return container


def apply_mobile_breakpoints() -> widgets.HTML:
    """Apply mobile responsive breakpoints 📱"""
    return widgets.HTML(value=get_responsive_css())