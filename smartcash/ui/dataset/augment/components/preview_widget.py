"""
File: smartcash/ui/dataset/augment/components/preview_widget.py
Description: Live preview widget for augmentation module
"""

import os
from pathlib import Path
import ipywidgets as widgets
from IPython.display import display, Image as IPyImage
from typing import Optional, Dict, Any, Tuple


def create_preview_widget() -> Dict[str, Any]:
    """
    Create a live preview widget that loads an image from a path.
    
    Returns:
        Dictionary containing the preview widget and its container
    """
    # Define the preview image path
    preview_path = Path("data/previews/augmentation_preview.jpg")
    
    # Create the image widget
    if preview_path.exists():
        with open(preview_path, "rb") as f:
            preview_image = widgets.Image(
                value=f.read(),
                format='jpg',
                width=300,
                height=200,
                layout=widgets.Layout(
                    margin='10px auto',
                    border='1px solid #ddd',
                    object_fit='contain',
                    max_width='100%',
                    height='auto'
                )
            )
    else:
        # Create a placeholder with dashed border if image not found
        preview_image = widgets.HTML(
            value='''
            <div style="
                width: 300px;
                height: 200px;
                margin: 10px auto;
                border: 2px dashed #ccc;
                border-radius: 4px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: #666;
                font-family: Arial, sans-serif;
                font-size: 12px;
                text-align: center;
                padding: 10px;
                box-sizing: border-box;
            ">
                <div>
                    <div>Preview image not found</div>
                    <div style="font-size: 10px; margin-top: 5px;">
                        Expected at: data/previews/augmentation_preview.jpg
                    </div>
                </div>
            </div>
            '''
        )
    
    # Create the container
    preview_container = widgets.VBox([
        widgets.HTML(
            '<div style="font-weight: bold; margin: 10px 0 5px 0; text-align: center;">'
            '👁️ Live Preview</div>'
        ),
        preview_image,
        widgets.HTML(
            '<div style="font-size: 0.8em; color: #666; text-align: center; margin-top: 5px;">'
            'Preview updates with augmentation settings</div>'
        )
    ], layout=widgets.Layout(
        border='1px solid #e0e0e0',
        border_radius='8px',
        padding='10px',
        margin='0',
        background='#f9f9f9',
        width='100%',
        align_items='center',
        min_height='300px',
        justify_content='center'
    ))
    
    # Add refresh method to update the preview
    def refresh_preview():
        if preview_path.exists():
            with open(preview_path, "rb") as f:
                preview_image.value = f.read()
    
    return {
        'container': preview_container,
        'image': preview_image,
        'refresh': refresh_preview
    }
