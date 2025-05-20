"""
File: smartcash/ui/dataset/visualization/components/dashboard_cards.py
Deskripsi: Komponen UI untuk dashboard cards visualisasi dataset
"""

import ipywidgets as widgets
from typing import Dict, Any, List, Optional

from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.color_utils import get_color_for_status


def create_status_card(
    title: str,
    value: int,
    icon: str,
    status: str = "default",
    description: str = "",
    width: str = "100%"
) -> widgets.VBox:
    """
    Membuat card status dengan nilai dan ikon.
    
    Args:
        title: Judul card
        value: Nilai yang ditampilkan
        icon: Ikon yang ditampilkan
        status: Status card (default, preprocessing, augmentation)
        description: Deskripsi tambahan
        width: Lebar card
        
    Returns:
        Box widget yang berisi card
    """
    # Tentukan warna berdasarkan status
    bg_color, text_color = get_color_for_status(status)
    
    # Buat komponen card
    title_widget = widgets.HTML(
        value=f"<div style='font-size: 14px; font-weight: bold; color: {text_color};'>{title}</div>"
    )
    
    value_widget = widgets.HTML(
        value=f"<div style='font-size: 24px; font-weight: bold; color: {text_color};'>{icon} {value}</div>"
    )
    
    description_widget = widgets.HTML(
        value=f"<div style='font-size: 12px; color: {text_color}; opacity: 0.8;'>{description}</div>"
    )
    
    # Buat container card
    card = widgets.VBox(
        [title_widget, value_widget, description_widget],
        layout=widgets.Layout(
            width=width,
            padding="10px",
            margin="5px",
            border="1px solid #ddd",
            border_radius="5px"
        )
    )
    
    # Set background color dengan style
    card.add_class(f"bg-{status}")
    card.style = f"background-color: {bg_color};"
    
    return card


def create_split_cards(
    split_stats: Dict[str, Dict[str, int]],
    preprocessing_status: Dict[str, bool],
    augmentation_status: Dict[str, bool]
) -> widgets.HBox:
    """
    Membuat cards untuk setiap split dataset.
    
    Args:
        split_stats: Dictionary berisi statistik per split
        preprocessing_status: Status preprocessing per split
        augmentation_status: Status augmentasi per split
        
    Returns:
        HBox widget yang berisi cards
    """
    cards = []
    
    for split, stats in split_stats.items():
        # Tentukan status card
        status = "default"
        if augmentation_status.get(split, False):
            status = "augmentation"
        elif preprocessing_status.get(split, False):
            status = "preprocessing"
        
        # Buat card untuk split
        card = create_status_card(
            title=f"Split {split.capitalize()}",
            value=stats.get("images", 0),
            icon=ICONS.get("image", "🖼️"),
            status=status,
            description=f"Label: {stats.get('labels', 0)} | Objek: {stats.get('objects', 0)}",
            width="200px"
        )
        
        cards.append(card)
    
    # Buat container untuk cards
    cards_container = widgets.HBox(
        cards,
        layout=widgets.Layout(
            display="flex",
            flex_flow="row wrap",
            align_items="stretch",
            width="100%"
        )
    )
    
    return cards_container


def create_card(
    title: str,
    value: int,
    description: str,
    icon: str,
    color: str,
    bg_class: str
) -> widgets.VBox:
    """
    Membuat card dengan nilai dan ikon.
    
    Args:
        title: Judul card
        value: Nilai yang ditampilkan
        description: Deskripsi tambahan
        icon: Ikon yang ditampilkan
        color: Warna teks
        bg_class: Kelas background
        
    Returns:
        VBox widget yang berisi card
    """
    # Buat komponen card
    title_widget = widgets.HTML(
        value=f"<div style='font-size: 14px; font-weight: bold; color: {color};'>{title}</div>"
    )
    
    value_widget = widgets.HTML(
        value=f"<div style='font-size: 24px; font-weight: bold; color: {color};'>{icon} {value}</div>"
    )
    
    description_widget = widgets.HTML(
        value=f"<div style='font-size: 12px; color: {color}; opacity: 0.8;'>{description}</div>"
    )
    
    # Buat container card
    card = widgets.VBox(
        [title_widget, value_widget, description_widget],
        layout=widgets.Layout(
            padding="10px",
            margin="5px",
            border="1px solid #ddd",
            border_radius="5px"
        )
    )
    
    # Set background color dengan style
    card.add_class(bg_class)
    
    return card


def create_preprocessing_cards(preprocessing_stats: Dict[str, int]) -> widgets.VBox:
    """
    Buat card untuk statistik preprocessing.
    
    Args:
        preprocessing_stats: Dictionary berisi statistik preprocessing
        
    Returns:
        Widget VBox berisi card preprocessing
    """
    # Dapatkan statistik
    processed_images = preprocessing_stats.get('processed_images', 0)
    filtered_images = preprocessing_stats.get('filtered_images', 0)
    normalized_images = preprocessing_stats.get('normalized_images', 0)
    
    # Buat card
    card = widgets.VBox([
        widgets.HTML(f"""
        <div style="border: 1px solid #ddd; border-radius: 5px; padding: 10px; margin: 5px;">
            <h4 style="margin-top: 0;">Processed Images</h4>
            <p style="font-size: 24px; font-weight: bold;">{processed_images}</p>
        </div>
        """),
        widgets.HTML(f"""
        <div style="border: 1px solid #ddd; border-radius: 5px; padding: 10px; margin: 5px;">
            <h4 style="margin-top: 0;">Filtered Images</h4>
            <p style="font-size: 24px; font-weight: bold;">{filtered_images}</p>
        </div>
        """),
        widgets.HTML(f"""
        <div style="border: 1px solid #ddd; border-radius: 5px; padding: 10px; margin: 5px;">
            <h4 style="margin-top: 0;">Normalized Images</h4>
            <p style="font-size: 24px; font-weight: bold;">{normalized_images}</p>
        </div>
        """)
    ])
    
    return card


def create_augmentation_cards(augmentation_stats: Dict[str, int]) -> widgets.VBox:
    """
    Buat card untuk statistik augmentasi.
    
    Args:
        augmentation_stats: Dictionary berisi statistik augmentasi
        
    Returns:
        Widget VBox berisi card augmentasi
    """
    # Dapatkan statistik
    augmented_images = augmentation_stats.get('augmented_images', 0)
    augmentations_created = augmentation_stats.get('augmentations_created', 0)
    augmentation_types = augmentation_stats.get('augmentation_types', 0)
    
    # Buat card
    card = widgets.VBox([
        widgets.HTML(f"""
        <div style="border: 1px solid #ddd; border-radius: 5px; padding: 10px; margin: 5px;">
            <h4 style="margin-top: 0;">Augmented Images</h4>
            <p style="font-size: 24px; font-weight: bold;">{augmented_images}</p>
        </div>
        """),
        widgets.HTML(f"""
        <div style="border: 1px solid #ddd; border-radius: 5px; padding: 10px; margin: 5px;">
            <h4 style="margin-top: 0;">Augmentations Created</h4>
            <p style="font-size: 24px; font-weight: bold;">{augmentations_created}</p>
        </div>
        """),
        widgets.HTML(f"""
        <div style="border: 1px solid #ddd; border-radius: 5px; padding: 10px; margin: 5px;">
            <h4 style="margin-top: 0;">Augmentation Types</h4>
            <p style="font-size: 24px; font-weight: bold;">{augmentation_types}</p>
        </div>
        """)
    ])
    
    return card
