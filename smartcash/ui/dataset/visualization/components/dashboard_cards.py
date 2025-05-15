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


def create_preprocessing_cards(
    preprocessing_stats: Dict[str, Any]
) -> widgets.HBox:
    """
    Membuat cards untuk statistik preprocessing.
    
    Args:
        preprocessing_stats: Dictionary berisi statistik preprocessing
        
    Returns:
        HBox widget yang berisi cards
    """
    cards = []
    
    # Card untuk jumlah gambar yang diproses
    processed_card = create_status_card(
        title="Gambar Diproses",
        value=preprocessing_stats.get("processed_images", 0),
        icon=ICONS.get("process", "⚙️"),
        status="preprocessing",
        description="Total gambar yang telah diproses",
        width="200px"
    )
    cards.append(processed_card)
    
    # Card untuk jumlah gambar yang difilter
    filtered_card = create_status_card(
        title="Gambar Difilter",
        value=preprocessing_stats.get("filtered_images", 0),
        icon=ICONS.get("filter", "🔍"),
        status="preprocessing",
        description="Gambar yang difilter karena kualitas rendah",
        width="200px"
    )
    cards.append(filtered_card)
    
    # Card untuk jumlah gambar yang dinormalisasi
    normalized_card = create_status_card(
        title="Gambar Dinormalisasi",
        value=preprocessing_stats.get("normalized_images", 0),
        icon=ICONS.get("normalize", "📐"),
        status="preprocessing",
        description="Gambar yang telah dinormalisasi",
        width="200px"
    )
    cards.append(normalized_card)
    
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


def create_augmentation_cards(
    augmentation_stats: Dict[str, Any]
) -> widgets.HBox:
    """
    Membuat cards untuk statistik augmentasi.
    
    Args:
        augmentation_stats: Dictionary berisi statistik augmentasi
        
    Returns:
        HBox widget yang berisi cards
    """
    cards = []
    
    # Card untuk jumlah gambar yang diaugmentasi
    augmented_card = create_status_card(
        title="Gambar Diaugmentasi",
        value=augmentation_stats.get("augmented_images", 0),
        icon=ICONS.get("augment", "🔄"),
        status="augmentation",
        description="Total gambar yang telah diaugmentasi",
        width="200px"
    )
    cards.append(augmented_card)
    
    # Card untuk jumlah augmentasi yang dibuat
    generated_card = create_status_card(
        title="Augmentasi Dibuat",
        value=augmentation_stats.get("generated_images", 0),
        icon=ICONS.get("generate", "✨"),
        status="augmentation",
        description="Total gambar augmentasi yang dibuat",
        width="200px"
    )
    cards.append(generated_card)
    
    # Card untuk jumlah tipe augmentasi
    types_card = create_status_card(
        title="Tipe Augmentasi",
        value=augmentation_stats.get("augmentation_types", 0),
        icon=ICONS.get("type", "🔠"),
        status="augmentation",
        description="Jumlah tipe augmentasi yang digunakan",
        width="200px"
    )
    cards.append(types_card)
    
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
