"""
File: smartcash/ui/components/dataset_stats_cards.py
Deskripsi: Komponen shared untuk menampilkan statistik dataset dalam bentuk card
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional, List, Tuple
from smartcash.ui.utils.constants import ICONS, COLORS

def create_dataset_stats_cards(
    title: str = "Statistik Dataset",
    description: str = "Statistik jumlah gambar per split dataset",
    stats: Dict[str, int] = None,
    total: int = None,
    width: str = "100%",
    icon: str = "stats",
    with_percentages: bool = True,
    card_colors: Dict[str, str] = None
) -> Dict[str, Any]:
    """
    Buat komponen statistik dataset dalam bentuk card yang dapat digunakan di berbagai modul.
    
    Args:
        title: Judul komponen
        description: Deskripsi komponen
        stats: Dictionary berisi statistik per split {split_name: count}
        total: Total jumlah data (jika None, akan dihitung dari stats)
        width: Lebar komponen
        icon: Ikon untuk judul
        with_percentages: Tampilkan persentase di setiap card
        card_colors: Dictionary berisi warna untuk setiap card {split_name: color}
        
    Returns:
        Dictionary berisi komponen statistik dataset
    """
    # Default stats jika tidak disediakan
    if stats is None:
        stats = {
            "Train": 0,
            "Validation": 0,
            "Test": 0
        }
    
    # Hitung total jika tidak disediakan
    if total is None:
        total = sum(stats.values())
    
    # Default warna untuk card
    if card_colors is None:
        card_colors = {
            "Train": COLORS.get("primary", "#3498db"),
            "Validation": COLORS.get("warning", "#f39c12"),
            "Test": COLORS.get("info", "#2980b9"),
            "Total": COLORS.get("success", "#27ae60")
        }
    
    # Tambahkan ikon jika tersedia
    display_title = title
    if icon and icon in ICONS:
        display_title = f"{ICONS[icon]} {title}"
    
    # Buat header untuk komponen
    header = widgets.HTML(
        value=f"<h4 style='margin-top: 5px; margin-bottom: 10px; color: {COLORS.get('dark', '#333')};'>{display_title}</h4>"
    )
    
    # Buat deskripsi jika ada
    description_widget = None
    if description:
        description_widget = widgets.HTML(
            value=f"<div style='margin-bottom: 10px; color: {COLORS.get('secondary', '#666')};'>{description}</div>"
        )
    
    # Buat card untuk setiap split
    cards = {}
    card_widgets = []
    
    for split_name, count in stats.items():
        # Hitung persentase
        percentage = 0
        if total > 0:
            percentage = (count / total) * 100
        
        # Tentukan warna card
        color = card_colors.get(split_name, COLORS.get("secondary", "#95a5a6"))
        
        # Buat konten card
        card_content = f"""
        <div style='background-color: {color}20; border-left: 4px solid {color}; padding: 10px; margin-bottom: 10px; border-radius: 4px;'>
            <div style='font-weight: bold; font-size: 1.1em; color: {color};'>{split_name}</div>
            <div style='font-size: 1.5em; margin: 5px 0;'>{count:,}</div>
        """
        
        # Tambahkan persentase jika diperlukan
        if with_percentages and total > 0:
            card_content += f"<div style='color: {COLORS.get('secondary', '#666')};'>{percentage:.1f}% dari total</div>"
        
        card_content += "</div>"
        
        # Buat widget card
        card = widgets.HTML(value=card_content)
        
        # Tambahkan ke dictionary dan list
        cards[split_name.lower()] = card
        card_widgets.append(card)
    
    # Buat card untuk total
    if total > 0:
        total_card_content = f"""
        <div style='background-color: {card_colors.get('Total', COLORS.get('success', '#27ae60'))}20; 
             border-left: 4px solid {card_colors.get('Total', COLORS.get('success', '#27ae60'))}; 
             padding: 10px; margin-bottom: 10px; border-radius: 4px;'>
            <div style='font-weight: bold; font-size: 1.1em; color: {card_colors.get('Total', COLORS.get('success', '#27ae60'))};'>Total</div>
            <div style='font-size: 1.5em; margin: 5px 0;'>{total:,}</div>
        </div>
        """
        
        total_card = widgets.HTML(value=total_card_content)
        cards['total'] = total_card
        card_widgets.append(total_card)
    
    # Buat container untuk card dalam grid layout
    grid_layout = widgets.GridBox(
        card_widgets,
        layout=widgets.Layout(
            grid_template_columns=f"repeat({min(4, len(card_widgets))}, 1fr)",
            grid_gap='10px',
            width=width
        )
    )
    
    # Buat container untuk komponen
    widgets_list = [header]
    if description_widget:
        widgets_list.append(description_widget)
    widgets_list.append(grid_layout)
    
    container = widgets.VBox(
        widgets_list,
        layout=widgets.Layout(
            margin='10px 0px',
            padding='10px',
            border='1px solid #eee',
            border_radius='4px',
            width=width
        )
    )
    
    return {
        'container': container,
        'cards': cards,
        'header': header,
        'grid': grid_layout
    }
