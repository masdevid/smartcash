"""
File: smartcash/ui/components/dataset_stats_cards.py
Deskripsi: Komponen shared untuk menampilkan statistik dataset dengan one-liner style
"""

import ipywidgets as widgets
from typing import Dict, Any
from smartcash.ui.utils.constants import ICONS, COLORS

def create_dataset_stats_cards(title: str = "Statistik Dataset", description: str = "Statistik jumlah gambar per split dataset",
                              stats: Dict[str, int] = None, total: int = None, width: str = "100%",
                              icon: str = "stats", with_percentages: bool = True, card_colors: Dict[str, str] = None) -> Dict[str, Any]:
    """Buat komponen statistik dataset dengan one-liner style."""
    stats = stats or {"Train": 0, "Validation": 0, "Test": 0}
    total = total or sum(stats.values())
    card_colors = card_colors or {"Train": COLORS.get("primary", "#3498db"), "Validation": COLORS.get("warning", "#f39c12"),
                                  "Test": COLORS.get("info", "#2980b9"), "Total": COLORS.get("success", "#27ae60")}
    
    display_title = f"{ICONS.get(icon, '')} {title}" if icon and icon in ICONS else title
    header = widgets.HTML(f"<h4 style='margin-top: 5px; margin-bottom: 10px; color: {COLORS.get('dark', '#333')};'>{display_title}</h4>")
    description_widget = widgets.HTML(f"<div style='margin-bottom: 10px; color: {COLORS.get('secondary', '#666')};'>{description}</div>") if description else None
    
    def _create_card(split_name, count):
        percentage = (count / total * 100) if total > 0 else 0
        color = card_colors.get(split_name, COLORS.get("secondary", "#95a5a6"))
        percentage_html = f"<div style='color: {COLORS.get('secondary', '#666')};'>{percentage:.1f}% dari total</div>" if with_percentages and total > 0 else ""
        return widgets.HTML(f"""
        <div style='background-color: {color}20; border-left: 4px solid {color}; padding: 10px; margin-bottom: 10px; border-radius: 4px;'>
            <div style='font-weight: bold; font-size: 1.1em; color: {color};'>{split_name}</div>
            <div style='font-size: 1.5em; margin: 5px 0;'>{count:,}</div>
            {percentage_html}
        </div>""")
    
    cards = {split_name.lower(): _create_card(split_name, count) for split_name, count in stats.items()}
    card_widgets = list(cards.values())
    
    if total > 0:
        total_card = _create_card("Total", total)
        cards['total'] = total_card
        card_widgets.append(total_card)
    
    grid_layout = widgets.GridBox(card_widgets, layout=widgets.Layout(grid_template_columns=f"repeat({min(4, len(card_widgets))}, 1fr)", grid_gap='10px', width=width))
    widgets_list = [header] + ([description_widget] if description_widget else []) + [grid_layout]
    container = widgets.VBox(widgets_list, layout=widgets.Layout(margin='10px 0px', padding='10px', border='1px solid #eee', border_radius='4px', width=width))
    
    return {'container': container, 'cards': cards, 'header': header, 'grid': grid_layout}
