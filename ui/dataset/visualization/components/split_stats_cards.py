"""
File: smartcash/ui/dataset/visualization/components/split_stats_cards.py
Deskripsi: Komponen card untuk statistik split dataset
"""

from typing import Dict, Any
import ipywidgets as widgets

def create_split_stats_cards(split_stats: Dict[str, Dict[str, int]]) -> widgets.HBox:
    """
    Buat card untuk statistik split dataset.
    
    Args:
        split_stats: Dictionary berisi statistik split dataset
        
    Returns:
        Widget HBox berisi card split dataset
    """
    # Dapatkan statistik
    train_stats = split_stats.get('train', {'images': 0, 'labels': 0})
    val_stats = split_stats.get('val', {'images': 0, 'labels': 0})
    test_stats = split_stats.get('test', {'images': 0, 'labels': 0})
    
    # Hitung total
    total_images = train_stats.get('images', 0) + val_stats.get('images', 0) + test_stats.get('images', 0)
    
    # Buat card
    train_card = widgets.HTML(f"""
    <div style="border: 1px solid #4285F4; border-radius: 5px; padding: 10px; margin: 5px; background-color: rgba(66, 133, 244, 0.1);">
        <h4 style="margin-top: 0; color: #4285F4;">Train</h4>
        <p style="font-size: 24px; font-weight: bold;">{train_stats.get('images', 0)}</p>
        <p>Images: {train_stats.get('images', 0)} | Labels: {train_stats.get('labels', 0)}</p>
        <p>{train_stats.get('images', 0) / total_images * 100:.1f}% dari total dataset</p>
    </div>
    """)
    
    val_card = widgets.HTML(f"""
    <div style="border: 1px solid #FBBC05; border-radius: 5px; padding: 10px; margin: 5px; background-color: rgba(251, 188, 5, 0.1);">
        <h4 style="margin-top: 0; color: #FBBC05;">Validation</h4>
        <p style="font-size: 24px; font-weight: bold;">{val_stats.get('images', 0)}</p>
        <p>Images: {val_stats.get('images', 0)} | Labels: {val_stats.get('labels', 0)}</p>
        <p>{val_stats.get('images', 0) / total_images * 100:.1f}% dari total dataset</p>
    </div>
    """)
    
    test_card = widgets.HTML(f"""
    <div style="border: 1px solid #34A853; border-radius: 5px; padding: 10px; margin: 5px; background-color: rgba(52, 168, 83, 0.1);">
        <h4 style="margin-top: 0; color: #34A853;">Test</h4>
        <p style="font-size: 24px; font-weight: bold;">{test_stats.get('images', 0)}</p>
        <p>Images: {test_stats.get('images', 0)} | Labels: {test_stats.get('labels', 0)}</p>
        <p>{test_stats.get('images', 0) / total_images * 100:.1f}% dari total dataset</p>
    </div>
    """)
    
    total_card = widgets.HTML(f"""
    <div style="border: 1px solid #333; border-radius: 5px; padding: 10px; margin: 5px; background-color: rgba(0, 0, 0, 0.05);">
        <h4 style="margin-top: 0;">Total</h4>
        <p style="font-size: 24px; font-weight: bold;">{total_images}</p>
        <p>Images: {total_images} | Labels: {train_stats.get('labels', 0) + val_stats.get('labels', 0) + test_stats.get('labels', 0)}</p>
        <p>100% dari total dataset</p>
    </div>
    """)
    
    # Buat container
    container = widgets.HBox([train_card, val_card, test_card, total_card], 
                             layout=widgets.Layout(width='100%', justify_content='space-between'))
    
    return container
