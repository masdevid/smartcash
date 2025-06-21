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
    total_labels = train_stats.get('labels', 0) + val_stats.get('labels', 0) + test_stats.get('labels', 0)
    
    # Hitung persentase dengan pengecekan pembagian nol
    train_percentage = (train_stats.get('images', 0) / total_images * 100) if total_images > 0 else 0
    val_percentage = (val_stats.get('images', 0) / total_images * 100) if total_images > 0 else 0
    test_percentage = (test_stats.get('images', 0) / total_images * 100) if total_images > 0 else 0
    
    # Buat card
    train_card = widgets.HTML(f"""
    <div style="border: 1px solid #4285F4; border-radius: 4px; padding: 6px; margin: 2px; background-color: rgba(66, 133, 244, 0.1);">
        <h4 style="margin-top: 0; margin-bottom: 2px; font-size: 13px; color: #4285F4;">Train</h4>
        <p style="font-size: 18px; font-weight: bold; margin: 2px 0;">{train_stats.get('images', 0)}</p>
        <p style="font-size: 11px; margin: 2px 0;">Images: {train_stats.get('images', 0)} | Labels: {train_stats.get('labels', 0)}</p>
        <p style="font-size: 11px; margin: 2px 0;">{train_percentage:.1f}% dari total dataset</p>
    </div>
    """)
    
    val_card = widgets.HTML(f"""
    <div style="border: 1px solid #FBBC05; border-radius: 4px; padding: 6px; margin: 2px; background-color: rgba(251, 188, 5, 0.1);">
        <h4 style="margin-top: 0; margin-bottom: 2px; font-size: 13px; color: #FBBC05;">Validation</h4>
        <p style="font-size: 18px; font-weight: bold; margin: 2px 0;">{val_stats.get('images', 0)}</p>
        <p style="font-size: 11px; margin: 2px 0;">Images: {val_stats.get('images', 0)} | Labels: {val_stats.get('labels', 0)}</p>
        <p style="font-size: 11px; margin: 2px 0;">{val_percentage:.1f}% dari total dataset</p>
    </div>
    """)
    
    test_card = widgets.HTML(f"""
    <div style="border: 1px solid #34A853; border-radius: 4px; padding: 6px; margin: 2px; background-color: rgba(52, 168, 83, 0.1);">
        <h4 style="margin-top: 0; margin-bottom: 2px; font-size: 13px; color: #34A853;">Test</h4>
        <p style="font-size: 18px; font-weight: bold; margin: 2px 0;">{test_stats.get('images', 0)}</p>
        <p style="font-size: 11px; margin: 2px 0;">Images: {test_stats.get('images', 0)} | Labels: {test_stats.get('labels', 0)}</p>
        <p style="font-size: 11px; margin: 2px 0;">{test_percentage:.1f}% dari total dataset</p>
    </div>
    """)
    
    total_card = widgets.HTML(f"""
    <div style="border: 1px solid #333; border-radius: 4px; padding: 6px; margin: 2px; background-color: rgba(0, 0, 0, 0.05);">
        <h4 style="margin-top: 0; margin-bottom: 2px; font-size: 13px;">Total</h4>
        <p style="font-size: 18px; font-weight: bold; margin: 2px 0;">{total_images}</p>
        <p style="font-size: 11px; margin: 2px 0;">Images: {total_images} | Labels: {total_labels}</p>
        <p style="font-size: 11px; margin: 2px 0;">100% dari total dataset</p>
    </div>
    """)
    
    # Buat container
    container = widgets.HBox([train_card, val_card, test_card, total_card], 
                             layout=widgets.Layout(width='100%', justify_content='space-between', gap='2px'))
    
    return container
