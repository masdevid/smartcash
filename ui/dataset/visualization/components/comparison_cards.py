"""
File: smartcash/ui/dataset/visualization/components/comparison_cards.py
Deskripsi: Komponen kartu perbandingan antara data preprocessing dan augmentation
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
import numpy as np

def create_comparison_cards(
    original_stats: Dict[str, Dict[str, int]],
    preprocessing_stats: Optional[Dict[str, Dict[str, int]]] = None,
    augmentation_stats: Optional[Dict[str, Dict[str, int]]] = None
) -> widgets.VBox:
    """
    Buat kartu perbandingan antara data original, preprocessing, dan augmentation.
    
    Args:
        original_stats: Dictionary berisi statistik split dataset original
        preprocessing_stats: Dictionary berisi statistik split dataset preprocessing
        augmentation_stats: Dictionary berisi statistik split dataset augmentation
        
    Returns:
        Widget VBox berisi kartu perbandingan
    """
    # Jika preprocessing atau augmentation stats tidak tersedia, gunakan data kosong
    if preprocessing_stats is None:
        preprocessing_stats = {
            'train': {'images': 0, 'labels': 0},
            'val': {'images': 0, 'labels': 0},
            'test': {'images': 0, 'labels': 0}
        }
    
    if augmentation_stats is None:
        augmentation_stats = {
            'train': {'images': 0, 'labels': 0},
            'val': {'images': 0, 'labels': 0},
            'test': {'images': 0, 'labels': 0}
        }
    
    # Dapatkan statistik
    original_train = original_stats.get('train', {'images': 0, 'labels': 0}).get('images', 0)
    original_val = original_stats.get('val', {'images': 0, 'labels': 0}).get('images', 0)
    original_test = original_stats.get('test', {'images': 0, 'labels': 0}).get('images', 0)
    
    preprocessing_train = preprocessing_stats.get('train', {'images': 0, 'labels': 0}).get('images', 0)
    preprocessing_val = preprocessing_stats.get('val', {'images': 0, 'labels': 0}).get('images', 0)
    preprocessing_test = preprocessing_stats.get('test', {'images': 0, 'labels': 0}).get('images', 0)
    
    augmentation_train = augmentation_stats.get('train', {'images': 0, 'labels': 0}).get('images', 0)
    augmentation_val = augmentation_stats.get('val', {'images': 0, 'labels': 0}).get('images', 0)
    augmentation_test = augmentation_stats.get('test', {'images': 0, 'labels': 0}).get('images', 0)
    
    # Hitung total
    original_total = original_train + original_val + original_test
    preprocessing_total = preprocessing_train + preprocessing_val + preprocessing_test
    augmentation_total = augmentation_train + augmentation_val + augmentation_test
    
    # Hitung persentase perubahan dengan pengecekan pembagian nol
    preprocessing_train_pct = (preprocessing_train / original_train * 100 - 100) if original_train > 0 else 0
    preprocessing_val_pct = (preprocessing_val / original_val * 100 - 100) if original_val > 0 else 0
    preprocessing_test_pct = (preprocessing_test / original_test * 100 - 100) if original_test > 0 else 0
    preprocessing_total_pct = (preprocessing_total / original_total * 100 - 100) if original_total > 0 else 0
    
    augmentation_train_pct = (augmentation_train / original_train * 100 - 100) if original_train > 0 else 0
    augmentation_val_pct = (augmentation_val / original_val * 100 - 100) if original_val > 0 else 0
    augmentation_test_pct = (augmentation_test / original_test * 100 - 100) if original_test > 0 else 0
    augmentation_total_pct = (augmentation_total / original_total * 100 - 100) if original_total > 0 else 0
    
    # Buat judul
    title = widgets.HTML(
        value="<h3 style='margin-bottom: 6px; font-size: 16px;'>Perbandingan Jumlah Gambar</h3>"
    )
    
    # Buat kartu perbandingan train
    train_card = widgets.HTML(f"""
    <div style="border: 1px solid #4285F4; border-radius: 4px; padding: 6px; margin: 2px; background-color: rgba(66, 133, 244, 0.05);">
        <h4 style="margin-top: 0; margin-bottom: 2px; font-size: 13px; color: #4285F4;">Train Split</h4>
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
            <div style="flex: 1; text-align: center; padding: 4px; background-color: rgba(66, 133, 244, 0.1); border-radius: 4px; margin-right: 2px;">
                <p style="font-weight: bold; margin: 0; font-size: 11px;">Original</p>
                <p style="font-size: 16px; font-weight: bold; margin: 2px 0;">{original_train}</p>
            </div>
            <div style="flex: 1; text-align: center; padding: 4px; background-color: rgba(251, 188, 5, 0.1); border-radius: 4px; margin-right: 2px;">
                <p style="font-weight: bold; margin: 0; font-size: 11px;">Preprocessing</p>
                <p style="font-size: 16px; font-weight: bold; margin: 2px 0;">{preprocessing_train}</p>
                <p style="font-size: 10px; margin: 0;">{'+' if preprocessing_train > original_train else ''}{preprocessing_train - original_train} ({preprocessing_train_pct:.1f}%)</p>
            </div>
            <div style="flex: 1; text-align: center; padding: 4px; background-color: rgba(52, 168, 83, 0.1); border-radius: 4px;">
                <p style="font-weight: bold; margin: 0; font-size: 11px;">Augmentation</p>
                <p style="font-size: 16px; font-weight: bold; margin: 2px 0;">{augmentation_train}</p>
                <p style="font-size: 10px; margin: 0;">{'+' if augmentation_train > original_train else ''}{augmentation_train - original_train} ({augmentation_train_pct:.1f}%)</p>
            </div>
        </div>
    </div>
    """)
    
    # Buat kartu perbandingan validation
    val_card = widgets.HTML(f"""
    <div style="border: 1px solid #FBBC05; border-radius: 4px; padding: 6px; margin: 2px; background-color: rgba(251, 188, 5, 0.05);">
        <h4 style="margin-top: 0; margin-bottom: 2px; font-size: 13px; color: #FBBC05;">Validation Split</h4>
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
            <div style="flex: 1; text-align: center; padding: 4px; background-color: rgba(66, 133, 244, 0.1); border-radius: 4px; margin-right: 2px;">
                <p style="font-weight: bold; margin: 0; font-size: 11px;">Original</p>
                <p style="font-size: 16px; font-weight: bold; margin: 2px 0;">{original_val}</p>
            </div>
            <div style="flex: 1; text-align: center; padding: 4px; background-color: rgba(251, 188, 5, 0.1); border-radius: 4px; margin-right: 2px;">
                <p style="font-weight: bold; margin: 0; font-size: 11px;">Preprocessing</p>
                <p style="font-size: 16px; font-weight: bold; margin: 2px 0;">{preprocessing_val}</p>
                <p style="font-size: 10px; margin: 0;">{'+' if preprocessing_val > original_val else ''}{preprocessing_val - original_val} ({preprocessing_val_pct:.1f}%)</p>
            </div>
            <div style="flex: 1; text-align: center; padding: 4px; background-color: rgba(52, 168, 83, 0.1); border-radius: 4px;">
                <p style="font-weight: bold; margin: 0; font-size: 11px;">Augmentation</p>
                <p style="font-size: 16px; font-weight: bold; margin: 2px 0;">{augmentation_val}</p>
                <p style="font-size: 10px; margin: 0;">{'+' if augmentation_val > original_val else ''}{augmentation_val - original_val} ({augmentation_val_pct:.1f}%)</p>
            </div>
        </div>
    </div>
    """)
    
    # Buat kartu perbandingan test
    test_card = widgets.HTML(f"""
    <div style="border: 1px solid #34A853; border-radius: 4px; padding: 6px; margin: 2px; background-color: rgba(52, 168, 83, 0.05);">
        <h4 style="margin-top: 0; margin-bottom: 2px; font-size: 13px; color: #34A853;">Test Split</h4>
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
            <div style="flex: 1; text-align: center; padding: 4px; background-color: rgba(66, 133, 244, 0.1); border-radius: 4px; margin-right: 2px;">
                <p style="font-weight: bold; margin: 0; font-size: 11px;">Original</p>
                <p style="font-size: 16px; font-weight: bold; margin: 2px 0;">{original_test}</p>
            </div>
            <div style="flex: 1; text-align: center; padding: 4px; background-color: rgba(251, 188, 5, 0.1); border-radius: 4px; margin-right: 2px;">
                <p style="font-weight: bold; margin: 0; font-size: 11px;">Preprocessing</p>
                <p style="font-size: 16px; font-weight: bold; margin: 2px 0;">{preprocessing_test}</p>
                <p style="font-size: 10px; margin: 0;">{'+' if preprocessing_test > original_test else ''}{preprocessing_test - original_test} ({preprocessing_test_pct:.1f}%)</p>
            </div>
            <div style="flex: 1; text-align: center; padding: 4px; background-color: rgba(52, 168, 83, 0.1); border-radius: 4px;">
                <p style="font-weight: bold; margin: 0; font-size: 11px;">Augmentation</p>
                <p style="font-size: 16px; font-weight: bold; margin: 2px 0;">{augmentation_test}</p>
                <p style="font-size: 10px; margin: 0;">{'+' if augmentation_test > original_test else ''}{augmentation_test - original_test} ({augmentation_test_pct:.1f}%)</p>
            </div>
        </div>
    </div>
    """)
    
    # Buat kartu perbandingan total
    total_card = widgets.HTML(f"""
    <div style="border: 1px solid #673AB7; border-radius: 4px; padding: 6px; margin: 2px; background-color: rgba(103, 58, 183, 0.05);">
        <h4 style="margin-top: 0; margin-bottom: 2px; font-size: 13px; color: #673AB7;">Total</h4>
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
            <div style="flex: 1; text-align: center; padding: 4px; background-color: rgba(66, 133, 244, 0.1); border-radius: 4px; margin-right: 2px;">
                <p style="font-weight: bold; margin: 0; font-size: 11px;">Original</p>
                <p style="font-size: 16px; font-weight: bold; margin: 2px 0;">{original_total}</p>
            </div>
            <div style="flex: 1; text-align: center; padding: 4px; background-color: rgba(251, 188, 5, 0.1); border-radius: 4px; margin-right: 2px;">
                <p style="font-weight: bold; margin: 0; font-size: 11px;">Preprocessing</p>
                <p style="font-size: 16px; font-weight: bold; margin: 2px 0;">{preprocessing_total}</p>
                <p style="font-size: 10px; margin: 0;">{'+' if preprocessing_total > original_total else ''}{preprocessing_total - original_total} ({preprocessing_total_pct:.1f}%)</p>
            </div>
            <div style="flex: 1; text-align: center; padding: 4px; background-color: rgba(52, 168, 83, 0.1); border-radius: 4px;">
                <p style="font-weight: bold; margin: 0; font-size: 11px;">Augmentation</p>
                <p style="font-size: 16px; font-weight: bold; margin: 2px 0;">{augmentation_total}</p>
                <p style="font-size: 10px; margin: 0;">{'+' if augmentation_total > original_total else ''}{augmentation_total - original_total} ({augmentation_total_pct:.1f}%)</p>
            </div>
        </div>
    </div>
    """)
    
    # Buat container untuk semua kartu
    container = widgets.VBox([title, train_card, val_card, test_card, total_card],
                             layout=widgets.Layout(width='100%', gap='2px'))
    
    return container 