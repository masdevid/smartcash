"""
File: smartcash/ui/dataset/split_config_visualization.py
Deskripsi: Visualisasi dataset untuk komponen konfigurasi split
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from IPython.display import display, HTML, clear_output
from pathlib import Path

# Coba import utility visualization jika tersedia
try:
    from smartcash.ui.utils.visualization_utils import (
        create_class_distribution_plot,
        create_metrics_dashboard
    )
    has_viz_utils = True
except ImportError:
    has_viz_utils = False

def get_dataset_stats(
    config: Dict[str, Any], 
    env=None, 
    logger=None
) -> Dict[str, Any]:
    """
    Mendapatkan statistik dataset dari DatasetManager atau direktori langsung.
    
    Args:
        config: Konfigurasi dengan informasi dataset
        env: Environment manager untuk path
        logger: Logger untuk output
        
    Returns:
        Dictionary berisi statistik dataset
    """
    try:
        # Coba gunakan DatasetManager untuk statistik
        try:
            from smartcash.dataset.manager import DatasetManager
            dataset_manager = DatasetManager(config)
            stats = dataset_manager.get_split_statistics()
            if logger:
                logger.info(f"✅ Statistik dataset berhasil dimuat dari DatasetManager")
            return stats
        except (ImportError, AttributeError) as e:
            if logger:
                logger.debug(f"⚠️ Tidak dapat menggunakan DatasetManager: {str(e)}")
            
            # Fallback ke pembacaan direktori langsung
            data_dir = config.get('data', {}).get('dir', 'data')
            if env and hasattr(env, 'is_drive_mounted') and env.is_drive_mounted and hasattr(env, 'drive_path'):
                data_dir = env.drive_path / 'data'
            
            stats = {}
            for split in ['train', 'valid', 'test']:
                split_dir = Path(data_dir) / split
                if split_dir.exists():
                    images_dir = split_dir / 'images'
                    labels_dir = split_dir / 'labels'
                    
                    img_count = len(list(images_dir.glob('*.*'))) if images_dir.exists() else 0
                    label_count = len(list(labels_dir.glob('*.txt'))) if labels_dir.exists() else 0
                    
                    stats[split] = {
                        'images': img_count,
                        'labels': label_count,
                        'status': 'valid' if img_count > 0 and label_count > 0 else 'empty'
                    }
                else:
                    stats[split] = {'images': 0, 'labels': 0, 'status': 'missing'}
            
            if logger:
                logger.info(f"✅ Statistik dataset berhasil dimuat dari direktori {data_dir}")
            return stats
            
    except Exception as e:
        if logger:
            logger.error(f"❌ Error saat mendapatkan statistik dataset: {str(e)}")
        return {}

def get_class_distribution(
    config: Dict[str, Any], 
    logger=None
) -> Dict[str, Dict[str, int]]:
    """
    Mendapatkan distribusi kelas dari dataset.
    
    Args:
        config: Konfigurasi dengan informasi dataset
        logger: Logger untuk output
        
    Returns:
        Dictionary berisi distribusi kelas per split
    """
    try:
        # Coba gunakan DatasetManager untuk statistik kelas
        try:
            from smartcash.dataset.manager import DatasetManager
            dataset_manager = DatasetManager(config)
            # Mendapatkan distribusi kelas dari manager
            class_stats = {}
            for split in ['train', 'valid', 'test']:
                try:
                    split_stats = dataset_manager.explore_class_distribution(split)
                    class_stats[split] = split_stats
                except Exception:
                    class_stats[split] = {}
            
            if logger:
                logger.info(f"✅ Distribusi kelas berhasil dimuat dari DatasetManager")
            return class_stats
        except (ImportError, AttributeError):
            # Fallback ke data simulasi
            class_stats = {
                'train': {
                    'Rp1000': 350, 'Rp2000': 320, 'Rp5000': 380, 
                    'Rp10000': 390, 'Rp20000': 360, 'Rp50000': 340, 'Rp100000': 370
                },
                'valid': {
                    'Rp1000': 50, 'Rp2000': 45, 'Rp5000': 55, 
                    'Rp10000': 60, 'Rp20000': 55, 'Rp50000': 48, 'Rp100000': 52
                },
                'test': {
                    'Rp1000': 50, 'Rp2000': 45, 'Rp5000': 55, 
                    'Rp10000': 55, 'Rp20000': 50, 'Rp50000': 47, 'Rp100000': 53
                }
            }
            
            if logger:
                logger.warning(f"⚠️ Menggunakan data simulasi untuk distribusi kelas")
            return class_stats
            
    except Exception as e:
        if logger:
            logger.error(f"❌ Error saat mendapatkan distribusi kelas: {str(e)}")
        return {}

def show_class_distribution_visualization(
    output_box,
    class_stats: Dict[str, Dict[str, int]],
    colors: Dict[str, str]
) -> None:
    """
    Tampilkan visualisasi distribusi kelas di output box.
    
    Args:
        output_box: Output widget untuk menampilkan visualisasi
        class_stats: Statistik distribusi kelas per split
        colors: Dictionary berisi palet warna
    """
    from smartcash.ui.utils.constants import ICONS
    
    with output_box:
        clear_output(wait=True)
        
        # Cek apakah ada data untuk ditampilkan
        if not class_stats or not all(len(stats) > 0 for split, stats in class_stats.items()):
            display(HTML(f"""
                <div style="padding:10px; background-color:{colors['alert_warning_bg']}; 
                          border-left:4px solid {colors['alert_warning_text']}; 
                          color:{colors['alert_warning_text']}; border-radius:4px;">
                    <p style="margin:5px 0">{ICONS['warning']} Data distribusi kelas tidak tersedia. Pastikan dataset sudah didownload.</p>
                </div>
            """))
            return
        
        # Buat DataFrame untuk visualisasi
        all_classes = set()
        for split_stats in class_stats.values():
            all_classes.update(split_stats.keys())
            
        df_data = []
        for cls in sorted(all_classes):
            row = {'Class': cls}
            for split in ['train', 'valid', 'test']:
                row[split.capitalize()] = class_stats[split].get(cls, 0)
            df_data.append(row)
            
        df = pd.DataFrame(df_data)
        
        # Gunakan visualization_utils jika tersedia
        if has_viz_utils:
            # Plot menggunakan utility
            fig = create_class_distribution_plot(
                {cls: df[['Train', 'Valid', 'Test']].loc[i].to_dict() 
                 for i, cls in enumerate(df['Class'])},
                title='Distribusi Kelas per Split',
                figsize=(10, 6),
                sort_by='name'
            )
            plt.figure(fig)
            plt.show()
        else:
            # Plot visualization manual
            plt.figure(figsize=(10, 6))
            ax = plt.subplot(111)
            
            # Set width of bar
            barWidth = 0.25
            
            # Set positions of bars on X axis
            r1 = np.arange(len(df))
            r2 = [x + barWidth for x in r1]
            r3 = [x + barWidth for x in r2]
            
            # Create bars
            ax.bar(r1, df['Train'], width=barWidth, label='Train', color=colors['primary'])
            ax.bar(r2, df['Valid'], width=barWidth, label='Valid', color=colors['success'])
            ax.bar(r3, df['Test'], width=barWidth, label='Test', color=colors['warning'])
            
            # Add labels and title
            plt.xlabel('Kelas')
            plt.ylabel('Jumlah Sampel')
            plt.title('Distribusi Kelas per Split')
            plt.xticks([r + barWidth for r in range(len(df))], df['Class'])
            plt.legend()
            
            plt.tight_layout()
            plt.show()
        
        # Tampilkan tabel distribusi
        display(HTML(f"<h3>{ICONS['chart']} Tabel Distribusi Kelas per Split</h3>"))
        display(df.style.background_gradient(cmap='Blues', subset=['Train', 'Valid', 'Test']))
        
        # Tambahkan informasi save/load
        display(HTML(f"""
        <div style="margin-top:20px; padding:10px; background-color:{colors['alert_info_bg']}; 
                   color:{colors['alert_info_text']}; border-radius:4px;">
            <p style="margin:5px 0">{ICONS['save']} Gunakan tombol <strong>Simpan Konfigurasi</strong> untuk menyimpan pengaturan persentase split ke <code>configs/dataset_config.yaml</code></p>
        </div>
        """))

def update_stats_cards(
    html_component,
    stats: Dict[str, Dict[str, Any]],
    colors: Dict[str, str]
) -> None:
    """
    Update komponen HTML dengan statistik distribusi dataset dalam bentuk cards.
    
    Args:
        html_component: Komponen HTML untuk diupdate
        stats: Statistik dataset per split
        colors: Dictionary berisi palet warna
    """
    from smartcash.ui.utils.constants import ICONS
    
    # Hitung total dan persentase
    total_images = sum(split_data.get('images', 0) for split_data in stats.values())
    
    if total_images == 0:
        # Tidak ada data atau error
        html_component.value = f"""
        <div style="padding:10px; background-color:{colors['alert_warning_bg']}; 
                  border-left:4px solid {colors['alert_warning_text']}; 
                  color:{colors['alert_warning_text']}; border-radius:4px; margin:5px 0">
            <p style="margin:5px 0">{ICONS['warning']} Dataset tidak ditemukan atau kosong. Silakan download dataset terlebih dahulu.</p>
        </div>
        """
        return
    
    # Create HTML untuk cards distributions
    html = f"""
    <h3 style="margin-top:0; color:{colors['dark']}">{ICONS['dataset']} Distribusi Dataset Saat Ini</h3>
    <div style="display:flex; flex-wrap:wrap; gap:15px; margin-bottom:15px">
    """
    
    for split in ['train', 'valid', 'test']:
        split_data = stats.get(split, {})
        images = split_data.get('images', 0)
        labels = split_data.get('labels', 0)
        status = split_data.get('status', 'valid')
        percentage = (images / total_images * 100) if total_images > 0 else 0
        
        # Pick color based on status
        color = colors['primary'] if status == 'valid' else colors['danger']
        
        html += f"""
        <div style="flex:1; min-width:150px; border:1px solid {color}; border-radius:5px; padding:10px; background-color:{colors['light']}">
            <h4 style="margin-top:0; color:{color}">{split.capitalize()}</h4>
            <p style="margin:5px 0; font-weight:bold; font-size:1.2em; color:{colors['dark']}">{percentage:.1f}%</p>
            <p style="margin:5px 0; color:{colors['dark']}"><strong>Images:</strong> {images}</p>
            <p style="margin:5px 0; color:{colors['dark']}"><strong>Labels:</strong> {labels}</p>
        </div>
        """
    
    html += "</div>"
    
    # Update HTML component
    html_component.value = html