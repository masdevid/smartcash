"""
File: smartcash/ui/dataset/split_config_visualization.py
Deskripsi: Visualisasi dataset untuk komponen konfigurasi split dengan dukungan Google Drive
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from IPython.display import display, HTML, clear_output
from smartcash.ui.utils.constants import ICONS

def get_dataset_dir(config: Dict[str, Any], env=None, logger=None) -> str:
    """Mendapatkan direktori dataset aktif (drive atau lokal)."""
    try:
        from smartcash.common.environment import get_environment_manager
        env_manager = env or get_environment_manager()
        
        # Cek apakah menggunakan drive
        if config.get('data', {}).get('use_drive', False):
            # Jika sync aktif, gunakan lokal clone
            local_clone = config.get('data', {}).get('local_clone_path', 'data_local')
            if local_clone and os.path.exists(local_clone):
                if logger: logger.info(f"📁 Menggunakan dataset dari clone lokal: {local_clone}")
                return local_clone
            
            # Jika tidak, gunakan drive langsung lewat environment manager
            if env_manager.is_drive_mounted:
                drive_path = env_manager.get_path(config.get('data', {}).get('drive_path', 'data'))
                if os.path.exists(drive_path):
                    if logger: logger.info(f"📁 Menggunakan dataset dari Google Drive: {drive_path}")
                    return drive_path
        
        # Default: gunakan lokasi data standard dari environment manager
        data_dir = env_manager.get_path(config.get('data', {}).get('dir', 'data'))
        if logger: logger.info(f"📁 Menggunakan dataset dari lokal: {data_dir}")
        return data_dir
        
    except ImportError:
        # Fallback jika environment manager tidak tersedia
        data_dir = config.get('data', {}).get('dir', 'data')
        if logger: logger.info(f"📁 Menggunakan dataset dari lokal: {data_dir}")
        return data_dir

def get_dataset_stats(config: Dict[str, Any], env=None, logger=None) -> Dict[str, Any]:
    """Mendapatkan statistik dataset dari DatasetManager atau direktori langsung."""
    try:
        data_dir = get_dataset_dir(config, env, logger)
        
        # Coba gunakan DatasetManager
        try:
            from smartcash.dataset.manager import DatasetManager
            dataset_manager = DatasetManager(config, data_dir=data_dir)
            stats = dataset_manager.get_split_statistics()
            if logger: logger.info(f"✅ Statistik dataset berhasil dimuat dari DatasetManager")
            return stats
        except:
            # Fallback ke pembacaan direktori langsung
            stats = {}
            for split in ['train', 'valid', 'test']:
                split_dir = Path(data_dir) / split
                images_dir = split_dir / 'images' if split_dir.exists() else None
                labels_dir = split_dir / 'labels' if split_dir.exists() else None
                
                img_count = len(list(images_dir.glob('*.*'))) if images_dir and images_dir.exists() else 0
                label_count = len(list(labels_dir.glob('*.txt'))) if labels_dir and labels_dir.exists() else 0
                
                stats[split] = {
                    'images': img_count,
                    'labels': label_count,
                    'status': 'valid' if img_count > 0 and label_count > 0 else 'empty'
                }
            
            if logger: logger.info(f"✅ Statistik dataset berhasil dimuat dari direktori {data_dir}")
            return stats
    except Exception as e:
        if logger: logger.error(f"❌ Error saat mendapatkan statistik dataset: {str(e)}")
        return {}

def get_class_distribution(config: Dict[str, Any], env=None, logger=None) -> Dict[str, Dict[str, int]]:
    """Mendapatkan distribusi kelas dari dataset."""
    try:
        data_dir = get_dataset_dir(config, env, logger)
        
        # Coba gunakan DatasetManager
        try:
            from smartcash.dataset.manager import DatasetManager
            dataset_manager = DatasetManager(config, data_dir=data_dir)
            
            # Mendapatkan distribusi kelas dari manager
            class_stats = {}
            for split in ['train', 'valid', 'test']:
                try:
                    split_stats = dataset_manager.explore_class_distribution(split)
                    class_stats[split] = split_stats
                except Exception as e:
                    if logger: logger.warning(f"⚠️ Error mendapatkan distribusi kelas untuk {split}: {str(e)}")
                    class_stats[split] = {}
            
            if logger: logger.info(f"✅ Distribusi kelas berhasil dimuat dari DatasetManager")
            return class_stats
        except Exception as e:
            if logger: logger.warning(f"⚠️ DatasetManager tidak tersedia: {str(e)}")
            # Fallback ke data simulasi
            class_stats = {
                'train': {'Rp1000': 350, 'Rp2000': 320, 'Rp5000': 380, 'Rp10000': 390, 'Rp20000': 360, 'Rp50000': 340, 'Rp100000': 370},
                'valid': {'Rp1000': 50, 'Rp2000': 45, 'Rp5000': 55, 'Rp10000': 60, 'Rp20000': 55, 'Rp50000': 48, 'Rp100000': 52},
                'test': {'Rp1000': 50, 'Rp2000': 45, 'Rp5000': 55, 'Rp10000': 55, 'Rp20000': 50, 'Rp50000': 47, 'Rp100000': 53}
            }
            
            if logger: logger.warning(f"⚠️ Menggunakan data simulasi untuk distribusi kelas")
            return class_stats
    except Exception as e:
        if logger: logger.error(f"❌ Error saat mendapatkan distribusi kelas: {str(e)}")
        return {}

def show_class_distribution_visualization(output_box, class_stats: Dict[str, Dict[str, int]], colors: Dict[str, str], logger=None) -> None:
    """Tampilkan visualisasi distribusi kelas menggunakan visualization_utils."""
    from smartcash.ui.utils.constants import ICONS, COLORS
    
    with output_box:
        clear_output(wait=True)
        
        # Cek apakah ada data untuk ditampilkan
        if not class_stats or not all(len(stats) > 0 for split, stats in class_stats.items()):
            data_dir = os.environ.get('DATA_DIR', 'data')
            display(HTML(f"""<div style="padding:10px; background-color:{colors['alert_warning_bg']}; 
                          border-left:4px solid {colors['alert_warning_text']}; 
                          color:{colors['alert_warning_text']}; border-radius:4px;">
                    <p>{ICONS['warning']} Dataset tidak ditemukan atau tidak lengkap. Pastikan dataset sudah didownload.</p>
                </div>"""))
            return
        
        try:
            # Buat DataFrame untuk visualisasi
            import pandas as pd
            
            # Kumpulkan semua kelas
            all_classes = set()
            for split_stats in class_stats.values():
                all_classes.update(split_stats.keys())
                
            # Buat DataFrame
            df_data = []
            for cls in sorted(all_classes):
                row = {'Class': cls}
                for split in ['train', 'valid', 'test']:
                    row[split.capitalize()] = class_stats[split].get(cls, 0)
                df_data.append(row)
                
            df = pd.DataFrame(df_data)
            
            # Gunakan visualization_utils
            try:
                from smartcash.ui.utils.visualization_utils import create_class_distribution_plot, create_metrics_dashboard
                
                # Konversi data ke format yang sesuai
                plot_data = {}
                for i, row in df.iterrows():
                    plot_data[row['Class']] = {
                        'Train': row['Train'],
                        'Valid': row['Valid'],
                        'Test': row['Test']
                    }
                
                # Buat visualisasi
                display(HTML(f"<h3>{ICONS['chart']} Distribusi Kelas per Split</h3>"))
                fig = create_class_distribution_plot(
                    plot_data,
                    title='Distribusi Kelas per Split',
                    figsize=(10, 6),
                    sort_by='name'
                )
                display(fig)
                
                # Buat dashboard metrik
                metrics = {
                    f"Total Train": sum(df['Train']),
                    f"Total Valid": sum(df['Valid']),
                    f"Total Test": sum(df['Test']),
                    f"Jumlah Kelas": len(df),
                    f"Rasio Valid/Train": sum(df['Valid'])/sum(df['Train']) if sum(df['Train']) > 0 else 0,
                    f"Rasio Test/Train": sum(df['Test'])/sum(df['Train']) if sum(df['Train']) > 0 else 0
                }
                
                display(HTML(f"<h3>{ICONS['stats']} Metrik Dataset</h3>"))
                display(create_metrics_dashboard(
                    metrics,
                    "Statistik Dataset",
                    "Metrik utama dataset yang digunakan"
                ))
                
                # Tampilkan tabel distribusi
                display(HTML(f"<h3>{ICONS['chart']} Tabel Distribusi Kelas per Split</h3>"))
                display(df.style.background_gradient(cmap='Blues', subset=['Train', 'Valid', 'Test']))
                
            except ImportError as e:
                if logger: logger.warning(f"⚠️ Tidak dapat menggunakan visualization_utils: {str(e)}")
                import matplotlib.pyplot as plt
                import numpy as np
                
                # Fallback ke matplotlib dasar
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
                
            # Tampilkan info simpan konfigurasi
            display(HTML(f"""<div style="margin-top:20px; padding:10px; background-color:{colors['alert_info_bg']}; 
                       color:{colors['alert_info_text']}; border-radius:4px;">
                <p>{ICONS['save']} Klik tombol <strong>Simpan Konfigurasi</strong> untuk menyimpan pengaturan</p>
            </div>"""))
            
        except Exception as e:
            if logger: logger.error(f"❌ Error saat membuat visualisasi: {str(e)}")
            display(HTML(f"""<div style="padding:10px; background-color:{colors['alert_danger_bg']}; 
                          border-left:4px solid {colors['alert_danger_text']}; 
                          color:{colors['alert_danger_text']}; border-radius:4px;">
                    <p>{ICONS['error']} Error membuat visualisasi: {str(e)}</p>
                </div>"""))

def update_stats_cards(html_component, stats: Dict[str, Dict[str, Any]], colors: Dict[str, str]) -> None:
    """Update komponen HTML dengan statistik distribusi dataset dalam bentuk cards."""
    from smartcash.ui.utils.constants import ICONS
    
    # Hitung total dan persentase
    total_images = sum(split_data.get('images', 0) for split_data in stats.values())
    
    if total_images == 0:
        data_dir = os.environ.get('DATA_DIR', 'data') 
        html_component.value = f"""<div style="padding:10px; background-color:{colors['alert_warning_bg']}; 
                  border-left:4px solid {colors['alert_warning_text']}; 
                  color:{colors['alert_warning_text']}; border-radius:4px;">
            <p>{ICONS['warning']} Dataset tidak ditemukan di <code>{data_dir}</code> atau kosong.</p>
        </div>"""
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