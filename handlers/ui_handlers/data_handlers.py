"""
File: smartcash/handlers/ui_handlers/data_handlers.py
Author: Alfrida Sabar
Deskripsi: Handler untuk komponen UI data, termasuk statistik dataset dan fungsi utilitas data.
"""

import gc
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets  # Added for widget references
from IPython.display import display, clear_output, HTML
from typing import Dict, Any, Optional


def on_refresh_info_clicked(ui_components, data_manager, logger):
    """
    Handler untuk tombol refresh informasi dataset.
    
    Args:
        ui_components: Dictionary berisi komponen UI dari create_dataset_info_ui()
        data_manager: Instance dari DataManager untuk akses data
        logger: Logger untuk mencatat aktivitas
    """
    with ui_components['info_output']:
        clear_output()
        get_dataset_info(data_manager, logger)

def get_dataset_info(data_manager, logger):
    """
    Ambil dan tampilkan informasi dataset.
    
    Args:
        data_manager: Instance dari DataManager untuk akses data
        logger: Logger untuk mencatat aktivitas
    """
    try:
        # Tampilkan statistik dalam bentuk tabel
        stats = data_manager.get_dataset_stats()
        
        # Persiapkan data untuk DataFrame
        df_data = []
        layer_stats = {}
        
        # Proses statistik untuk tiap split
        for split, split_stats in stats.items():
            row = {'Split': split.capitalize()}
            
            # Ambil jumlah gambar dan label
            row['images'] = split_stats.get('image_count', 0)
            row['labels'] = split_stats.get('label_count', 0)
            
            # Dapatkan statistik original vs augmented jika tersedia
            if 'original' in split_stats and 'augmented' in split_stats:
                row['original'] = split_stats['original']
                row['augmented'] = split_stats['augmented']
            
            # Ambil statistik layer
            layer_stats_for_split = split_stats.get('layer_stats', {})
            for layer, count in layer_stats_for_split.items():
                if layer not in layer_stats:
                    layer_stats[layer] = {}
                layer_stats[layer][split] = count
            
            df_data.append(row)
        
        # Konversi ke DataFrame
        stats_df = pd.DataFrame(df_data)
        
        # Tambahkan total
        total_row = {'Split': 'Total'}
        for col in stats_df.columns:
            if col != 'Split' and stats_df[col].dtype.kind in 'iuf':  # Integer, unsigned int, atau float
                total_row[col] = stats_df[col].sum()
        
        stats_df = pd.concat([stats_df, pd.DataFrame([total_row])], ignore_index=True)
        
        # Tampilkan statistik dalam bentuk tabel
        display(HTML("<h3>📊 Statistik Dataset</h3>"))
        display(stats_df.style.format({
            'images': '{:,}', 
            'labels': '{:,}', 
            'augmented': '{:,}', 
            'original': '{:,}'
        }).highlight_max(axis=0, color='lightgreen', subset=['images', 'labels']))
        
        # Tampilkan statistik per layer jika tersedia
        if layer_stats:
            display(HTML("<h3>📊 Distribusi Layer</h3>"))
            layer_df = pd.DataFrame(layer_stats)
            display(layer_df)
            
            # Visualisasi layer distribution
            plt.figure(figsize=(10, 5))
            layer_df.plot(kind='bar', stacked=True, figsize=(10, 5))
            plt.title('Distribusi Layer per Split Dataset')
            plt.ylabel('Jumlah Objek')
            plt.xlabel('Split')
            plt.legend(title='Layer')
            plt.tight_layout()
            plt.show()
        
        # Plot distribusi data jika ada data
        if not stats_df.empty and len(stats_df) > 1:  # Minimal ada satu split + total
            plt.figure(figsize=(10, 5))
            
            # Exclude Total row
            plot_df = stats_df[stats_df['Split'] != 'Total']
            splits = plot_df['Split']
            
            # Siapkan data untuk plotting
            if 'original' in plot_df.columns and 'augmented' in plot_df.columns:
                original = plot_df['original'].fillna(0).values
                augmented = plot_df['augmented'].fillna(0).values
                
                # Create stacked bar
                bar_width = 0.6
                plt.bar(splits, original, bar_width, label='Original', color='#1f77b4')
                plt.bar(splits, augmented, bar_width, bottom=original, label='Augmented', color='#ff7f0e')
                
                # Add total labels on top
                for i, split in enumerate(splits):
                    total = original[i] + augmented[i]
                    plt.text(i, total + 5, f'Total: {int(total)}', ha='center', va='bottom', fontweight='bold')
            else:
                # Fallback jika kolom tidak tersedia
                images = plot_df['images'].fillna(0).values
                plt.bar(splits, images, color='#1f77b4')
                
                # Add labels on top
                for i, (split, count) in enumerate(zip(splits, images)):
                    plt.text(i, count + 5, f'Total: {int(count)}', ha='center', va='bottom', fontweight='bold')
            
            plt.title('Distribusi Dataset per Split')
            plt.ylabel('Jumlah Gambar')
            plt.legend()
            plt.tight_layout()
            plt.show()
        else:
            display(HTML("<p><i>Tidak cukup data untuk visualisasi.</i></p>"))
        
        # Tambahkan informasi ketersediaan dataloaders
        display(HTML("<h4>🔄 Ketersediaan DataLoader</h4>"))
        for split in ['train', 'valid', 'test']:
            try:
                # Dapatkan ukuran dataset
                size = data_manager.get_dataset_sizes().get(split, 0)
                status = "✅ Tersedia" if size > 0 else "❌ Tidak tersedia"
                display(HTML(f"<p><b>{split.capitalize()}</b>: {size} gambar - {status}</p>"))
            except Exception as e:
                display(HTML(f"<p><b>{split.capitalize()}</b>: Error - {str(e)}</p>"))
    except Exception as e:
        display(HTML(f"<p>❌ Error mendapatkan informasi dataset: {str(e)}</p>"))
        logger.error(f"Error mendapatkan informasi dataset: {str(e)}")

def on_split_button_clicked(ui_components, data_manager, preprocessor, logger):
    """
    Handler untuk tombol split dataset.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        data_manager: Instance dari DataManager
        preprocessor: Instance dari UnifiedPreprocessingHandler
        logger: Logger untuk mencatat aktivitas
    """
    # Disable tombol selama proses
    ui_components['split_button'].disabled = True
    ui_components['split_button'].description = "Memproses..."
    
    with ui_components['split_status_output']:
        clear_output()
        
        train_ratio = ui_components['train_ratio_slider'].value
        valid_ratio = ui_components['valid_ratio_slider'].value
        test_ratio = ui_components['test_ratio_slider'].value
        
        logger.info(f"🔄 Memulai split dataset dengan rasio: Train={train_ratio:.2f}, Valid={valid_ratio:.2f}, Test={test_ratio:.2f}...")
        
        # Pemanggilan preprocessor untuk melakukan split dataset
        try:
            results = preprocessor.split_dataset(
                train_ratio=train_ratio,
                valid_ratio=valid_ratio,
                test_ratio=test_ratio
            )
            
            if results.get('success', False):
                stats = results.get('stats', {})
                
                logger.success("✅ Dataset berhasil dibagi")
                logger.info(f"📊 Statistik:")
                logger.info(f"• Training: {stats.get('train', 0)} gambar")
                logger.info(f"• Validation: {stats.get('valid', 0)} gambar")
                logger.info(f"• Testing: {stats.get('test', 0)} gambar")
                
                # Visualisasikan statistik
                plt.figure(figsize=(8, 5))
                labels = ['Train', 'Validation', 'Test']
                sizes = [stats.get('train', 0), stats.get('valid', 0), stats.get('test', 0)]
                
                plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
                       colors=['#1f77b4', '#ff7f0e', '#2ca02c'])
                plt.axis('equal')
                plt.title('Distribusi Dataset')
                plt.show()
            else:
                logger.error(f"❌ Gagal membagi dataset: {results.get('message', 'Unknown error')}")
        except Exception as e:
            logger.error(f"❌ Error saat split dataset: {str(e)}")
        
        # Re-enable tombol
        ui_components['split_button'].disabled = False
        ui_components['split_button'].description = "Split Dataset"

def update_total_ratio(ui_components):
    """
    Handler untuk update tampilan total ratio saat slider berubah.
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    train = ui_components['train_ratio_slider'].value
    valid = ui_components['valid_ratio_slider'].value
    test = ui_components['test_ratio_slider'].value
    
    total = train + valid + test
    
    if abs(total - 1.0) < 0.001:  # Mendekati 1.0
        ui_components['total_ratio_text'].value = f"<b>Total Ratio: {total:.2f}</b> ✅"
        ui_components['split_button'].disabled = False
    else:
        ui_components['total_ratio_text'].value = f"<b>Total Ratio: {total:.2f}</b> ❌ (harus 1.0)"
        ui_components['split_button'].disabled = True

def check_data_availability(data_manager, logger):
    """
    Memeriksa ketersediaan data dan menampilkan statistik.
    
    Args:
        data_manager: Instance dari DataManager
        logger: Logger untuk mencatat aktivitas
        
    Returns:
        Boolean yang menunjukkan apakah data tersedia untuk training
    """
    try:
        # Dapatkan ukuran dataset
        sizes = {}
        for split in ['train', 'valid', 'test']:
            try:
                dataset = data_manager.get_dataset(split)
                sizes[split] = len(dataset) if dataset else 0
            except Exception as e:
                logger.warning(f"⚠️ Error mendapatkan ukuran dataset {split}: {str(e)}")
                sizes[split] = 0
        
        # Tampilkan informasi
        logger.info("🔍 Memeriksa ketersediaan data...")
        for split, count in sizes.items():
            status = "✅ Tersedia" if count > 0 else "❌ Tidak tersedia"
            logger.info(f"{split.capitalize()}: {count} gambar - {status}")
        
        # Periksa apakah data cukup untuk training
        if sizes.get('train', 0) > 0 and sizes.get('valid', 0) > 0:
            logger.success("✅ Data tersedia untuk training dan validasi")
            return True
        else:
            if sizes.get('train', 0) == 0:
                logger.warning("⚠️ Data training tidak tersedia")
            if sizes.get('valid', 0) == 0:
                logger.warning("⚠️ Data validasi tidak tersedia")
            
            logger.info("💡 Gunakan tab 'Split Dataset' untuk menyiapkan data training dan validasi")
            return False
    except Exception as e:
        logger.error(f"❌ Error saat memeriksa ketersediaan data: {str(e)}")
        return False

def visualize_batch(data_manager, split='train', num_images=4, figsize=(15, 10), logger=None):
    """
    Visualisasikan batch dataset untuk debugging dan verifikasi.
    
    Args:
        data_manager: Instance dari DataManager
        split: Split dataset ('train', 'valid', 'test')
        num_images: Jumlah gambar yang akan divisualisasikan
        figsize: Ukuran figure matplotlib
        logger: Optional logger untuk logging
    """
    try:
        # Dapatkan dataloader dengan batch size = num_images
        loader = data_manager.get_dataloader(
            split=split,
            batch_size=num_images,
            num_workers=0,  # Mengurangi kompleksitas
            shuffle=True    # Acak untuk mendapatkan sampel yang berbeda
        )
        
        # Ambil satu batch
        for images, targets in loader:
            # Konversi ke numpy untuk visualisasi
            if isinstance(images, torch.Tensor):
                images = images.cpu().numpy()
                
                # Transpose dari [B, C, H, W] ke [B, H, W, C]
                images = images.transpose(0, 2, 3, 1)
                
                # Denormalisasi jika perlu
                if images.max() <= 1.0:
                    images = images * 255
                
                images = images.astype(np.uint8)
            
            # Plot images
            fig, axes = plt.subplots(1, min(len(images), num_images), figsize=figsize)
            if min(len(images), num_images) == 1:
                axes = [axes]
                
            for i, ax in enumerate(axes):
                if i < len(images):
                    ax.imshow(images[i])
                    ax.set_title(f"Image {i+1}")
                    ax.axis('off')
            
            plt.tight_layout()
            plt.show()
            
            # Hanya tampilkan batch pertama
            break
    except Exception as e:
        if logger:
            logger.error(f"❌ Error saat visualisasi batch: {str(e)}")
        print(f"Gagal melakukan visualisasi: {str(e)}")

def setup_dataset_info_handlers(ui_components, data_manager, logger):
    """
    Setup handler untuk komponen UI informasi dataset.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        data_manager: Instance dari DataManager
        logger: Logger untuk mencatat aktivitas
        
    Returns:
        Dictionary berisi updated UI components dengan handler yang telah ditambahkan
    """
    # Bind event handler untuk tombol refresh
    ui_components['refresh_info_button'].on_click(
        lambda b: on_refresh_info_clicked(ui_components, data_manager, logger)
    )
    
    return ui_components

def setup_split_dataset_handlers(ui_components, data_manager, preprocessor, logger):
    """
    Setup handler untuk komponen UI split dataset.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        data_manager: Instance dari DataManager
        preprocessor: Instance dari UnifiedPreprocessingHandler
        logger: Logger untuk mencatat aktivitas
        
    Returns:
        Dictionary berisi updated UI components dengan handler yang telah ditambahkan
    """
    # Bind event handlers untuk slider
    ui_components['train_ratio_slider'].observe(
        lambda _: update_total_ratio(ui_components), names='value'
    )
    ui_components['valid_ratio_slider'].observe(
        lambda _: update_total_ratio(ui_components), names='value'
    )
    ui_components['test_ratio_slider'].observe(
        lambda _: update_total_ratio(ui_components), names='value'
    )
    
    # Panggil sekali untuk update nilai awal
    update_total_ratio(ui_components)
    
    # Bind event handler untuk tombol split
    ui_components['split_button'].on_click(
        lambda b: on_split_button_clicked(ui_components, data_manager, preprocessor, logger)
    )
    
    return ui_components