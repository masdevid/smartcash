"""
File: handlers/ui_handlers/augmentation_handlers.py
Author: Alfrida Sabar
Deskripsi: Handler untuk UI komponen augmentasi data, menangani proses augmentasi dan visualisasi hasil.
"""

import gc
import time
import pandas as pd
import torch  # Added for CUDA checks
from IPython.display import display, clear_output, HTML
import ipywidgets as widgets  # Added for widget references

def on_augment_button_clicked(ui_components, aug_manager, data_manager, logger):
    """
    Handler untuk tombol augmentasi data.
    
    Args:
        ui_components: Dictionary komponen UI dari create_augmentation_ui()
        aug_manager: Instance dari OptimizedAugmentation manager
        data_manager: Instance dari DataManager
        logger: Logger untuk mencatat aktivitas
    """
    # Validate UI components
    required_components = ['augment_button', 'output', 'split_selection', 
                          'augmentation_type', 'num_workers_slider', 'num_variations_slider']
    
    for component in required_components:
        if component not in ui_components:
            logger.error(f"❌ Required UI component '{component}' not found")
            return
    
    # Disable tombol selama proses
    ui_components['augment_button'].disabled = True
    ui_components['augment_button'].description = "Memproses..."
    
    with ui_components['output']:
        clear_output()
        
        splits = ui_components['split_selection'].value
        aug_type = ui_components['augmentation_type'].value
        num_workers = ui_components['num_workers_slider'].value
        num_variations = ui_components['num_variations_slider'].value
        
        if not splits:
            print("⚠️ Pilih setidaknya satu split dataset!")
            ui_components['augment_button'].disabled = False
            ui_components['augment_button'].description = "Augmentasi Data"
            return
            
        logger.info(f"🎨 Memulai augmentasi untuk {', '.join(splits)} dengan tipe {aug_type}...")
        
        all_stats = {}
        for split in splits:
            logger.info(f"🔄 Memproses split {split}...")
            
            try:
                # Gunakan OptimizedAugmentation
                stats = aug_manager.augment_dataset(
                    split=split,
                    augmentation_types=[aug_type],
                    num_variations=num_variations,
                    num_workers=num_workers,
                    validate_results=True
                )
                
                all_stats[split] = {
                    'processed': stats.get('processed', 0),
                    'augmented': stats.get('augmented', 0),
                    'failed': stats.get('failed', 0),
                    'duration': stats.get('duration', 0)
                }
                
                logger.success(f"✅ Augmentasi {split} selesai: {stats.get('augmented', 0)} gambar dihasilkan")
                
            except Exception as e:
                logger.error(f"❌ Error saat mengaugmentasi {split}: {str(e)}")
                all_stats[split] = {'error': str(e)}
            
        logger.success("✨ Augmentasi selesai!")
        
        # Tampilkan statistik dalam bentuk tabel
        stats_data = []
        for split, stat in all_stats.items():
            if 'error' in stat:
                stats_data.append({
                    'Split': split.capitalize(),
                    'Status': 'Error',
                    'Error': stat['error']
                })
            else:
                stats_data.append({
                    'Split': split.capitalize(),
                    'Diproses': stat['processed'],
                    'Hasil Augmentasi': stat['augmented'],
                    'Gagal': stat['failed'],
                    'Durasi (detik)': round(stat.get('duration', 0), 2)
                })
            
        if stats_data:
            display(pd.DataFrame(stats_data))
            
        # Refresh statistik dataset
        try:
            # Mencoba mendapatkan statistik dataset terbaru
            refresh_dataset_stats(data_manager, logger)
        except Exception as e:
            logger.warning(f"⚠️ Tidak dapat merefresh statistik dataset: {str(e)}")
        
        # Force garbage collection
        gc.collect()
        
    # Re-enable tombol
    ui_components['augment_button'].disabled = False
    ui_components['augment_button'].description = "Augmentasi Data"

def on_clean_button_clicked(ui_components, aug_manager, data_manager, logger):
    """
    Handler untuk tombol membersihkan hasil augmentasi.
    
    Args:
        ui_components: Dictionary komponen UI dari create_augmentation_ui()
        aug_manager: Instance dari OptimizedAugmentation manager
        data_manager: Instance dari DataManager
        logger: Logger untuk mencatat aktivitas
    """
    # Validate UI components
    required_components = ['clean_button', 'output', 'split_selection']
    
    for component in required_components:
        if component not in ui_components:
            logger.error(f"❌ Required UI component '{component}' not found")
            return
    
    # Disable tombol selama proses
    ui_components['clean_button'].disabled = True
    ui_components['clean_button'].description = "Membersihkan..."
    
    with ui_components['output']:
        clear_output()
        
        splits = ui_components['split_selection'].value
        
        if not splits:
            print("⚠️ Pilih setidaknya satu split dataset!")
            ui_components['clean_button'].disabled = False
            ui_components['clean_button'].description = "Bersihkan Augmentasi"
            return
            
        logger.info(f"🧹 Membersihkan hasil augmentasi untuk {', '.join(splits)}...")
        
        try:
            # Gunakan clean_augmented_data dari OptimizedAugmentation
            stats = aug_manager.clean_augmented_data(splits=list(splits))
            
            logger.success("🧹 Pembersihan selesai!")
            logger.info(f"📊 Statistik:")
            logger.info(f"• Gambar dihapus: {stats.get('removed_images', 0)}")
            logger.info(f"• Label dihapus: {stats.get('removed_labels', 0)}")
            
            # Tampilkan statistik
            display(pd.DataFrame([{
                'Gambar Dihapus': stats.get('removed_images', 0),
                'Label Dihapus': stats.get('removed_labels', 0),
                'Error': stats.get('errors', 0)
            }]))
            
            # Refresh statistik dataset
            try:
                refresh_dataset_stats(data_manager, logger)
            except Exception as e:
                logger.warning(f"⚠️ Tidak dapat merefresh statistik dataset: {str(e)}")
            
        except Exception as e:
            logger.error(f"❌ Error saat membersihkan augmentasi: {str(e)}")
            display(HTML(f"<p><strong>Error:</strong> {str(e)}</p>"))
        
        # Force garbage collection
        gc.collect()
        
    # Re-enable tombol
    ui_components['clean_button'].disabled = False
    ui_components['clean_button'].description = "Bersihkan Augmentasi"

def refresh_dataset_stats(data_manager, logger):
    """
    Refresh dan tampilkan statistik dataset setelah augmentasi.
    
    Args:
        data_manager: Instance dari DataManager
        logger: Logger untuk mencatat aktivitas
    """
    try:
        # Dapatkan statistik dataset
        stats = data_manager.get_dataset_stats()
        
        # Persiapkan data untuk DataFrame
        df_data = []
        layer_stats = {}
        
        # Proses statistik untuk tiap split
        for split, split_stats in stats.items():
            row = {'Split': split.capitalize()}
            
            # Ambil jumlah gambar dan label
            row['Images'] = split_stats.get('image_count', 0)
            row['Labels'] = split_stats.get('label_count', 0)
            
            # Dapatkan statistik original vs augmented jika tersedia
            if 'original' in split_stats and 'augmented' in split_stats:
                row['Original'] = split_stats['original']
                row['Augmented'] = split_stats['augmented']
            
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
        display(HTML("<h3>📊 Statistik Dataset (Terbaru)</h3>"))
        display(stats_df.style.format({
            'Images': '{:,}', 
            'Labels': '{:,}', 
            'Augmented': '{:,}', 
            'Original': '{:,}'
        }).highlight_max(axis=0, color='lightgreen', subset=['Images', 'Labels']))
        
        logger.info("✅ Statistik dataset berhasil diperbarui")
        
    except Exception as e:
        logger.error(f"❌ Error saat refresh statistik dataset: {str(e)}")
        raise

def setup_augmentation_handlers(ui_components, aug_manager, data_manager, logger):
    """
    Setup semua event handlers untuk UI augmentasi data.
    
    Args:
        ui_components: Dictionary komponen UI dari create_augmentation_ui()
        aug_manager: Instance dari OptimizedAugmentation manager
        data_manager: Instance dari DataManager
        logger: Logger untuk mencatat aktivitas
        
    Returns:
        Dictionary updated UI components dengan handlers yang sudah di-attach
    """
    # Validate UI components
    required_components = ['augment_button', 'clean_button', 'output', 'split_selection', 
                         'augmentation_type', 'num_workers_slider', 'num_variations_slider']
    
    for component in required_components:
        if component not in ui_components:
            logger.error(f"❌ Required UI component '{component}' not found")
            return ui_components
    
    # Setup handler untuk tombol augmentasi
    ui_components['augment_button'].on_click(
        lambda button: on_augment_button_clicked(ui_components, aug_manager, data_manager, logger)
    )
    
    # Setup handler untuk tombol clean
    ui_components['clean_button'].on_click(
        lambda button: on_clean_button_clicked(ui_components, aug_manager, data_manager, logger)
    )
    
    return ui_components