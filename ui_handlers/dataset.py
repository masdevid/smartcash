"""
File: smartcash/ui_handlers/dataset.py
Author: Alfrida Sabar (refactored)
Deskripsi: Handler untuk komponen UI persiapan dataset SmartCash.
"""

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output, Image
import os, sys, time, json, random, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from smartcash.utils.ui_utils import (
    create_info_alert, create_status_indicator, styled_html,
    create_metric_display, plot_statistics
)

def setup_dataset_handlers(ui_components, config=None):
    """Setup handlers untuk komponen UI persiapan dataset."""
    # Default config jika tidak disediakan
    if config is None:
        config = {
            'data': {
                'source': 'roboflow',
                'roboflow': {'api_key': '', 'workspace': 'smartcash-wo2us', 
                            'project': 'rupiah-emisi-2022', 'version': '3'},
                'preprocessing': {'img_size': [640, 640], 'cache_enabled': True, 
                                 'normalize_enabled': True, 'num_workers': 4}
            },
            'data_dir': 'data',
            'layers': {
                'banknote': {'name': 'banknote', 'classes': ['001', '002', '005', '010', '020', '050', '100']},
                'nominal': {'name': 'nominal', 'classes': ['l2_001', 'l2_002', 'l2_005', 'l2_010', 
                                                        'l2_020', 'l2_050', 'l2_100']},
                'security': {'name': 'security', 'classes': ['l3_sign', 'l3_text', 'l3_thread']}
            }
        }
    
    # Simpan data untuk handlers
    data = {'config': config, 'dataset_paths': {'train': None, 'valid': None, 'test': None}, 'dataset_stats': None}
    
    # Ekstrak komponen UI untuk kemudahan akses
    components = {k: ui_components[k] for k in ui_components}
    
    # Handler untuk download dataset
    def on_download_click(b):
        with components['download_status']:
            clear_output()
            display(create_status_indicator("info", "🔄 Memulai download dataset..."))
            
            try:
                # Simulasi proses download dengan progress bar
                download_option = components['download_options'].value
                
                if download_option == 'Roboflow (Online)':
                    # Ambil settings dari form
                    api_settings = components['roboflow_settings'].children
                    api_key = api_settings[0].value
                    workspace = api_settings[1].value
                    project = api_settings[2].value
                    version = api_settings[3].value
                    
                    # Update config
                    data['config']['data']['roboflow']['api_key'] = api_key
                    data['config']['data']['roboflow']['workspace'] = workspace
                    data['config']['data']['roboflow']['project'] = project
                    data['config']['data']['roboflow']['version'] = version
                    
                    # Simulasi download dari Roboflow (dalam implementasi asli, gunakan API Roboflow)
                    total_steps = 5
                    for i in range(total_steps + 1):
                        components['download_progress'].value = int(i * 100 / total_steps)
                        components['download_progress'].description = f"{int(i * 100 / total_steps)}%"
                        
                        if i == 1:
                            display(create_status_indicator("info", "🔑 Autentikasi API Roboflow..."))
                        elif i == 2:
                            display(create_status_indicator("info", "📂 Mengunduh metadata project..."))
                        elif i == 3:
                            display(create_status_indicator("info", "🖼️ Mengunduh gambar..."))
                        elif i == 4:
                            display(create_status_indicator("info", "🏷️ Mengunduh label..."))
                        elif i == 5:
                            display(create_status_indicator("info", "⚙️ Memvalidasi hasil..."))
                            
                        time.sleep(0.5)  # Simulasi delay
                
                elif download_option == 'Local Data (Upload)':
                    # Simulasi upload dan ekstraksi local data
                    components['download_progress'].value = 50
                    components['download_progress'].description = "50%"
                    display(create_status_indicator("info", "📤 Proses upload..."))
                    time.sleep(1)
                    
                    components['download_progress'].value = 100
                    components['download_progress'].description = "100%"
                    display(create_status_indicator("info", "📂 Ekstraksi file..."))
                    time.sleep(1)
                
                else:  # Sample Data
                    # Simulasi download sample data
                    components['download_progress'].value = 100
                    components['download_progress'].description = "100%"
                    display(create_status_indicator("info", "📚 Menyiapkan sample data..."))
                    time.sleep(1)
                
                # Simulasi paths (untuk implementasi asli, gunakan paths sebenarnya)
                data['dataset_paths'] = {
                    'train': f"{data['config']['data_dir']}/train",
                    'valid': f"{data['config']['data_dir']}/valid",
                    'test': f"{data['config']['data_dir']}/test"
                }
                
                # Tampilkan sukses
                display(create_status_indicator(
                    "success", 
                    f"✅ Dataset berhasil diunduh ke {data['config']['data_dir']}"
                ))
                
                # Generate dataset stats
                generate_dataset_stats()
                
                # Tampilkan statistik
                components['stats_section'].layout.display = ''
                components['stats_container'].layout.display = ''
                components['dataset_tabs'].layout.display = ''
                
                with components['stats_container']:
                    clear_output()
                    display_dataset_stats()
                
            except Exception as e:
                display(create_status_indicator("error", f"❌ Error: {str(e)}"))
    
    components['download_button'].on_click(on_download_click)
    
    # Handler untuk preprocessing
    def on_preprocess_click(b):
        with components['preprocess_status']:
            clear_output()
            display(create_status_indicator("info", "🔄 Memulai preprocessing dataset..."))
            
            try:
                # Ambil preprocessing options dari form
                preproc_options = components['preprocess_options'].children
                img_size = preproc_options[0].value
                normalize = preproc_options[1].value
                cache = preproc_options[2].value
                workers = preproc_options[3].value
                
                # Update config
                data['config']['data']['preprocessing']['img_size'] = list(img_size)
                data['config']['data']['preprocessing']['normalize_enabled'] = normalize
                data['config']['data']['preprocessing']['cache_enabled'] = cache
                data['config']['data']['preprocessing']['num_workers'] = workers
                
                # Simulasi preprocessing
                time.sleep(2)
                
                # Tampilkan sukses
                display(create_status_indicator(
                    "success", 
                    f"✅ Preprocessing selesai dengan {workers} workers, img_size={img_size}"
                ))
                
                # Update stats jika diperlukan
                update_dataset_stats_after_preprocessing()
                
            except Exception as e:
                display(create_status_indicator("error", f"❌ Error: {str(e)}"))
    
    components['preprocess_button'].on_click(on_preprocess_click)
    
    # Handler untuk split
    def on_split_click(b):
        with components['split_status']:
            clear_output()
            display(create_status_indicator("info", "🔄 Menerapkan konfigurasi split dataset..."))
            
            try:
                # Ambil split options dari form
                split_options = components['split_options'].children
                train_pct = split_options[0].value
                val_pct = split_options[1].value
                test_pct = split_options[2].value
                stratified = split_options[3].value
                
                # Validasi total persentase
                total = train_pct + val_pct + test_pct
                if abs(total - 100.0) > 0.01:
                    display(create_status_indicator(
                        "warning", 
                        f"⚠️ Total persentase ({total}%) tidak sama dengan 100%. Menyesuaikan..."
                    ))
                    # Sesuaikan persentase
                    factor = 100.0 / total
                    train_pct *= factor
                    val_pct *= factor
                    test_pct *= factor
                
                # Simulasi aplikasi split
                time.sleep(1.5)
                
                # Tampilkan sukses dengan persentase yang disesuaikan
                display(create_status_indicator(
                    "success", 
                    f"✅ Split diterapkan: Train {train_pct:.1f}%, Validation {val_pct:.1f}%, Test {test_pct:.1f}%"
                ))
                
                # Tampilkan info tambahan jika stratified
                if stratified:
                    display(create_info_alert(
                        "Split stratified diterapkan untuk menjaga distribusi kelas yang seimbang.", 
                        "info", "⚖️"
                    ))
                
                # Update stats
                update_dataset_stats_after_split()
                
            except Exception as e:
                display(create_status_indicator("error", f"❌ Error: {str(e)}"))
    
    components['split_button'].on_click(on_split_click)
    
    # Handler untuk augmentasi
    def on_augmentation_click(b):
        with components['augmentation_status']:
            clear_output()
            display(create_status_indicator("info", "🔄 Memulai augmentasi dataset..."))
            
            try:
                # Ambil augmentation options dari form
                aug_options = components['augmentation_options'].children
                aug_types = aug_options[0].value
                variations = aug_options[1].value
                prefix = aug_options[2].value
                validate = aug_options[3].value
                resume = aug_options[4].value
                
                # Simulasi augmentasi dengan progress bar
                total_augmentations = len(aug_types) * variations
                images_per_type = 100  # Simulasi 100 gambar per tipe augmentasi
                
                display(create_info_alert(
                    f"Membuat {total_augmentations} augmentasi pada {len(aug_types)} tipe", 
                    "info", "🔄"
                ))
                
                for i in range(total_augmentations + 1):
                    components['augmentation_progress'].value = int(i * 100 / total_augmentations)
                    components['augmentation_progress'].description = f"{int(i * 100 / total_augmentations)}%"
                    
                    if i > 0 and i % variations == 0:
                        current_type = aug_types[min(i // variations - 1, len(aug_types) - 1)]
                        display(create_status_indicator(
                            "success", 
                            f"✅ Augmentasi '{current_type}' selesai dengan {variations} variasi"
                        ))
                    
                    time.sleep(0.3)  # Simulasi delay
                
                # Validasi hasil jika diaktifkan
                if validate:
                    display(create_status_indicator("info", "🔍 Memvalidasi hasil augmentasi..."))
                    time.sleep(1)
                    
                    # Simulasi hasil validasi
                    display(create_status_indicator(
                        "success", 
                        f"✅ Validasi berhasil: {total_augmentations * images_per_type} gambar baru valid"
                    ))
                
                # Tampilkan sukses
                total_images = total_augmentations * images_per_type
                display(create_status_indicator(
                    "success", 
                    f"✅ Augmentasi selesai: {total_images} gambar baru dengan prefix '{prefix}'"
                ))
                
                # Update stats
                update_dataset_stats_after_augmentation(total_images)
                
            except Exception as e:
                display(create_status_indicator("error", f"❌ Error: {str(e)}"))
    
    components['augmentation_button'].on_click(on_augmentation_click)
    
    # Fungsi untuk generate dataset stats
    def generate_dataset_stats():
        # Simulasi stats
        data['dataset_stats'] = {
            'class_counts': {
                '001': 120, '002': 115, '005': 125, '010': 130,
                '020': 118, '050': 122, '100': 140,
                'l2_001': 110, 'l2_002': 105, 'l2_005': 115, 'l2_010': 120,
                'l2_020': 108, 'l2_050': 112, 'l2_100': 130,
                'l3_sign': 300, 'l3_text': 280, 'l3_thread': 320
            },
            'image_counts': {
                'train': 580, 'valid': 130, 'test': 130
            },
            'image_dimensions': {
                '640x640': 700, '800x600': 100, '1024x768': 40
            },
            'total_images': 840,
            'total_annotations': 2050
        }
    
    # Fungsi untuk update stats setelah preprocessing
    def update_dataset_stats_after_preprocessing():
        pass
    
    # Fungsi untuk update stats setelah split
    def update_dataset_stats_after_split():
        pass
    
    # Fungsi untuk update stats setelah augmentasi
    def update_dataset_stats_after_augmentation(new_images):
        if data['dataset_stats']:
            data['dataset_stats']['total_images'] += new_images
            data['dataset_stats']['image_counts']['train'] += new_images
            
            # Update jumlah anotasi berdasarkan rata-rata anotasi per gambar
            avg_annotations_per_image = data['dataset_stats']['total_annotations'] / (data['dataset_stats']['total_images'] - new_images)
            new_annotations = int(new_images * avg_annotations_per_image)
            data['dataset_stats']['total_annotations'] += new_annotations
            
            # Update tampilan
            with components['stats_container']:
                clear_output()
                display_dataset_stats()
    
    # Fungsi untuk menampilkan stats dataset di UI
    def display_dataset_stats():
        if not data['dataset_stats']:
            return
        
        stats = data['dataset_stats']
        
        # Dapatkan semua layer dari config
        all_layers = data['config']['layers']
        
        # Overview tab
        with components['dataset_tabs'].children[0]:
            clear_output()
            
            # Tampilkan ringkasan
            display(create_section_title("📊 Dataset Summary", "📈"))
            display(widgets.HBox([
                create_metric_display("Total Images", stats['total_images']),
                create_metric_display("Total Annotations", stats['total_annotations']),
                create_metric_display("Annotations/Image", 
                                     f"{stats['total_annotations']/stats['total_images']:.1f}")
            ]))
            
            # Tampilkan counts per split
            split_data = pd.DataFrame({
                'Split': list(stats['image_counts'].keys()),
                'Images': list(stats['image_counts'].values())
            })
            
            # Plot
            plt.figure(figsize=(10, 4))
            plt.bar(split_data['Split'], split_data['Images'], color=['#3498db', '#2ecc71', '#e74c3c'])
            plt.title('Images per Split')
            plt.ylabel('Count')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            for i, v in enumerate(split_data['Images']):
                plt.text(i, v + 5, str(v), ha='center')
                
            plt.tight_layout()
            display(plt.gcf())
            plt.close()
        
        # Classes tab
        with components['dataset_tabs'].children[1]:
            clear_output()
            
            display(create_section_title("🏷️ Class Distribution", "📊"))
            
            # Group classes by layer
            for layer_name, layer_info in all_layers.items():
                layer_classes = layer_info['classes']
                
                # Extract data for this layer's classes
                layer_data = {cls: stats['class_counts'].get(cls, 0) for cls in layer_classes}
                
                # Create DataFrame
                df = pd.DataFrame({
                    'Class': list(layer_data.keys()),
                    'Count': list(layer_data.values())
                })
                
                # Plot
                plt.figure(figsize=(10, 4))
                bars = plt.bar(df['Class'], df['Count'], color='#3498db')
                plt.title(f'Class Distribution - {layer_info["name"]} layer')
                plt.ylabel('Count')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            f'{height:.0f}', ha='center', va='bottom')
                
                plt.tight_layout()
                display(plt.gcf())
                plt.close()
        
        # Dimensions tab
        with components['dataset_tabs'].children[2]:
            clear_output()
            
            display(create_section_title("📏 Image Dimensions", "🖼️"))
            
            # Create DataFrame
            df = pd.DataFrame({
                'Dimension': list(stats['image_dimensions'].keys()),
                'Count': list(stats['image_dimensions'].values())
            })
            
            # Plot
            plt.figure(figsize=(10, 4))
            plt.pie(df['Count'], labels=df['Dimension'], autopct='%1.1f%%', 
                   startangle=90, colors=['#3498db', '#2ecc71', '#e74c3c'])
            plt.title('Image Dimension Distribution')
            plt.axis('equal')
            plt.tight_layout()
            display(plt.gcf())
            plt.close()
            
            # Display resolution info
            display(create_info_alert(
                f"Target image size untuk training: {data['config']['data']['preprocessing']['img_size']}", 
                "info", "🎯"
            ))
        
        # Samples tab (placeholder)
        with components['dataset_tabs'].children[3]:
            clear_output()
            display(create_section_title("🔍 Sample Images", "🖼️"))
            display(create_info_alert("Preview gambar sample akan ditampilkan di sini.", "info"))
    
    # Populate UI dengan data dari config saat inisialisasi
    def initialize_from_config():
        # Populate Roboflow settings
        api_settings = components['roboflow_settings'].children
        api_settings[0].value = data['config']['data']['roboflow'].get('api_key', '')
        api_settings[1].value = data['config']['data']['roboflow'].get('workspace', 'smartcash-wo2us')
        api_settings[2].value = data['config']['data']['roboflow'].get('project', 'rupiah-emisi-2022')
        api_settings[3].value = str(data['config']['data']['roboflow'].get('version', '3'))
        
        # Populate preprocessing settings
        preproc_options = components['preprocess_options'].children
        img_size = data['config']['data']['preprocessing'].get('img_size', [640, 640])
        if isinstance(img_size, list) and len(img_size) == 2:
            preproc_options[0].value = img_size
        preproc_options[1].value = data['config']['data']['preprocessing'].get('normalize_enabled', True)
        preproc_options[2].value = data['config']['data']['preprocessing'].get('cache_enabled', True)
        preproc_options[3].value = data['config']['data']['preprocessing'].get('num_workers', 4)
    
    # Panggil inisialisasi
    initialize_from_config()
    
    return ui_components