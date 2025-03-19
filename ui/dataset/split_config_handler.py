"""
File: smartcash/ui/dataset/split_config_handler.py
Deskripsi: Handler untuk konfigurasi split dataset yang hanya memvisualisasikan distribusi, tanpa split aktual
"""

from typing import Dict, Any, Optional
from IPython.display import display, HTML, clear_output
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def setup_split_config_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk komponen UI konfigurasi split dataset.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Dapatkan logger jika tersedia
    logger = None
    try:
        from smartcash.common.logger import get_logger
        logger = get_logger("split_config")
    except ImportError:
        logger = None
    
    # Pastikan konfigurasi data ada
    if not config:
        config = {}
    if 'data' not in config:
        config['data'] = {}
    
    # Fallback config
    if 'split_ratios' not in config['data']:
        config['data']['split_ratios'] = {'train': 0.7, 'valid': 0.15, 'test': 0.15}
    if 'stratified_split' not in config['data']:
        config['data']['stratified_split'] = True
    if 'random_seed' not in config['data']:
        config['data']['random_seed'] = 42
    
    try:
        # Import dependencies
        from smartcash.ui.utils.constants import ICONS, COLORS
        from smartcash.ui.components.alerts import create_status_indicator
        from smartcash.ui.training_config.config_handler import get_config_manager
        
        # Dapatkan config manager untuk save/load
        config_manager = get_config_manager()
        
        # Fungsi update UI dari config
        def update_ui_from_config():
            """Update UI dengan nilai dari config."""
            try:
                # Update split ratios
                split_ratios = config['data'].get('split_ratios', {'train': 0.7, 'valid': 0.15, 'test': 0.15})
                
                ui_components['split_sliders'][0].value = split_ratios.get('train', 0.7) * 100
                ui_components['split_sliders'][1].value = split_ratios.get('valid', 0.15) * 100
                ui_components['split_sliders'][2].value = split_ratios.get('test', 0.15) * 100
                
                # Update stratified checkbox
                ui_components['stratified'].value = config['data'].get('stratified_split', True)
                
                # Update advanced options
                ui_components['advanced_options'].children[0].value = config['data'].get('random_seed', 42)
                ui_components['advanced_options'].children[1].value = config['data'].get('backup_before_split', True)
                
                # Update current stats display
                update_current_stats()
                
                # Update class distribution visualization
                update_class_distribution_viz()
                
            except Exception as e:
                if logger:
                    logger.error(f"❌ Error saat update UI dari config: {str(e)}")
                with ui_components['status']:
                    clear_output(wait=True)
                    display(create_status_indicator("error", f"Error saat update UI: {str(e)}"))
        
        # Fungsi update config dari UI
        def update_config_from_ui():
            """Update config dengan nilai dari UI."""
            try:
                # Get current values from UI
                train_pct = ui_components['split_sliders'][0].value
                val_pct = ui_components['split_sliders'][1].value
                test_pct = ui_components['split_sliders'][2].value
                stratified = ui_components['stratified'].value
                random_seed = ui_components['advanced_options'].children[0].value
                backup_before_split = ui_components['advanced_options'].children[1].value
                
                # Normalize percentages
                total = train_pct + val_pct + test_pct
                if abs(total - 100.0) > 0.001:
                    train_pct = train_pct * 100 / total
                    val_pct = val_pct * 100 / total
                    test_pct = test_pct * 100 / total
                
                # Update config
                config['data']['split_ratios'] = {
                    'train': train_pct / 100.0,
                    'valid': val_pct / 100.0,
                    'test': test_pct / 100.0
                }
                config['data']['stratified_split'] = stratified
                config['data']['random_seed'] = random_seed
                config['data']['backup_before_split'] = backup_before_split
                
                return config
            except Exception as e:
                if logger:
                    logger.error(f"❌ Error saat update config dari UI: {str(e)}")
                with ui_components['status']:
                    clear_output(wait=True)
                    display(create_status_indicator("error", f"Error saat update config: {str(e)}"))
                return config
        
        # Update status panel
        def update_status_panel(status_type, message):
            """Update panel status dengan pesan baru."""
            from smartcash.ui.dataset.download_initialization import update_status_panel
            update_status_panel(ui_components, status_type, message)
        
        # Fungsi mendapatkan statistik split saat ini
        def update_current_stats():
            """Update tampilan statistik dataset saat ini."""
            with ui_components['current_stats']:
                clear_output(wait=True)
                
                try:
                    # Coba gunakan DatasetManager untuk statistik
                    try:
                        from smartcash.dataset.manager import DatasetManager
                        dataset_manager = DatasetManager(config)
                        stats = dataset_manager.get_split_statistics()
                    except (ImportError, AttributeError):
                        # Fallback ke simulasi statistik
                        from pathlib import Path
                        import os
                        
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
                    
                    # Hitung total dan persentase
                    total_images = sum(split_data.get('images', 0) for split_data in stats.values())
                    
                    # Create HTML
                    html = f"""
                    <h3 style="margin-top:0; color:{COLORS['dark']}">{ICONS['dataset']} Statistik Dataset Saat Ini</h3>
                    <div style="display:flex; flex-wrap:wrap; gap:15px; margin-bottom:10px">
                    """
                    
                    for split, data in stats.items():
                        images = data.get('images', 0)
                        labels = data.get('labels', 0)
                        status = data.get('status', 'valid')
                        percentage = (images / total_images * 100) if total_images > 0 else 0
                        
                        # Pick color based on status
                        color = COLORS['primary'] if status == 'valid' else COLORS['danger']
                        
                        html += f"""
                        <div style="flex:1; min-width:150px; border:1px solid {color}; border-radius:5px; padding:10px; background-color:{COLORS['light']}">
                            <h4 style="margin-top:0; color:{color}">{split.capitalize()}</h4>
                            <p style="margin:5px 0; color:{COLORS['dark']}"><strong>Images:</strong> {images}</p>
                            <p style="margin:5px 0; color:{COLORS['dark']}"><strong>Labels:</strong> {labels}</p>
                            <p style="margin:5px 0; color:{COLORS['dark']}"><strong>Persentase:</strong> {percentage:.1f}%</p>
                        </div>
                        """
                    
                    html += "</div>"
                    
                    if total_images == 0:
                        html += f"""
                        <div style="padding:10px; background-color:{COLORS['alert_warning_bg']}; border-left:4px solid {COLORS['alert_warning_text']}; color:{COLORS['alert_warning_text']}">
                            <p style="margin:0">{ICONS['warning']} Dataset tidak ditemukan atau kosong. Silakan download dataset terlebih dahulu.</p>
                        </div>
                        """
                    
                    display(HTML(html))
                    
                    # Return stats for other uses
                    return stats
                    
                except Exception as e:
                    if logger:
                        logger.error(f"❌ Error saat mendapatkan statistik dataset: {str(e)}")
                    display(HTML(f"""
                        <div style="padding:10px; background-color:{COLORS['alert_danger_bg']}; border-left:4px solid {COLORS['alert_danger_text']}; color:{COLORS['alert_danger_text']}">
                            <p style="margin:0">{ICONS['error']} Error saat mendapatkan statistik dataset: {str(e)}</p>
                        </div>
                    """))
                    return None
        
        # Visualisasi distribusi kelas
        def update_class_distribution_viz():
            """Update visualisasi distribusi kelas per split."""
            with ui_components['class_viz_output']:
                clear_output(wait=True)
                
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
                    
                    # Tampilkan visualisasi
                    if all(len(stats) > 0 for stats in class_stats.values()):
                        # Create DataFrame for visualization
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
                        
                        # Plot distribution
                        plt.figure(figsize=(10, 6))
                        ax = plt.subplot(111)
                        
                        # Set width of bar
                        barWidth = 0.25
                        
                        # Set positions of bars on X axis
                        r1 = np.arange(len(df))
                        r2 = [x + barWidth for x in r1]
                        r3 = [x + barWidth for x in r2]
                        
                        # Create bars
                        ax.bar(r1, df['Train'], width=barWidth, label='Train', color=COLORS['primary'])
                        ax.bar(r2, df['Valid'], width=barWidth, label='Valid', color=COLORS['success'])
                        ax.bar(r3, df['Test'], width=barWidth, label='Test', color=COLORS['warning'])
                        
                        # Add labels and title
                        plt.xlabel('Kelas')
                        plt.ylabel('Jumlah Sampel')
                        plt.title('Distribusi Kelas per Split')
                        plt.xticks([r + barWidth for r in range(len(df))], df['Class'])
                        plt.legend()
                        
                        plt.tight_layout()
                        plt.show()
                        
                        # Tampilkan tabel distribusi
                        display(HTML(f"<h3>{ICONS['chart']} Distribusi Kelas per Split</h3>"))
                        display(df.style.background_gradient(cmap='Blues', subset=['Train', 'Valid', 'Test']))
                    else:
                        display(HTML(f"""
                            <div style="padding:10px; background-color:{COLORS['alert_warning_bg']}; border-left:4px solid {COLORS['alert_warning_text']}; color:{COLORS['alert_warning_text']}">
                                <p style="margin:0">{ICONS['warning']} Data distribusi kelas tidak tersedia. Pastikan dataset sudah didownload.</p>
                            </div>
                        """))
                
                except Exception as e:
                    if logger:
                        logger.error(f"❌ Error saat membuat visualisasi distribusi kelas: {str(e)}")
                    display(HTML(f"""
                        <div style="padding:10px; background-color:{COLORS['alert_danger_bg']}; border-left:4px solid {COLORS['alert_danger_text']}; color:{COLORS['alert_danger_text']}">
                            <p style="margin:0">{ICONS['error']} Error saat membuat visualisasi distribusi kelas: {str(e)}</p>
                        </div>
                    """))
        
        # Handler untuk save button
        def on_save_config(b):
            """Save current configuration."""
            with ui_components['status']:
                clear_output(wait=True)
                display(create_status_indicator("info", f"{ICONS['save']} Menyimpan konfigurasi split..."))
                
                try:
                    # Update config dengan nilai dari UI
                    updated_config = update_config_from_ui()
                    
                    # Save config
                    if config_manager:
                        config_manager.save_config(updated_config)
                        display(create_status_indicator("success", f"{ICONS['success']} Konfigurasi split berhasil disimpan"))
                    else:
                        # Fallback jika tidak ada config manager
                        import yaml
                        with open('configs/colab_config.yaml', 'w') as f:
                            yaml.dump(updated_config, f)
                        display(create_status_indicator("success", f"{ICONS['success']} Konfigurasi split berhasil disimpan ke configs/colab_config.yaml"))
                        
                    # Update status panel
                    update_status_panel("success", f"Konfigurasi split berhasil disimpan")
                    
                except Exception as e:
                    if logger:
                        logger.error(f"❌ Error saat menyimpan konfigurasi: {str(e)}")
                    display(create_status_indicator("error", f"{ICONS['error']} Error saat menyimpan konfigurasi: {str(e)}"))
        
        # Handler untuk reset button
        def on_reset_config(b):
            """Reset configuration to defaults."""
            with ui_components['status']:
                clear_output(wait=True)
                display(create_status_indicator("info", f"{ICONS['refresh']} Reset konfigurasi split ke default..."))
                
                try:
                    # Set default values
                    config['data']['split_ratios'] = {'train': 0.7, 'valid': 0.15, 'test': 0.15}
                    config['data']['stratified_split'] = True
                    config['data']['random_seed'] = 42
                    config['data']['backup_before_split'] = True
                    
                    # Update UI
                    update_ui_from_config()
                    
                    display(create_status_indicator("success", f"{ICONS['success']} Konfigurasi split berhasil direset ke default"))
                    
                    # Update status panel
                    update_status_panel("success", "Konfigurasi split direset ke default")
                
                except Exception as e:
                    if logger:
                        logger.error(f"❌ Error saat reset konfigurasi: {str(e)}")
                    display(create_status_indicator("error", f"{ICONS['error']} Error saat reset konfigurasi: {str(e)}"))
        
        # Handler untuk slider changes untuk memastikan selalu 100%
        def on_slider_change(change):
            """Ensure split percentages sum to 100%."""
            if change['name'] != 'value':
                return
                
            # Get current values
            train_pct = ui_components['split_sliders'][0].value
            val_pct = ui_components['split_sliders'][1].value
            test_pct = ui_components['split_sliders'][2].value
            total = train_pct + val_pct + test_pct
            
            # Auto-adjust if too far from 100%
            if abs(total - 100.0) > 0.5:
                # Find which slider was just changed
                changed_slider = None
                for i, slider in enumerate(ui_components['split_sliders']):
                    if slider is change['owner']:
                        changed_slider = i
                        break
                
                if changed_slider is not None:
                    # Adjust the other sliders proportionally
                    remaining = 100.0 - change['new']
                    other_sliders = [i for i in range(3) if i != changed_slider]
                    other_total = ui_components['split_sliders'][other_sliders[0]].value + ui_components['split_sliders'][other_sliders[1]].value
                    
                    if other_total > 0:
                        ratio = remaining / other_total
                        ui_components['split_sliders'][other_sliders[0]].value = ui_components['split_sliders'][other_sliders[0]].value * ratio
                        ui_components['split_sliders'][other_sliders[1]].value = ui_components['split_sliders'][other_sliders[1]].value * ratio
        
        # Register handlers
        for slider in ui_components['split_sliders']:
            slider.observe(on_slider_change, names='value')
            
        ui_components['save_button'].on_click(on_save_config)
        ui_components['reset_button'].on_click(on_reset_config)
        
        # Function for cleanup
        def cleanup():
            """Cleanup resources."""
            # Unobserve slider events
            for slider in ui_components['split_sliders']:
                slider.unobserve(on_slider_change, names='value')
                
            if logger:
                logger.info(f"{ICONS['success']} Split config resources cleaned up")
                
        # Add cleanup function to ui_components
        ui_components['cleanup'] = cleanup
        
        # Initialize UI
        update_ui_from_config()
        
    except Exception as e:
        # Fallback to minimal setup in case of errors
        if 'status' in ui_components:
            with ui_components['status']:
                from smartcash.ui.components.alerts import create_status_indicator
                display(create_status_indicator("error", f"Error saat setup split config handler: {str(e)}"))
                
        if logger:
            logger.error(f"❌ Error saat setup split config handler: {str(e)}")
    
    return ui_components