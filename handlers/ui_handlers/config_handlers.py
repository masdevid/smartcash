"""
File: smartcash/handlers/ui_handlers/config_handlers.py
Author: Alfrida Sabar
Deskripsi: Handler untuk UI komponen konfigurasi, menangani penyimpanan dan update konfigurasi global.
"""

import os
import yaml
import pickle
from datetime import datetime
from pathlib import Path
from IPython.display import display, clear_output, HTML
import ipywidgets as widgets  # Added for widget references
import sys  # Added for system operations


def update_config_from_ui(ui_components, config):
    """
    Update konfigurasi dari nilai-nilai UI.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        config: Dictionary konfigurasi yang akan diupdate
        
    Returns:
        Dictionary berisi konfigurasi yang telah diupdate
    """
    # Update model configurations
    if 'model' not in config:
        config['model'] = {}
    
    config['model']['backbone'] = ui_components['backbone_dropdown'].value
    config['model']['pretrained'] = ui_components['pretrained_checkbox'].value
    config['model']['img_size'] = [ui_components['img_size_slider'].value, ui_components['img_size_slider'].value]
    config['model']['batch_size'] = ui_components['batch_size_slider'].value
    config['model']['workers'] = ui_components['workers_slider'].value
    
    # Update training configurations
    if 'training' not in config:
        config['training'] = {}
    
    config['training']['learning_rate'] = ui_components['lr_dropdown'].value
    config['training']['epochs'] = ui_components['epochs_slider'].value
    config['training']['optimizer'] = ui_components['optimizer_dropdown'].value
    config['training']['scheduler'] = ui_components['scheduler_dropdown'].value
    
    # Update layers
    selected_layers = list(ui_components['layer_selection'].value)
    if selected_layers:  # Jika ada layer yang dipilih
        config['layers'] = selected_layers
    
    return config

def save_config_to_file(config, config_path, logger):
    """
    Simpan konfigurasi ke file YAML.
    
    Args:
        config: Dictionary konfigurasi
        config_path: Path ke file konfigurasi
        logger: Logger untuk mencatat aktivitas
        
    Returns:
        Boolean yang menunjukkan keberhasilan
    """
    try:
        # Pastikan direktori ada
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Simpan ke YAML
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Buat backup pickle untuk digunakan di cells lain
        with open('config.pkl', 'wb') as f:
            pickle.dump(config, f)
            
        logger.info(f"📝 Konfigurasi disimpan di {config_path}")
        return True
    
    except Exception as e:
        logger.error(f"❌ Error saat menyimpan konfigurasi: {str(e)}")
        return False

def update_layer_info(ui_components, active_layers):
    """
    Update tampilan informasi layer aktif.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        active_layers: List layer yang aktif
    """
    ui_components['layer_info'].value = f"""
    <div style="margin: 10px 0; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
        <p><b>🔍 Layer yang diaktifkan:</b> {', '.join(active_layers)}</p>
    </div>
    """

def on_save_config_button_clicked(b, ui_components, config, config_path, logger):
    """
    Handler untuk tombol simpan konfigurasi.
    
    Args:
        b: Button instance
        ui_components: Dictionary berisi komponen UI
        config: Dictionary konfigurasi
        config_path: Path ke file konfigurasi
        logger: Logger untuk mencatat aktivitas
    """
    # Disable tombol selama proses
    ui_components['save_config_button'].disabled = True
    
    with ui_components['output_area']:
        clear_output()
        
        try:
            # Update konfigurasi dari UI
            updated_config = update_config_from_ui(ui_components, config)
            
            # Simpan konfigurasi
            if save_config_to_file(updated_config, config_path, logger):
                print(f"✅ Konfigurasi berhasil disimpan ke {config_path}")
                
                # Update tampilan layer info
                update_layer_info(ui_components, updated_config['layers'])
                
                # Tampilkan ringkasan konfigurasi
                print("\n📋 Ringkasan konfigurasi:")
                print(f"• Backbone: {updated_config['model']['backbone']}")
                print(f"• Pretrained: {'Ya' if updated_config['model']['pretrained'] else 'Tidak'}")
                print(f"• Image Size: {updated_config['model']['img_size'][0]}x{updated_config['model']['img_size'][1]}")
                print(f"• Batch Size: {updated_config['model']['batch_size']}")
                print(f"• Workers: {updated_config['model']['workers']}")
                print(f"• Learning Rate: {updated_config['training']['learning_rate']}")
                print(f"• Epochs: {updated_config['training']['epochs']}")
                print(f"• Optimizer: {updated_config['training']['optimizer']}")
                print(f"• Scheduler: {updated_config['training']['scheduler']}")
                print(f"• Layers: {', '.join(updated_config['layers'])}")
            else:
                print("❌ Gagal menyimpan konfigurasi")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        # Re-enable tombol
        ui_components['save_config_button'].disabled = False

def on_layer_selection_change(change, ui_components):
    """
    Handler untuk perubahan seleksi layer.
    
    Args:
        change: Change event
        ui_components: Dictionary berisi komponen UI
    """
    selected_layers = change['new']
    if selected_layers:
        update_layer_info(ui_components, selected_layers)
    else:
        ui_components['layer_info'].value = """
        <div style="margin: 10px 0; padding: 10px; background-color: #ffebee; border-radius: 5px;">
            <p><b>⚠️ Peringatan:</b> Tidak ada layer yang dipilih</p>
        </div>
        """

def setup_global_config_handlers(ui_components, config, config_path, logger):
    """
    Setup event handlers untuk komponen UI konfigurasi global.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        config: Dictionary konfigurasi
        config_path: Path ke file konfigurasi
        logger: Logger untuk mencatat aktivitas
        
    Returns:
        Dictionary berisi komponen UI yang telah disetup handler-nya
    """
    # Setup handler untuk tombol save
    ui_components['save_config_button'].on_click(
        lambda b: on_save_config_button_clicked(b, ui_components, config, config_path, logger)
    )
    
    # Setup handler untuk perubahan layer selection
    ui_components['layer_selection'].observe(
        lambda change: on_layer_selection_change(change, ui_components),
        names='value'
    )
    
    # Update tampilan awal
    update_layer_info(ui_components, config['layers'])
    
    return ui_components

def setup_pipeline_config_handlers(ui_components, config, logger, from_google_colab=False):
    """
    Setup event handlers untuk komponen UI konfigurasi pipeline.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        config: Dictionary konfigurasi
        logger: Logger untuk mencatat aktivitas
        from_google_colab: Boolean yang menunjukkan apakah berjalan dari Google Colab
        
    Returns:
        Dictionary berisi komponen UI yang telah disetup handler-nya
    """
    # Handler untuk tombol save
    ui_components['save_config_button'].on_click(
        lambda b: on_pipeline_save_button_clicked(b, ui_components, config, logger, from_google_colab)
    )
    
    # Handler untuk tombol reload
    ui_components['reload_config_button'].on_click(
        lambda b: on_pipeline_reload_button_clicked(b, ui_components, config, logger)
    )
    
    # Handler untuk perubahan data source
    ui_components['data_source_radio'].observe(
        lambda change: on_data_source_change(change, ui_components),
        names='value'
    )
    
    # Setup initial state
    if from_google_colab:
        try:
            from google.colab import userdata
            api_key = userdata.get('ROBOFLOW_API_KEY')
            if api_key and 'data' not in config:
                config['data'] = {}
            if api_key and 'roboflow' not in config['data']:
                config['data']['roboflow'] = {}
            if api_key:
                config['data']['roboflow']['api_key'] = api_key
        except:
            pass
    
    # Initial UI update based on data source
    on_data_source_change({'new': ui_components['data_source_radio'].value}, ui_components)
    
    return ui_components

def on_pipeline_save_button_clicked(b, ui_components, config, logger, from_google_colab=False):
    """
    Handler untuk tombol simpan konfigurasi pipeline.
    
    Args:
        b: Button instance
        ui_components: Dictionary berisi komponen UI
        config: Dictionary konfigurasi
        logger: Logger untuk mencatat aktivitas
        from_google_colab: Boolean yang menunjukkan apakah berjalan dari Google Colab
    """
    # Disable tombol selama proses
    ui_components['save_config_button'].disabled = True
    
    with ui_components['output_area']:
        clear_output()
        
        try:
            # Update konfigurasi dari UI
            if 'model' not in config:
                config['model'] = {}
            
            config['model']['backbone'] = ui_components['backbone_dropdown'].value
            
            if 'training' not in config:
                config['training'] = {}
                
            config['training']['batch_size'] = ui_components['batch_size_slider'].value
            config['training']['epochs'] = ui_components['epochs_slider'].value
            config['training']['learning_rate'] = ui_components['lr_dropdown'].value
            
            config['detection_mode'] = ui_components['detection_mode_radio'].value
            
            # Update data source configurations
            if 'data' not in config:
                config['data'] = {}
                
            config['data']['source'] = ui_components['data_source_radio'].value
            
            if config['data']['source'] == 'roboflow':
                if 'roboflow' not in config['data']:
                    config['data']['roboflow'] = {}
                
                config['data']['roboflow']['workspace'] = ui_components['workspace_input'].value
                config['data']['roboflow']['project'] = ui_components['project_input'].value
                config['data']['roboflow']['version'] = ui_components['version_input'].value
                
                # Try to get API key from Google Colab secrets
                if from_google_colab:
                    try:
                        from google.colab import userdata
                        api_key = userdata.get('ROBOFLOW_API_KEY')
                        if api_key:
                            config['data']['roboflow']['api_key'] = api_key
                    except:
                        pass
            
            # Setup correct layers based on detection mode
            if config['detection_mode'] == 'single':
                config['layers'] = ['banknote']
            else:
                config['layers'] = ['banknote', 'nominal', 'security']
            
            # Simpan konfigurasi
            config_path = 'configs/experiment_config.yaml'
            os.makedirs('configs', exist_ok=True)
            
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            # Buat backup pickle untuk digunakan di cells lain
            with open('config.pkl', 'wb') as f:
                pickle.dump(config, f)
                
            logger.info(f"📝 Konfigurasi pipeline disimpan di {config_path}")
            print(f"✅ Konfigurasi pipeline berhasil disimpan ke {config_path}")
            
            # Tampilkan ringkasan konfigurasi
            print("\n📋 Ringkasan konfigurasi:")
            print(f"• Backbone: {config['model']['backbone']}")
            print(f"• Batch Size: {config['training']['batch_size']}")
            print(f"• Epochs: {config['training']['epochs']}")
            print(f"• Learning Rate: {config['training']['learning_rate']}")
            print(f"• Detection Mode: {config['detection_mode']}")
            print(f"• Data Source: {config['data']['source']}")
            
            if config['data']['source'] == 'roboflow':
                print(f"• Roboflow Workspace: {config['data']['roboflow']['workspace']}")
                print(f"• Roboflow Project: {config['data']['roboflow']['project']}")
                print(f"• Roboflow Version: {config['data']['roboflow']['version']}")
                if 'api_key' in config['data']['roboflow']:
                    print(f"• Roboflow API Key: {'*'*len(config['data']['roboflow']['api_key'])}")
            
            print(f"• Layers: {', '.join(config['layers'])}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            logger.error(f"Error saat menyimpan konfigurasi pipeline: {str(e)}")
        
        # Re-enable tombol
        ui_components['save_config_button'].disabled = False

def on_pipeline_reload_button_clicked(b, ui_components, config, logger):
    """
    Handler untuk tombol reload konfigurasi pipeline.
    
    Args:
        b: Button instance
        ui_components: Dictionary berisi komponen UI
        config: Dictionary konfigurasi
        logger: Logger untuk mencatat aktivitas
    """
    with ui_components['output_area']:
        clear_output()
        
        try:
            # Coba load dari file config
            config_path = 'configs/experiment_config.yaml'
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    
                # Update UI dari loaded config
                if 'model' in loaded_config and 'backbone' in loaded_config['model']:
                    ui_components['backbone_dropdown'].value = loaded_config['model']['backbone']
                
                if 'training' in loaded_config:
                    if 'batch_size' in loaded_config['training']:
                        ui_components['batch_size_slider'].value = loaded_config['training']['batch_size']
                    if 'epochs' in loaded_config['training']:
                        ui_components['epochs_slider'].value = loaded_config['training']['epochs']
                    if 'learning_rate' in loaded_config['training']:
                        ui_components['lr_dropdown'].value = loaded_config['training']['learning_rate']
                
                if 'detection_mode' in loaded_config:
                    ui_components['detection_mode_radio'].value = loaded_config['detection_mode']
                
                if 'data' in loaded_config:
                    if 'source' in loaded_config['data']:
                        ui_components['data_source_radio'].value = loaded_config['data']['source']
                    
                    if 'roboflow' in loaded_config['data']:
                        if 'workspace' in loaded_config['data']['roboflow']:
                            ui_components['workspace_input'].value = loaded_config['data']['roboflow']['workspace']
                        if 'project' in loaded_config['data']['roboflow']:
                            ui_components['project_input'].value = loaded_config['data']['roboflow']['project']
                        if 'version' in loaded_config['data']['roboflow']:
                            ui_components['version_input'].value = loaded_config['data']['roboflow']['version']
                
                # Update config dengan loaded_config
                for key, value in loaded_config.items():
                    config[key] = value
                
                print(f"✅ Konfigurasi berhasil dimuat dari {config_path}")
                logger.info(f"Konfigurasi dimuat dari {config_path}")
            else:
                print(f"⚠️ File konfigurasi {config_path} tidak ditemukan")
                logger.warning(f"File konfigurasi {config_path} tidak ditemukan")
        except Exception as e:
            print(f"❌ Error saat memuat konfigurasi: {str(e)}")
            logger.error(f"Error saat memuat konfigurasi: {str(e)}")

def on_data_source_change(change, ui_components):
    """
    Handler untuk perubahan sumber data.
    
    Args:
        change: Change event
        ui_components: Dictionary berisi komponen UI
    """
    if change['new'] == 'roboflow':
        # Enable Roboflow inputs
        ui_components['workspace_input'].disabled = False
        ui_components['project_input'].disabled = False
        ui_components['version_input'].disabled = False
    else:
        # Disable Roboflow inputs
        ui_components['workspace_input'].disabled = True
        ui_components['project_input'].disabled = True
        ui_components['version_input'].disabled = True