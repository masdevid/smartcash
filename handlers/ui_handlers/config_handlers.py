"""
File: smartcash/handlers/ui_handlers/config_handlers.py
Author: Alfrida Sabar
Deskripsi: Handler untuk UI komponen konfigurasi, menangani penyimpanan dan update konfigurasi global.
          Refactored to use centralized ConfigManager.
"""

import os
from datetime import datetime
from pathlib import Path
from IPython.display import display, clear_output, HTML
import ipywidgets as widgets

# Import the ConfigManager
from smartcash.utils.config_manager import ConfigManager

def on_save_config_button_clicked(b, ui_components, config_manager, logger):
    """
    Handler untuk tombol simpan konfigurasi.
    
    Args:
        b: Button instance
        ui_components: Dictionary berisi komponen UI
        config_manager: Instance dari ConfigManager
        logger: Logger untuk mencatat aktivitas
    """
    # Validate required UI components
    required_components = ['save_config_button', 'output_area', 'backbone_dropdown',
                         'pretrained_checkbox', 'img_size_slider', 'batch_size_slider',
                         'workers_slider', 'lr_dropdown', 'epochs_slider',
                         'optimizer_dropdown', 'scheduler_dropdown', 'layer_selection']
    
    for component in required_components:
        if component not in ui_components:
            logger.error(f"❌ Required UI component '{component}' not found")
            return
    
    # Disable tombol selama proses
    ui_components['save_config_button'].disabled = True
    
    with ui_components['output_area']:
        clear_output()
        
        try:
            # Define UI to config mapping
            ui_mapping = {
                'backbone_dropdown': 'model.backbone',
                'pretrained_checkbox': 'model.pretrained',
                'batch_size_slider': 'model.batch_size',
                'workers_slider': 'model.workers',
                'lr_dropdown': 'training.learning_rate',
                'epochs_slider': 'training.epochs',
                'optimizer_dropdown': 'training.optimizer',
                'scheduler_dropdown': 'training.scheduler'
            }
            
            # Update config using the mapping
            for ui_key, config_path in ui_mapping.items():
                value = ui_components[ui_key].value
                config_manager.set(config_path, value)
            
            # Special handling for img_size which needs to be a list [640, 640]
            img_size_value = ui_components['img_size_slider'].value
            config_manager.set('model.img_size', [img_size_value, img_size_value])
            
            # Update layer configuration
            config_manager.set('layers', list(ui_components['layer_selection'].value))
            
            # Save the configuration
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"global_config_{timestamp}.yaml"
            success = config_manager.save(filename)
            
            if success:
                print(f"✅ Konfigurasi berhasil disimpan ke configs/{filename}")
                
                # Update tampilan layer info jika ada
                if 'layer_info' in ui_components:
                    update_layer_info(ui_components, config_manager.get('layers', []))
                
                # Tampilkan ringkasan konfigurasi
                print("\n📋 Ringkasan konfigurasi:")
                print(f"• Backbone: {config_manager.get('model.backbone')}")
                print(f"• Pretrained: {'Ya' if config_manager.get('model.pretrained') else 'Tidak'}")
                print(f"• Image Size: {config_manager.get('model.img_size')[0]}x{config_manager.get('model.img_size')[1]}")
                print(f"• Batch Size: {config_manager.get('model.batch_size')}")
                print(f"• Workers: {config_manager.get('model.workers')}")
                print(f"• Learning Rate: {config_manager.get('training.learning_rate')}")
                print(f"• Epochs: {config_manager.get('training.epochs')}")
                print(f"• Optimizer: {config_manager.get('training.optimizer')}")
                print(f"• Scheduler: {config_manager.get('training.scheduler')}")
                print(f"• Layers: {', '.join(config_manager.get('layers', []))}")
            else:
                print("❌ Gagal menyimpan konfigurasi")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        # Re-enable tombol
        ui_components['save_config_button'].disabled = False

def update_layer_info(ui_components, active_layers):
    """
    Update tampilan informasi layer aktif.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        active_layers: List layer yang aktif
    """
    if 'layer_info' not in ui_components:
        return
        
    ui_components['layer_info'].value = f"""
    <div style="margin: 10px 0; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
        <p><b>🔍 Layer yang diaktifkan:</b> {', '.join(active_layers)}</p>
    </div>
    """

def on_layer_selection_change(change, ui_components):
    """
    Handler untuk perubahan seleksi layer.
    
    Args:
        change: Change event
        ui_components: Dictionary berisi komponen UI
    """
    if 'layer_info' not in ui_components:
        return
        
    selected_layers = change['new']
    if selected_layers:
        update_layer_info(ui_components, selected_layers)
    else:
        ui_components['layer_info'].value = """
        <div style="margin: 10px 0; padding: 10px; background-color: #ffebee; border-radius: 5px;">
            <p><b>⚠️ Peringatan:</b> Tidak ada layer yang dipilih</p>
        </div>
        """

def on_pipeline_save_button_clicked(b, ui_components, config_manager, logger, from_google_colab=False):
    """
    Handler untuk tombol simpan konfigurasi pipeline.
    
    Args:
        b: Button instance
        ui_components: Dictionary berisi komponen UI
        config_manager: Instance dari ConfigManager
        logger: Logger untuk mencatat aktivitas
        from_google_colab: Boolean yang menunjukkan apakah berjalan dari Google Colab
    """
    # Validate required UI components
    required_components = ['save_config_button', 'output_area', 'backbone_dropdown',
                          'batch_size_slider', 'epochs_slider', 'lr_dropdown',
                          'detection_mode_radio', 'data_source_radio']
    
    for component in required_components:
        if component not in ui_components:
            logger.error(f"❌ Required UI component '{component}' not found")
            return
    
    # Disable tombol selama proses
    ui_components['save_config_button'].disabled = True
    
    with ui_components['output_area']:
        clear_output()
        
        try:
            # Update configuration using ConfigManager
            config_manager.set('model.backbone', ui_components['backbone_dropdown'].value)
            config_manager.set('training.batch_size', ui_components['batch_size_slider'].value)
            config_manager.set('training.epochs', ui_components['epochs_slider'].value)
            config_manager.set('training.learning_rate', ui_components['lr_dropdown'].value)
            config_manager.set('detection_mode', ui_components['detection_mode_radio'].value)
            config_manager.set('data.source', ui_components['data_source_radio'].value)
            
            # Handle Roboflow settings if data source is 'roboflow'
            if ui_components['data_source_radio'].value == 'roboflow':
                if 'workspace_input' in ui_components:
                    config_manager.set('data.roboflow.workspace', ui_components['workspace_input'].value)
                if 'project_input' in ui_components:
                    config_manager.set('data.roboflow.project', ui_components['project_input'].value)
                if 'version_input' in ui_components:
                    config_manager.set('data.roboflow.version', ui_components['version_input'].value)
                
                # Try to get API key from Google Colab secrets
                if from_google_colab:
                    try:
                        from google.colab import userdata
                        api_key = userdata.get('ROBOFLOW_API_KEY')
                        if api_key:
                            config_manager.set('data.roboflow.api_key', api_key)
                    except:
                        pass
            
            # Setup correct layers based on detection mode
            if ui_components['detection_mode_radio'].value == 'single':
                config_manager.set('layers', ['banknote'])
            else:
                config_manager.set('layers', ['banknote', 'nominal', 'security'])
            
            # Save the configuration
            success = config_manager.save('experiment_config.yaml')
                
            if success:
                print(f"✅ Konfigurasi pipeline berhasil disimpan")
                
                # Display configuration summary
                print("\n📋 Ringkasan konfigurasi:")
                print(f"• Backbone: {config_manager.get('model.backbone')}")
                print(f"• Batch Size: {config_manager.get('training.batch_size')}")
                print(f"• Epochs: {config_manager.get('training.epochs')}")
                print(f"• Learning Rate: {config_manager.get('training.learning_rate')}")
                print(f"• Detection Mode: {config_manager.get('detection_mode')}")
                print(f"• Data Source: {config_manager.get('data.source')}")
                
                if config_manager.get('data.source') == 'roboflow':
                    print(f"• Roboflow Workspace: {config_manager.get('data.roboflow.workspace')}")
                    print(f"• Roboflow Project: {config_manager.get('data.roboflow.project')}")
                    print(f"• Roboflow Version: {config_manager.get('data.roboflow.version')}")
                    if config_manager.get('data.roboflow.api_key'):
                        print(f"• Roboflow API Key: {'*'*len(config_manager.get('data.roboflow.api_key'))}")
                
                print(f"• Layers: {', '.join(config_manager.get('layers', []))}")
            else:
                print("❌ Gagal menyimpan konfigurasi pipeline")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            logger.error(f"Error saat menyimpan konfigurasi pipeline: {str(e)}")
        
        # Re-enable tombol
        ui_components['save_config_button'].disabled = False

def on_pipeline_reload_button_clicked(b, ui_components, config_manager, logger):
    """
    Handler untuk tombol reload konfigurasi pipeline.
    
    Args:
        b: Button instance
        ui_components: Dictionary berisi komponen UI
        config_manager: Instance dari ConfigManager
        logger: Logger untuk mencatat aktivitas
    """
    with ui_components['output_area']:
        clear_output()
        
        try:
            # Reload configuration using ConfigManager
            config = config_manager.load_config('experiment_config.yaml')
            
            # Update UI from loaded config
            if config:
                # Define config paths and corresponding UI components
                config_ui_mapping = {
                    'model.backbone': 'backbone_dropdown',
                    'training.batch_size': 'batch_size_slider',
                    'training.epochs': 'epochs_slider',
                    'training.learning_rate': 'lr_dropdown',
                    'detection_mode': 'detection_mode_radio',
                    'data.source': 'data_source_radio',
                    'data.roboflow.workspace': 'workspace_input',
                    'data.roboflow.project': 'project_input',
                    'data.roboflow.version': 'version_input'
                }
                
                # Update UI using the mapping
                for config_path, ui_key in config_ui_mapping.items():
                    if ui_key in ui_components:
                        value = config_manager.get(config_path)
                        if value is not None:
                            ui_components[ui_key].value = value
                
                print(f"✅ Konfigurasi berhasil dimuat dari experiment_config.yaml")
                logger.info(f"Konfigurasi dimuat dari experiment_config.yaml")
            else:
                print(f"⚠️ File konfigurasi experiment_config.yaml tidak ditemukan")
                logger.warning(f"File konfigurasi experiment_config.yaml tidak ditemukan")
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
    # Validate UI components
    required_components = ['workspace_input', 'project_input', 'version_input']
    for component in required_components:
        if component not in ui_components:
            return
            
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

def setup_global_config_handlers(ui_components, config, logger):
    """
    Setup event handlers untuk komponen UI konfigurasi global.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        config: Dictionary konfigurasi atau ConfigManager instance
        logger: Logger untuk mencatat aktivitas
        
    Returns:
        Dictionary berisi komponen UI yang telah disetup handler-nya
    """
    # Validate UI components
    required_components = ['save_config_button', 'layer_selection']
    for component in required_components:
        if component not in ui_components:
            logger.error(f"❌ Required UI component '{component}' not found")
            return ui_components
    
    # Create ConfigManager if needed
    if not isinstance(config, ConfigManager):
        config_manager = ConfigManager(logger=logger)
        config_manager.config = config  # Use the provided config dictionary
    else:
        config_manager = config
    
    # Setup handler untuk tombol save
    ui_components['save_config_button'].on_click(
        lambda b: on_save_config_button_clicked(b, ui_components, config_manager, logger)
    )
    
    # Setup handler untuk perubahan layer selection
    ui_components['layer_selection'].observe(
        lambda change: on_layer_selection_change(change, ui_components),
        names='value'
    )
    
    # Update tampilan awal
    update_layer_info(ui_components, config_manager.get('layers', []))
    
    # Store config_manager in ui_components for later use
    ui_components['_config_manager'] = config_manager
    
    return ui_components

def setup_pipeline_config_handlers(ui_components, config, logger, from_google_colab=False):
    """
    Setup event handlers untuk komponen UI konfigurasi pipeline.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        config: Dictionary konfigurasi atau ConfigManager instance
        logger: Logger untuk mencatat aktivitas
        from_google_colab: Boolean yang menunjukkan apakah berjalan dari Google Colab
        
    Returns:
        Dictionary berisi komponen UI yang telah disetup handler-nya
    """
    # Validate UI components
    required_components = ['save_config_button', 'reload_config_button', 'data_source_radio']
    for component in required_components:
        if component not in ui_components:
            logger.error(f"❌ Required UI component '{component}' not found")
            return ui_components
    
    # Create ConfigManager if needed
    if not isinstance(config, ConfigManager):
        config_manager = ConfigManager(logger=logger)
        config_manager.config = config  # Use the provided config dictionary
    else:
        config_manager = config
    
    # Handler untuk tombol save
    ui_components['save_config_button'].on_click(
        lambda b: on_pipeline_save_button_clicked(b, ui_components, config_manager, logger, from_google_colab)
    )
    
    # Handler untuk tombol reload
    ui_components['reload_config_button'].on_click(
        lambda b: on_pipeline_reload_button_clicked(b, ui_components, config_manager, logger)
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
            if api_key:
                config_manager.set('data.roboflow.api_key', api_key)
        except:
            pass
    
    # Initial UI update based on data source
    on_data_source_change({'new': ui_components['data_source_radio'].value}, ui_components)
    
    # Store config_manager in ui_components for later use
    ui_components['_config_manager'] = config_manager
    
    return ui_components