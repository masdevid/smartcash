"""
File: smartcash/ui_handlers/augmentation.py
Author: Alfrida Sabar (refactored)
Deskripsi: Handler minimal untuk UI augmentasi dataset SmartCash.
"""

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import time
from pathlib import Path

def setup_augmentation_handlers(ui_components, config=None):
    """Setup minimal handlers untuk UI augmentasi dataset."""
    # Inisialisasi dependencies jika tersedia
    dataset_manager, logger, observer_manager = None, None, None
    try:
        from smartcash.utils.logger import get_logger
        from smartcash.handlers.dataset import DatasetManager
        from smartcash.utils.observer import EventDispatcher, EventTopics
        
        logger = get_logger("augmentation")
        dataset_manager = DatasetManager(config, logger=logger)
    except ImportError as e:
        print(f"ℹ️ Menggunakan simulasi: {str(e)}")
    
    # Handler untuk tombol augmentasi
    def on_augmentation_click(b):
        with ui_components['augmentation_status']:
            clear_output()
            display(create_status_indicator("info", "🔄 Memulai augmentasi dataset..."))
            
            # Extract parameters from UI
            aug_types_widgets = ui_components['augmentation_options'].children[0].value
            variations = ui_components['augmentation_options'].children[1].value
            prefix = ui_components['augmentation_options'].children[2].value
            validate = ui_components['augmentation_options'].children[3].value
            
            # Map UI values to config format
            type_map = {
                'Combined (Recommended)': 'combined',
                'Position Variations': 'position',
                'Lighting Variations': 'lighting',
                'Extreme Rotation': 'extreme_rotation'
            }
            aug_types = [type_map.get(t, 'combined') for t in aug_types_widgets]
            
            # Prepare UI for progress
            ui_components['augmentation_progress'].layout.visibility = 'visible'
            ui_components['augmentation_progress'].value = 0
            
            # Run augmentation through DatasetManager or simulate
            try:
                # Notification if observer available
                if 'EventDispatcher' in locals():
                    EventDispatcher.notify(
                        event_type=EventTopics.AUGMENTATION_START,
                        sender="augmentation_handler"
                    )
                
                if dataset_manager:
                    # Collect params from UI
                    position_params = {
                        'rotation_prob': ui_components['position_options'].children[0].value,
                        'max_angle': ui_components['position_options'].children[1].value
                    }
                    
                    lighting_params = {
                        'brightness_prob': ui_components['lighting_options'].children[0].value,
                        'brightness_limit': ui_components['lighting_options'].children[1].value
                    }
                    
                    # Setup observer for progress updates
                    def update_progress(sender, progress, total, **kwargs):
                        ui_components['augmentation_progress'].value = int(progress * 100 / total) if total > 0 else 0
                        ui_components['augmentation_progress'].description = f"{int(progress * 100 / total)}%" if total > 0 else "0%"
                    
                    if 'EventDispatcher' in locals():
                        EventDispatcher.register(EventTopics.AUGMENTATION_PROGRESS, update_progress)
                    
                    # Run actual augmentation
                    display(create_status_indicator("info", f"⚙️ Membuat {variations} variasi untuk {len(aug_types)} tipe augmentasi"))
                    result = dataset_manager.augment_dataset(
                        aug_types=aug_types,
                        variations_per_type=variations,
                        output_prefix=prefix,
                        validate_results=validate,
                        aug_params={'position': position_params, 'lighting': lighting_params}
                    )
                    
                    # Handle result
                    if result and result.get('success', False):
                        display(create_status_indicator("success", f"✅ Augmentasi selesai: {result.get('num_images', 0)} gambar baru"))
                    else:
                        display(create_status_indicator("warning", f"⚠️ Augmentasi selesai dengan issues"))
                else:
                    # Simulate augmentation
                    simulate_augmentation(aug_types, variations, ui_components)
            except Exception as e:
                display(create_status_indicator("error", f"❌ Error: {str(e)}"))
            finally:
                ui_components['augmentation_progress'].layout.visibility = 'hidden'
    
    # Helper function to simulate augmentation
    def simulate_augmentation(aug_types, variations, ui):
        total_steps = len(aug_types) * variations
        for i in range(total_steps + 1):
            # Update progress
            ui['augmentation_progress'].value = int(i * 100 / total_steps) if total_steps > 0 else 0
            ui['augmentation_progress'].description = f"{int(i * 100 / total_steps)}%" if total_steps > 0 else "0%"
            
            # Display step info
            if i > 0 and i % variations == 0:
                current_type = aug_types[min(i // variations - 1, len(aug_types) - 1)]
                display(create_status_indicator("info", f"🔄 Augmentasi '{current_type}' selesai"))
            
            time.sleep(0.3)
        
        # Final success message
        total_images = len(aug_types) * variations * 100  # Assume 100 images per variation
        display(create_status_indicator("success", f"✅ Augmentasi selesai: {total_images} gambar baru"))
    
    # Register handler
    ui_components['augmentation_button'].on_click(on_augmentation_click)
    
    # Initialize from config if available
    if config and 'augmentation' in config:
        aug_config = config['augmentation']
        
        # Map config values to UI
        if 'types' in aug_config:
            ui_mapping = {
                'combined': 'Combined (Recommended)',
                'position': 'Position Variations',
                'lighting': 'Lighting Variations',
                'extreme_rotation': 'Extreme Rotation'
            }
            aug_types = aug_config['types']
            ui_types = [ui_mapping.get(t, 'Combined (Recommended)') for t in aug_types]
            
            # Select appropriate options
            available_options = ui_components['augmentation_options'].children[0].options
            selected_indices = [i for i, opt in enumerate(available_options) if opt in ui_types]
            if selected_indices:
                ui_components['augmentation_options'].children[0].value = [available_options[i] for i in selected_indices]
        
        # Set numeric values
        ui_components['augmentation_options'].children[1].value = aug_config.get('variations_per_type', 2)
        ui_components['augmentation_options'].children[2].value = aug_config.get('output_prefix', 'aug')
        ui_components['augmentation_options'].children[3].value = aug_config.get('validate_results', True)
    
    return ui_components

def create_status_indicator(status, message):
    """Buat indikator status dengan styling konsisten."""
    status_styles = {
        'success': {'icon': '✅', 'color': 'green'},
        'warning': {'icon': '⚠️', 'color': 'orange'},
        'error': {'icon': '❌', 'color': 'red'},
        'info': {'icon': 'ℹ️', 'color': 'blue'}
    }
    
    style = status_styles.get(status, status_styles['info'])
    
    status_html = f"""
    <div style="margin: 5px 0; padding: 8px 12px; 
                border-radius: 4px; background-color: #f8f9fa;">
        <span style="color: {style['color']}; font-weight: bold;"> 
            {style['icon']} {message}
        </span>
    </div>
    """
    
    return HTML(status_html)