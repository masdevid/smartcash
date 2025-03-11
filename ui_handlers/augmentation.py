"""
File: smartcash/ui_handlers/augmentation.py
Author: Alfrida Sabar
Deskripsi: Handler untuk UI augmentasi dataset SmartCash.
"""

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import time

from smartcash.utils.ui_utils import create_status_indicator

def setup_augmentation_handlers(ui_components, config=None):
    """Setup minimal handlers untuk UI augmentasi dataset."""
    # Inisialisasi dependencies jika tersedia
    dataset_manager, logger, observer_manager = None, None, None
    try:
        from smartcash.utils.logger import get_logger
        from smartcash.handlers.dataset import DatasetManager
        from smartcash.utils.observer import EventDispatcher, EventTopics
        from smartcash.utils.observer.observer_manager import ObserverManager
        
        logger = get_logger("augmentation")
        dataset_manager = DatasetManager(config, logger=logger)
        observer_manager = ObserverManager(auto_register=True)
    except ImportError as e:
        print(f"ℹ️ Menggunakan simulasi: {str(e)}")
    
    # Kelompok observer untuk augmentasi
    augmentation_observers_group = "augmentation_observers"
    
    # Fungsi untuk update progress UI
    def update_progress_callback(event_type, sender, progress=0, total=100, message=None, **kwargs):
        # Update progress bar
        ui_components['augmentation_progress'].value = int(progress * 100 / total) if total > 0 else 0
        ui_components['augmentation_progress'].description = f"{int(progress * 100 / total)}%" if total > 0 else "0%"
        
        # Display message jika ada
        if message and ui_components['augmentation_status']:
            with ui_components['augmentation_status']:
                display(create_status_indicator("info", message))
    
    # Setup observer untuk progress jika observer_manager tersedia
    if observer_manager:
        try:
            # Unregister any existing observers in this group first
            observer_manager.unregister_group(augmentation_observers_group)
            
            # Buat progress observer
            observer_manager.create_simple_observer(
                event_type=EventTopics.AUGMENTATION_PROGRESS,
                callback=update_progress_callback,
                name="AugmentationProgressObserver",
                group=augmentation_observers_group
            )
            
            # Buat logger observer untuk event augmentasi
            observer_manager.create_logging_observer(
                event_types=[
                    EventTopics.AUGMENTATION_START,
                    EventTopics.AUGMENTATION_END,
                    EventTopics.AUGMENTATION_ERROR
                ],
                log_level="info",
                name="AugmentationLoggerObserver",
                format_string="{event_type}: {message}",
                include_timestamp=True,
                logger_name="augmentation",
                group=augmentation_observers_group
            )
            
            if logger:
                logger.info("✅ Observer untuk augmentasi telah dikonfigurasi")
                
        except Exception as e:
            if logger:
                logger.error(f"❌ Error saat setup observer: {str(e)}")
    
    # Handler untuk tombol augmentasi
    def on_augmentation_click(b):
        # Pastikan semua observer dari grup ini dihapus untuk mencegah memory leak
        if observer_manager:
            observer_manager.unregister_group(augmentation_observers_group)
            
            # Buat ulang observer untuk progress
            observer_manager.create_simple_observer(
                event_type=EventTopics.AUGMENTATION_PROGRESS,
                callback=update_progress_callback,
                name="AugmentationProgressObserver",
                group=augmentation_observers_group
            )
                
        with ui_components['augmentation_status']:
            clear_output()
            display(create_status_indicator("info", "🔄 Memulai augmentasi dataset..."))
            
            # Extract parameters from UI
            aug_types_widgets = ui_components['augmentation_options'].children[0].value
            variations = ui_components['augmentation_options'].children[1].value
            prefix = ui_components['augmentation_options'].children[2].value
            validate = ui_components['augmentation_options'].children[3].value
            resume = ui_components['augmentation_options'].children[4].value
            
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
                if observer_manager:
                    EventDispatcher.notify(
                        event_type=EventTopics.AUGMENTATION_START,
                        sender="augmentation_handler",
                        aug_types=aug_types,
                        variations=variations
                    )
                
                if dataset_manager:
                    # Collect params from UI
                    position_params = {
                        'rotation_prob': ui_components['position_options'].children[0].value,
                        'max_angle': ui_components['position_options'].children[1].value,
                        'flip_prob': ui_components['position_options'].children[2].value,
                        'scale_ratio': ui_components['position_options'].children[3].value
                    }
                    
                    lighting_params = {
                        'brightness_prob': ui_components['lighting_options'].children[0].value,
                        'brightness_limit': ui_components['lighting_options'].children[1].value,
                        'contrast_prob': ui_components['lighting_options'].children[2].value,
                        'contrast_limit': ui_components['lighting_options'].children[3].value
                    }
                    
                    # Run actual augmentation
                    display(create_status_indicator("info", f"⚙️ Membuat {variations} variasi untuk {len(aug_types)} tipe augmentasi"))
                    result = dataset_manager.augment_dataset(
                        aug_types=aug_types,
                        variations_per_type=variations,
                        output_prefix=prefix,
                        validate_results=validate,
                        resume=resume,
                        aug_params={'position': position_params, 'lighting': lighting_params}
                    )
                    
                    # Handle result
                    if result and result.get('status', '') == 'success':
                        total_images = result.get('augmentation_stats', {}).get('total_augmented', 0)
                        display(create_status_indicator("success", f"✅ Augmentasi selesai: {total_images} gambar baru dibuat"))
                        
                        # Display summary
                        stats = result.get('augmentation_stats', {})
                        if stats:
                            stats_html = f"""
                            <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 10px;">
                                <h4>📊 Augmentation Statistics</h4>
                                <ul>
                                    <li><b>Total images processed:</b> {stats.get('total_images', 0)}</li>
                                    <li><b>Total augmented images:</b> {stats.get('total_augmented', 0)}</li>
                                    <li><b>Augmentation types:</b> {', '.join(aug_types)}</li>
                                    <li><b>Duration:</b> {stats.get('duration', 0):.2f} seconds</li>
                                </ul>
                            </div>
                            """
                            display(HTML(stats_html))
                    else:
                        display(create_status_indicator("warning", f"⚠️ Augmentasi selesai dengan issues: {result.get('message', 'unknown error')}"))
                else:
                    # Simulate augmentation
                    simulate_augmentation(aug_types, variations, ui_components)
                    
                # Notification if observer available
                if observer_manager:
                    EventDispatcher.notify(
                        event_type=EventTopics.AUGMENTATION_END,
                        sender="augmentation_handler"
                    )
            except Exception as e:
                display(create_status_indicator("error", f"❌ Error: {str(e)}"))
                
                # Notification if observer available
                if observer_manager:
                    EventDispatcher.notify(
                        event_type=EventTopics.AUGMENTATION_ERROR,
                        sender="augmentation_handler",
                        error=str(e)
                    )
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
        display(create_status_indicator("success", f"✅ Augmentasi selesai: {total_images} gambar baru (simulasi)"))
        
        # Display summary
        summary_html = f"""
        <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 10px;">
            <h4>📊 Augmentation Statistics (Simulasi)</h4>
            <ul>
                <li><b>Total images processed:</b> 100</li>
                <li><b>Total augmented images:</b> {total_images}</li>
                <li><b>Augmentation types:</b> {', '.join(aug_types)}</li>
                <li><b>Duration:</b> {total_steps * 0.3:.2f} seconds</li>
            </ul>
        </div>
        """
        with ui['augmentation_status']:
            display(HTML(summary_html))
    
    # Fungsi cleanup untuk unregister observer
    def cleanup():
        if observer_manager:
            try:
                observer_manager.unregister_group(augmentation_observers_group)
                if logger:
                    logger.info("✅ Observer untuk augmentasi telah dibersihkan")
            except Exception as e:
                if logger:
                    logger.error(f"❌ Error saat membersihkan observer: {str(e)}")
    
    # Register handlers
    ui_components['augmentation_button'].on_click(on_augmentation_click)
    
    # Tambahkan fungsi cleanup ke komponen UI
    ui_components['cleanup'] = cleanup
    
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