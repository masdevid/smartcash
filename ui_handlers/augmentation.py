"""
File: smartcash/ui_handlers/augmentation.py
Author: Refactored
Deskripsi: Handler untuk UI augmentasi dataset SmartCash dengan pendekatan DRY.
"""

import threading
import time
import shutil
import json
import yaml
from pathlib import Path
from IPython.display import display, clear_output, HTML

from smartcash.utils.ui_utils import create_status_indicator

def setup_augmentation_handlers(ui_components, config=None):
    """Setup handlers untuk UI augmentasi dataset dengan pendekatan DRY."""
    # Inisialisasi dependencies
    logger = None
    dataset_manager = None
    observer_manager = None
    augmentation_manager = None
    env_manager = None
    config_manager = None
    
    try:
        from smartcash.utils.logger import get_logger
        from smartcash.handlers.dataset import DatasetManager
        from smartcash.utils.observer import EventDispatcher, EventTopics
        from smartcash.utils.observer.observer_manager import ObserverManager
        from smartcash.utils.augmentation import AugmentationManager
        from smartcash.utils.environment_manager import EnvironmentManager
        from smartcash.utils.config_manager import ConfigManager
        
        logger = get_logger("augmentation")
        env_manager = EnvironmentManager()
        config_manager = ConfigManager.get_instance(logger=logger)
        
        # Load config jika belum ada
        if not config or not isinstance(config, dict) or 'augmentation' not in config:
            config = config_manager.load_config(
                filename="configs/augmentation_config.yaml",
                fallback_to_pickle=True
            ) or {
                'augmentation': {
                    'enabled': True,
                    'num_variations': 2,
                    'output_prefix': 'aug',
                    'types': ['combined', 'position', 'lighting'],
                    'position': {
                        'fliplr': 0.5,
                        'degrees': 15,
                        'translate': 0.1,
                        'scale': 0.1
                    },
                    'lighting': {
                        'hsv_h': 0.015,
                        'hsv_s': 0.7,
                        'hsv_v': 0.4,
                        'contrast': 0.3,
                        'brightness': 0.3
                    }
                }
            }
            
            # Simpan config baru jika belum ada
            config_manager.save_config(config, "configs/augmentation_config.yaml")
        
        # Inisialisasi Manager
        dataset_manager = DatasetManager(config, logger=logger)
        observer_manager = ObserverManager(auto_register=True)
        
        # Inisialisasi AugmentationManager (lazy loading)
        output_dir = config.get('augmentation', {}).get('output_dir', 'data/augmented')
        augmentation_manager = AugmentationManager(
            config=config,
            output_dir=output_dir,
            logger=logger,
            num_workers=config.get('model', {}).get('workers', 4)
        )
        
    except ImportError as e:
        if logger:
            logger.warning(f"⚠️ Beberapa modul tidak tersedia: {str(e)}")
        else:
            print(f"⚠️ Beberapa modul tidak tersedia: {str(e)}")
    
    # Kelompok observer untuk augmentasi
    observer_group = "augmentation_ui_observers"
    
    # Pastikan observer lama dibersihkan
    if observer_manager:
        observer_manager.unregister_group(observer_group)
    
    # Fungsi untuk update progress UI
    def update_progress_callback(event_type, sender, progress=0, total=100, message=None, **kwargs):
        # Update progress bar
        ui_components['augmentation_progress'].value = int(progress * 100 / total) if total > 0 else 0
        ui_components['augmentation_progress'].description = f"{int(progress * 100 / total)}%" if total > 0 else "0%"
        
        # Display message jika ada
        if message and message.strip():
            with ui_components['augmentation_status']:
                display(create_status_indicator("info", message))
    
    # Register observer untuk progress jika tersedia
    if observer_manager:
        observer_manager.create_simple_observer(
            event_type=EventTopics.AUGMENTATION_PROGRESS,
            callback=update_progress_callback,
            name="AugmentationProgressObserver",
            group=observer_group
        )
        
        # Observer untuk event augmentasi lainnya
        observer_manager.create_logging_observer(
            event_types=[
                EventTopics.AUGMENTATION_START,
                EventTopics.AUGMENTATION_END,
                EventTopics.AUGMENTATION_ERROR
            ],
            log_level="info",
            name="AugmentationEventObserver",
            group=observer_group
        )
    
    # Handler untuk run augmentation
    def on_augmentation_click(b):
        # Disable buttons during processing
        b.disabled = True
        ui_components['reset_button'].disabled = True
        ui_components['cleanup_button'].disabled = True
        ui_components['save_config_button'].disabled = True
        
        # Show progress bar
        ui_components['augmentation_progress'].layout.visibility = 'visible'
        ui_components['augmentation_progress'].value = 0
        
        # Clear previous status
        with ui_components['augmentation_status']:
            clear_output()
            display(create_status_indicator("info", "🔄 Memulai augmentasi dataset..."))
        
        # Extract augmentation parameters
        aug_types_widgets = ui_components['augmentation_options'].children[0].value
        variations = ui_components['augmentation_options'].children[1].value
        prefix = ui_components['augmentation_options'].children[2].value
        validate = ui_components['augmentation_options'].children[3].value
        resume = ui_components['augmentation_options'].children[4].value
        
        # Map UI types to config format
        type_map = {
            'Combined (Recommended)': 'combined',
            'Position Variations': 'position',
            'Lighting Variations': 'lighting',
            'Extreme Rotation': 'extreme_rotation'
        }
        aug_types = [type_map.get(t, 'combined') for t in aug_types_widgets]
        
        # Collect parameters from UI
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
        
        # Update config with new values
        if config:
            config['augmentation'] = config.get('augmentation', {})
            config['augmentation'].update({
                'types': aug_types,
                'num_variations': variations,
                'output_prefix': prefix,
                'validate_results': validate,
                'resume': resume,
                'position': position_params,
                'lighting': lighting_params
            })
        
        # Run augmentation in a separate thread
        def run_augmentation():
            try:
                # Notify start if observer available
                if observer_manager:
                    EventDispatcher.notify(
                        event_type=EventTopics.AUGMENTATION_START,
                        sender="augmentation_handler",
                        message="Memulai augmentasi dataset",
                        aug_types=aug_types,
                        variations=variations
                    )
                
                result = None
                
                # Run actual augmentation if manager available
                if augmentation_manager:
                    with ui_components['augmentation_status']:
                        display(create_status_indicator(
                            "info", 
                            f"⚙️ Membuat {variations} variasi untuk {len(aug_types)} tipe augmentasi"
                        ))
                    
                    # Execute augmentation
                    result = augmentation_manager.augment_dataset(
                        split='train',  # Augmentasi hanya untuk train split
                        augmentation_types=aug_types,
                        num_variations=variations,
                        output_prefix=prefix,
                        validate_results=validate,
                        resume=resume
                    )
                    
                # Process result and update UI
                with ui_components['augmentation_status']:
                    if result:
                        if isinstance(result, dict) and result.get('augmented', 0) > 0:
                            # Success
                            display(create_status_indicator(
                                "success", 
                                f"✅ Augmentasi selesai: {result.get('augmented', 0)} gambar baru dibuat"
                            ))
                            
                            # Display summary
                            display_stats = {
                                'total_images': result.get('total_images', 0),
                                'augmented': result.get('augmented', 0),
                                'failed': result.get('failed', 0),
                                'duration': f"{result.get('duration', 0):.2f} detik",
                                'types': ", ".join(aug_types)
                            }
                            
                            # Display results
                            with ui_components['results_display']:
                                clear_output()
                                html = f"""
                                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 10px;">
                                    <h4>📊 Augmentation Results</h4>
                                    <ul>
                                        <li><b>Total images processed:</b> {display_stats['total_images']}</li>
                                        <li><b>Total augmented images:</b> {display_stats['augmented']}</li>
                                        <li><b>Failed augmentations:</b> {display_stats['failed']}</li>
                                        <li><b>Augmentation types:</b> {display_stats['types']}</li>
                                        <li><b>Duration:</b> {display_stats['duration']}</li>
                                    </ul>
                                </div>
                                """
                                display(HTML(html))
                            
                            # Make results visible
                            ui_components['results_display'].layout.display = 'block'
                            
                        else:
                            display(create_status_indicator(
                                "warning", 
                                f"⚠️ Augmentasi selesai dengan masalah: {str(result)}"
                            ))
                    else:
                        display(create_status_indicator(
                            "error", 
                            "❌ AugmentationManager tidak tersedia atau augmentasi gagal"
                        ))
                
                # Notify completion if observer available
                if observer_manager:
                    EventDispatcher.notify(
                        event_type=EventTopics.AUGMENTATION_END,
                        sender="augmentation_handler",
                        message="Augmentasi dataset selesai",
                        result=result
                    )
            
            except Exception as e:
                # Handle errors
                with ui_components['augmentation_status']:
                    display(create_status_indicator("error", f"❌ Error: {str(e)}"))
                
                # Notify error if observer available
                if observer_manager:
                    EventDispatcher.notify(
                        event_type=EventTopics.AUGMENTATION_ERROR,
                        sender="augmentation_handler",
                        message=f"Error saat augmentasi: {str(e)}",
                        error=str(e)
                    )
            
            finally:
                # Hide progress bar and re-enable buttons
                ui_components['augmentation_progress'].layout.visibility = 'hidden'
                ui_components['augmentation_button'].disabled = False
                ui_components['reset_button'].disabled = False
                ui_components['cleanup_button'].disabled = False
                ui_components['save_config_button'].disabled = False
        
        # Start augmentation thread
        threading.Thread(target=run_augmentation, daemon=True).start()
    
    # Handler untuk reset settings
    def on_reset_settings(b):
        with ui_components['augmentation_status']:
            clear_output()
            display(create_status_indicator("info", "🔄 Reset pengaturan augmentasi ke default..."))
            
            # Default values
            default_config = {
                'augmentation': {
                    'types': ['combined'],
                    'num_variations': 2,
                    'output_prefix': 'aug',
                    'validate_results': True,
                    'resume': True,
                    'position': {
                        'rotation_prob': 0.5,
                        'max_angle': 30,
                        'flip_prob': 0.5,
                        'scale_ratio': 0.3
                    },
                    'lighting': {
                        'brightness_prob': 0.5,
                        'brightness_limit': 0.3,
                        'contrast_prob': 0.5,
                        'contrast_limit': 0.3
                    }
                }
            }
            
            # Update UI from defaults
            try:
                # Augmentation options
                ui_components['augmentation_options'].children[0].value = ['Combined (Recommended)']
                ui_components['augmentation_options'].children[1].value = 2
                ui_components['augmentation_options'].children[2].value = 'aug'
                ui_components['augmentation_options'].children[3].value = True
                ui_components['augmentation_options'].children[4].value = True
                
                # Position options
                ui_components['position_options'].children[0].value = 0.5
                ui_components['position_options'].children[1].value = 30
                ui_components['position_options'].children[2].value = 0.5
                ui_components['position_options'].children[3].value = 0.3
                
                # Lighting options
                ui_components['lighting_options'].children[0].value = 0.5
                ui_components['lighting_options'].children[1].value = 0.3
                ui_components['lighting_options'].children[2].value = 0.5
                ui_components['lighting_options'].children[3].value = 0.3
                
                # Update config
                if config:
                    config['augmentation'] = default_config['augmentation']
                
                display(create_status_indicator("success", "✅ Pengaturan berhasil direset ke default"))
                
            except Exception as e:
                display(create_status_indicator("error", f"❌ Error saat reset: {str(e)}"))
    
    # Handler untuk cleanup augmented data
    def on_cleanup_data(b):
        with ui_components['augmentation_status']:
            clear_output()
            display(create_status_indicator("info", "🔄 Membersihkan data hasil augmentasi..."))
            
            try:
                # Get augmentation output directory
                output_dir = config.get('augmentation', {}).get('output_dir', 'data/augmented')
                if not output_dir:
                    output_dir = 'data/augmented'
                
                # Clean augmented directory
                output_path = Path(output_dir)
                if output_path.exists():
                    # Create backup if needed
                    backup_dir = config.get('cleanup', {}).get('backup_dir', 'data/backup/augmentation')
                    backup_path = Path(backup_dir)
                    backup_path.mkdir(parents=True, exist_ok=True)
                    
                    # Timestamp for backup
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    backup_target = backup_path / f"augmented_{timestamp}"
                    
                    # Try to back up first
                    try:
                        shutil.copytree(output_path, backup_target)
                        display(create_status_indicator(
                            "info", 
                            f"📦 Data augmentasi di-backup ke: {backup_target}"
                        ))
                    except Exception as e:
                        display(create_status_indicator(
                            "warning", 
                            f"⚠️ Tidak bisa membuat backup: {str(e)}"
                        ))
                    
                    # Delete augmented data
                    shutil.rmtree(output_path)
                    display(create_status_indicator(
                        "success", 
                        f"✅ Data augmentasi berhasil dihapus dari: {output_path}"
                    ))
                    
                    # Enable restore button for the future
                    ui_components['restore_button'].layout.display = 'inline-block'
                else:
                    display(create_status_indicator(
                        "info", 
                        f"ℹ️ Tidak ada data augmentasi untuk dibersihkan di: {output_path}"
                    ))
            
            except Exception as e:
                display(create_status_indicator("error", f"❌ Error saat membersihkan data: {str(e)}"))
    
    # Handler untuk save configuration
    def on_save_config(b):
        with ui_components['augmentation_status']:
            clear_output()
            display(create_status_indicator("info", "🔄 Menyimpan konfigurasi augmentasi..."))
            
            try:
                # Collect config dari UI
                aug_types_widgets = ui_components['augmentation_options'].children[0].value
                variations = ui_components['augmentation_options'].children[1].value
                prefix = ui_components['augmentation_options'].children[2].value
                validate = ui_components['augmentation_options'].children[3].value
                resume = ui_components['augmentation_options'].children[4].value
                
                # Map UI types to config format
                type_map = {
                    'Combined (Recommended)': 'combined',
                    'Position Variations': 'position',
                    'Lighting Variations': 'lighting',
                    'Extreme Rotation': 'extreme_rotation'
                }
                aug_types = [type_map.get(t, 'combined') for t in aug_types_widgets]
                
                # Position params
                position_params = {
                    'rotation_prob': ui_components['position_options'].children[0].value,
                    'max_angle': ui_components['position_options'].children[1].value,
                    'flip_prob': ui_components['position_options'].children[2].value,
                    'scale_ratio': ui_components['position_options'].children[3].value
                }
                
                # Lighting params
                lighting_params = {
                    'brightness_prob': ui_components['lighting_options'].children[0].value,
                    'brightness_limit': ui_components['lighting_options'].children[1].value,
                    'contrast_prob': ui_components['lighting_options'].children[2].value,
                    'contrast_limit': ui_components['lighting_options'].children[3].value
                }
                
                # Update config
                if not config:
                    config = {}
                
                config['augmentation'] = {
                    'enabled': True,
                    'types': aug_types,
                    'num_variations': variations,
                    'output_prefix': prefix,
                    'validate_results': validate,
                    'resume': resume,
                    'output_dir': 'data/augmented',
                    'position': position_params,
                    'lighting': lighting_params
                }
                
                # Save to file if config_manager available
                if config_manager:
                    success = config_manager.save_config(
                        config, 
                        filename="configs/augmentation_config.yaml",
                        backup=True
                    )
                    
                    if success:
                        display(create_status_indicator(
                            "success", 
                            "✅ Konfigurasi berhasil disimpan ke configs/augmentation_config.yaml"
                        ))
                    else:
                        display(create_status_indicator(
                            "warning", 
                            "⚠️ Konfigurasi diupdate dalam memori, tetapi gagal menyimpan ke file"
                        ))
                else:
                    # Just update in-memory if config_manager not available
                    display(create_status_indicator(
                        "success", 
                        "✅ Konfigurasi diupdate dalam memori"
                    ))
                
            except Exception as e:
                display(create_status_indicator("error", f"❌ Error saat menyimpan konfigurasi: {str(e)}"))
    
    # Update UI dari konfigurasi yang ada
    def update_ui_from_config():
        """Update UI components from current config."""
        if not config or 'augmentation' not in config:
            return
        
        try:
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
                
                # Select appropriate options if they exist in the dropdown
                available_options = ui_components['augmentation_options'].children[0].options
                valid_types = [opt for opt in ui_types if opt in available_options]
                
                if valid_types:
                    ui_components['augmentation_options'].children[0].value = valid_types
            
            # Set numeric values and checkboxes
            if 'num_variations' in aug_config:
                ui_components['augmentation_options'].children[1].value = aug_config['num_variations']
                
            if 'output_prefix' in aug_config:
                ui_components['augmentation_options'].children[2].value = aug_config['output_prefix']
                
            if 'validate_results' in aug_config:
                ui_components['augmentation_options'].children[3].value = aug_config['validate_results']
                
            if 'resume' in aug_config:
                ui_components['augmentation_options'].children[4].value = aug_config['resume']
            
            # Set position params
            if 'position' in aug_config:
                pos = aug_config['position']
                # Map config keys to UI indices
                pos_map = {
                    'rotation_prob': 0,
                    'max_angle': 1,
                    'flip_prob': 2,
                    'scale_ratio': 3
                }
                
                for key, idx in pos_map.items():
                    if key in pos:
                        ui_components['position_options'].children[idx].value = pos[key]
            
            # Set lighting params
            if 'lighting' in aug_config:
                light = aug_config['lighting']
                # Map config keys to UI indices
                light_map = {
                    'brightness_prob': 0,
                    'brightness_limit': 1,
                    'contrast_prob': 2,
                    'contrast_limit': 3
                }
                
                for key, idx in light_map.items():
                    if key in light:
                        ui_components['lighting_options'].children[idx].value = light[key]
        
        except Exception as e:
            if logger:
                logger.warning(f"⚠️ Error updating UI from config: {str(e)}")
    
    # Register callbacks
    ui_components['augmentation_button'].on_click(on_augmentation_click)
    ui_components['reset_button'].on_click(on_reset_settings)
    ui_components['cleanup_button'].on_click(on_cleanup_data)
    ui_components['save_config_button'].on_click(on_save_config)
    
    # Update UI from config
    update_ui_from_config()
    
    # Cleanup function
    def cleanup():
        """Bersihkan resources saat cell dihapus atau notebook ditutup."""
        if observer_manager:
            observer_manager.unregister_group(observer_group)
            if logger:
                logger.info("✅ Observer augmentasi telah dibersihkan")
    
    # Add cleanup function
    ui_components['cleanup'] = cleanup
    
    return ui_components