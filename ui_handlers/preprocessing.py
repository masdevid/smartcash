"""
File: smartcash/ui_handlers/preprocessing.py
Author: Alfrida Sabar (revisi)
Deskripsi: Handler untuk UI preprocessing dataset SmartCash dengan dukungan save/load konfigurasi dan cleanup.
"""

import threading
import time
import os
import shutil
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
from pathlib import Path

def setup_preprocessing_handlers(ui_components, config=None):
    """Setup handlers untuk UI preprocessing dataset."""
    # Defensive programming - exit early if ui_components is None
    if ui_components is None:
        print("⚠️ Error: UI components is None")
        return {}
        
    # Import necessities
    try:
        from smartcash.utils.logger import get_logger
        from smartcash.handlers.preprocessing import PreprocessingManager
        from smartcash.utils.observer import EventDispatcher, EventTopics
        from smartcash.utils.observer.observer_manager import ObserverManager
        from smartcash.utils.environment_manager import EnvironmentManager
        from smartcash.utils.config_manager import get_config_manager
        from smartcash.utils.ui_utils import create_status_indicator
        
        logger = get_logger("preprocessing")
        env_manager = EnvironmentManager(logger=logger)
        
        # Ensure observer cleanup from previous sessions
        observer_manager = ObserverManager(auto_register=True)
        observer_group = "preprocessing_observers"
        observer_manager.unregister_group(observer_group)
        
        # Make sure we have a valid config
        config_manager = get_config_manager(logger=logger)
        
        # If config is empty, load from file or create default
        if not config:
            config = {}
        
        if len(config) == 0 or 'data' not in config or 'preprocessing' not in config.get('data', {}):
            loaded_config = config_manager.load_config("configs/preprocessing_config.yaml")
            if loaded_config:
                config = loaded_config
            else:
                # Create default config
                config = {
                    'data': {
                        'preprocessing': {
                            'img_size': [640, 640],
                            'num_workers': 4,
                            'normalize_enabled': True,
                            'cache_enabled': True,
                            'validation': {
                                'enabled': True,
                                'fix_issues': True,
                                'move_invalid': True,
                                'invalid_dir': 'data/invalid'
                            },
                            'output_dir': 'data/preprocessed'
                        }
                    },
                    'data_dir': 'data'
                }
                # Save default config
                config_manager.save_config(config, "configs/preprocessing_config.yaml", sync_to_drive=True)
                if logger:
                    logger.info("📝 Default preprocessing config created and saved")
    except ImportError as e:
        # Fallback to basic functionality
        try:
            from smartcash.utils.ui_utils import create_status_indicator
        except ImportError:
            # Define a simple create_status_indicator function if ui_utils is not available
            def create_status_indicator(status, message):
                icon = "✅" if status == "success" else "⚠️" if status == "warning" else "❌" if status == "error" else "ℹ️"
                return HTML(f"<div style='margin: 5px 0'>{icon} {message}</div>")
        
        if 'preprocess_status' in ui_components:
            with ui_components['preprocess_status']:
                display(HTML(f"<p style='color:red'>⚠️ Warning: Limited functionality - {str(e)}</p>"))
        else:
            print(f"⚠️ Warning: Limited functionality - {str(e)}")
        return ui_components
    
    # State variables
    processing_active = False
    stop_requested = False
    current_thread = None
    
    # Setup PreprocessingManager
    data_dir = env_manager.get_path(config.get('data_dir', 'data')) if env_manager else config.get('data_dir', 'data')
    preprocessing_manager = PreprocessingManager(config=config, logger=logger, base_dir=data_dir)
    
    # Setup UI and helpers
    def init_ui():
        """Initialize UI from config"""
        if 'data' in config and 'preprocessing' in config['data']:
            cfg = config['data']['preprocessing']
            # Set image size
            if 'img_size' in cfg and isinstance(cfg['img_size'], list) and len(cfg['img_size']) == 2:
                ui_components['preprocess_options'].children[0].value = cfg['img_size']
            
            # Set checkboxes and workers
            for key, idx in [('normalize_enabled', 1), ('cache_enabled', 2), ('num_workers', 3)]:
                if key in cfg:
                    ui_components['preprocess_options'].children[idx].value = cfg[key]
                    
            # Set validation options
            if 'validation' in cfg:
                for key, idx in [('enabled', 0), ('fix_issues', 1), ('move_invalid', 2)]:
                    if key in cfg['validation']:
                        ui_components['validation_options'].children[idx].value = cfg['validation'][key]
                
                if 'invalid_dir' in cfg['validation']:
                    ui_components['validation_options'].children[3].value = cfg['validation']['invalid_dir']
    
    def update_ui_for_processing(is_processing):
        """Update UI based on processing state"""
        ui_components['progress_bar'].layout.visibility = 'visible' if is_processing else 'hidden'
        ui_components['current_progress'].layout.visibility = 'visible' if is_processing else 'hidden'
        ui_components['preprocess_button'].disabled = is_processing
        ui_components['stop_button'].layout.display = 'inline-block' if is_processing else 'none'
        ui_components['cleanup_button'].disabled = is_processing
        if is_processing:
            ui_components['log_accordion'].selected_index = 0
    
    def update_progress(_, __, progress=0, total=100, message=None, **kwargs):
        """Update progress bar and display messages"""
        if stop_requested:
            return
            
        bar = ui_components['progress_bar']
        bar.value = int(progress * 100 / total) if total > 0 else 0
        bar.description = f"{int(progress * 100 / total)}%" if total > 0 else "0%"
        if message:
            with ui_components['preprocess_status']:
                display(create_status_indicator("info", message))
    
    def update_summary(result):
        """Display preprocessing results summary"""
        with ui_components['summary_container']:
            clear_output()
            
            # Extract stats
            validation_stats = {split: result['validation'].get(split, {}).get('validation_stats', {}) 
                              for split in ['train', 'valid', 'test'] if split in result.get('validation', {})}
            
            analysis_stats = {split: result['analysis'].get(split, {}).get('analysis', {}) 
                            for split in ['train', 'valid', 'test'] if split in result.get('analysis', {})}
            
            # Display validation stats table
            if validation_stats:
                display(HTML("<h3>📊 Preprocessing Results</h3><h4>🔍 Validation</h4>"))
                
                stat_table = "<table style='width:100%; border-collapse:collapse'><tr style='background:#f2f2f2'>"
                stat_table += "<th>Split</th><th>Total</th><th>Valid</th><th>Invalid</th><th>Rate</th></tr>"
                
                for split, stats in validation_stats.items():
                    total = stats.get('total_images', 0)
                    valid = stats.get('valid_images', 0)
                    invalid = total - valid
                    rate = (valid / total * 100) if total > 0 else 0
                    stat_table += f"<tr><td>{split}</td><td>{total}</td><td>{valid}</td>"
                    stat_table += f"<td>{invalid}</td><td>{rate:.1f}%</td></tr>"
                
                display(HTML(stat_table + "</table>"))
            
            # Display class distribution
            if analysis_stats:
                display(HTML("<h4>📊 Class Distribution</h4>"))
                
                for split, stats in analysis_stats.items():
                    if 'class_distribution' in stats:
                        class_dist = stats['class_distribution']
                        display(HTML(f"<h5>{split.capitalize()}</h5>"))
                        
                        # Create distribution table
                        dist_table = "<table style='width:100%; border-collapse:collapse'><tr style='background:#f2f2f2'>"
                        dist_table += "<th>Class</th><th>Count</th><th>Percentage</th></tr>"
                        
                        total = sum(class_dist.values())
                        for cls, count in class_dist.items():
                            pct = (count / total * 100) if total > 0 else 0
                            dist_table += f"<tr><td>{cls}</td><td>{count}</td><td>{pct:.1f}%</td></tr>"
                        
                        display(HTML(dist_table + "</table>"))
            
            # Display execution info
            if 'elapsed' in result:
                display(HTML(f"<p><b>⏱️ Execution time:</b> {result['elapsed']:.2f} seconds</p>"))
            
            output_dir = config.get('data', {}).get('preprocessing', {}).get('output_dir', 'data/preprocessed')
            display(HTML(f"<p><b>📂 Output:</b> {output_dir}</p>"))
        
        ui_components['summary_container'].layout.display = 'block'
        ui_components['cleanup_button'].layout.display = 'inline-block'
    
    # Setup observers
    for event, callback, name in [
        (EventTopics.PREPROCESSING_PROGRESS, update_progress, "ProgressObserver"),
        (EventTopics.VALIDATION_PROGRESS, 
         lambda _, __, progress=0, total=100, **kwargs: 
             setattr(ui_components['current_progress'], 'value', int(progress * 100 / total) if total > 0 else 0) or
             setattr(ui_components['current_progress'], 'description', f"{int(progress * 100 / total)}%" if total > 0 else "0%"),
         "CurrentProgressObserver")
    ]:
        observer_manager.create_simple_observer(event_type=event, callback=callback, 
                                               name=name, group=observer_group)
    
    # Add logging observer for events
    observer_manager.create_logging_observer(
        event_types=[EventTopics.PREPROCESSING_START, EventTopics.PREPROCESSING_END, 
                    EventTopics.PREPROCESSING_ERROR, EventTopics.VALIDATION_EVENT],
        logger_name="preprocessing", name="LogObserver", group=observer_group
    )
    
    # Check if preprocessed data already exists
    def check_preprocessed_exists():
        """Check if preprocessed data already exists and confirm overwrite if needed"""
        preproc_dir = config.get('data', {}).get('preprocessing', {}).get('output_dir', 'data/preprocessed')
        preproc_path = env_manager.get_path(preproc_dir) if env_manager else Path(preproc_dir)
        
        if preproc_path.exists() and any(preproc_path.iterdir()):
            # Create confirmation dialog
            confirm_box = widgets.VBox([
                widgets.HTML(
                    f"""<div style="padding: 10px; background-color: #fff3cd; color: #856404; 
                    border-left: 4px solid #856404; border-radius: 4px; margin: 10px 0;">
                    <h4 style="margin-top: 0;">⚠️ Preprocessed data already exists</h4>
                    <p>Directory <code>{preproc_dir}</code> already contains preprocessed data.</p>
                    <p>Continuing will overwrite existing data. Do you want to proceed?</p>
                    </div>"""
                ),
                widgets.HBox([
                    widgets.Button(description="Cancel", button_style="danger"),
                    widgets.Button(description="Proceed", button_style="primary")
                ])
            ])
            
            # Show confirmation dialog
            with ui_components['preprocess_status']:
                clear_output()
                display(confirm_box)
            
            # Setup confirmation buttons
            def on_cancel(b):
                with ui_components['preprocess_status']:
                    clear_output()
                    display(create_status_indicator("info", "Operation cancelled"))
            
            def on_proceed(b):
                with ui_components['preprocess_status']:
                    clear_output()
                    display(create_status_indicator("info", "🔄 Proceeding with preprocessing..."))
                start_preprocessing()
            
            confirm_box.children[1].children[0].on_click(on_cancel)
            confirm_box.children[1].children[1].on_click(on_proceed)
            
            return True
        return False
    
    def get_config_from_ui():
        """Get preprocessing config from UI components"""
        # Get preprocessing parameters
        img_size = ui_components['preprocess_options'].children[0].value
        normalize = ui_components['preprocess_options'].children[1].value
        cache = ui_components['preprocess_options'].children[2].value
        workers = ui_components['preprocess_options'].children[3].value
        
        # Get validation options
        validate = ui_components['validation_options'].children[0].value
        fix_issues = ui_components['validation_options'].children[1].value
        move_invalid = ui_components['validation_options'].children[2].value
        invalid_dir = ui_components['validation_options'].children[3].value
        
        # Update config
        if 'data' not in config:
            config['data'] = {}
        if 'preprocessing' not in config['data']:
            config['data']['preprocessing'] = {}
        
        config['data']['preprocessing'].update({
            'img_size': list(img_size),
            'normalize_enabled': normalize,
            'cache_enabled': cache,
            'num_workers': workers,
            'validation': {
                'enabled': validate,
                'fix_issues': fix_issues,
                'move_invalid': move_invalid,
                'invalid_dir': invalid_dir
            },
            'output_dir': 'data/preprocessed'  # Set default output dir
        })
        
        return config
    
    def save_config():
        """Save current preprocessing config to file"""
        if config_manager:
            updated_config = get_config_from_ui()
            config_manager.save_config(updated_config, "configs/preprocessing_config.yaml", sync_to_drive=True)
            with ui_components['preprocess_status']:
                display(create_status_indicator("success", "✅ Configuration saved to configs/preprocessing_config.yaml"))
    
    def start_preprocessing():
        """Execute preprocessing in a separate thread"""
        nonlocal processing_active, stop_requested, current_thread
        
        # Update config from UI
        get_config_from_ui()
        
        # Save config
        if config_manager:
            config_manager.save_config(config, "configs/preprocessing_config.yaml", sync_to_drive=True)
        
        try:
            # Determine splits to process
            split_map = {
                'All Splits': ['train', 'valid', 'test'],
                'Train Only': ['train'],
                'Validation Only': ['valid'],
                'Test Only': ['test']
            }
            splits = split_map.get(ui_components['split_selector'].value, ['train', 'valid', 'test'])
            
            # Notify start
            EventDispatcher.notify(event_type=EventTopics.PREPROCESSING_START, 
                                  sender="preprocessing_handler",
                                  message="Starting preprocessing pipeline")
            
            # Check for stop request periodically
            start_time = time.time()
            
            # Run pipeline
            if not stop_requested:
                # Ensure validation settings are in config
                if 'validation' not in config['data']['preprocessing']:
                    validation_options = ui_components['validation_options']
                    config['data']['preprocessing']['validation'] = {
                        'enabled': validation_options.children[0].value,
                        'fix_issues': validation_options.children[1].value,
                        'move_invalid': validation_options.children[2].value,
                        'invalid_dir': validation_options.children[3].value
                    }
                
                result = preprocessing_manager.run_full_pipeline(
                    splits=splits, 
                    validate_dataset=config['data']['preprocessing']['validation']['enabled'],
                    fix_issues=config['data']['preprocessing']['validation']['fix_issues'], 
                    augment_data=False, 
                    analyze_dataset=True
                )
                
                # Process result
                if result['status'] == 'success':
                    with ui_components['preprocess_status']:
                        display(create_status_indicator("success", 
                                                     f"✅ Preprocessing completed in {result.get('elapsed', 0):.2f} seconds"))
                    update_summary(result)
                    EventDispatcher.notify(event_type=EventTopics.PREPROCESSING_END, 
                                        sender="preprocessing_handler",
                                        result=result)
                else:
                    with ui_components['preprocess_status']:
                        display(create_status_indicator("error", 
                                                     f"❌ Preprocessing failed: {result.get('error', 'Unknown error')}"))
                    EventDispatcher.notify(event_type=EventTopics.PREPROCESSING_ERROR, 
                                        sender="preprocessing_handler",
                                        error=result.get('error', 'Unknown error'))
        except Exception as e:
            with ui_components['preprocess_status']:
                display(create_status_indicator("error", f"❌ Error: {str(e)}"))
            EventDispatcher.notify(event_type=EventTopics.PREPROCESSING_ERROR, 
                                 sender="preprocessing_handler", error=str(e))
        finally:
            processing_active = False
            stop_requested = False
            update_ui_for_processing(False)
    
    def run_preprocessing_thread():
        """Start preprocessing in a thread"""
        nonlocal processing_active, current_thread
        
        if processing_active:
            return
        
        processing_active = True
        stop_requested = False
        with ui_components['preprocess_status']:
            clear_output()
            display(create_status_indicator("info", "🔄 Starting preprocessing..."))
        
        ui_components['progress_bar'].value = 0
        ui_components['current_progress'].value = 0
        ui_components['summary_container'].layout.display = 'none'
        ui_components['cleanup_button'].layout.display = 'none'
        update_ui_for_processing(True)
        
        # Run in thread
        current_thread = threading.Thread(target=start_preprocessing)
        current_thread.daemon = True
        current_thread.start()
    
    # Main handlers
    def on_preprocess_click(b):
        """Start preprocessing if no preprocessed data exists, otherwise confirm"""
        if check_preprocessed_exists():
            # Confirmation dialog will handle starting if user confirms
            pass
        else:
            # No existing data, start directly
            run_preprocessing_thread()
    
    def on_stop_click(b):
        """Stop processing"""
        nonlocal processing_active, stop_requested
        stop_requested = True
        with ui_components['preprocess_status']:
            display(create_status_indicator("warning", "⚠️ Stopping preprocessing..."))
        
        # Force update UI after a short delay
        def delayed_ui_update():
            time.sleep(0.5)  # Short delay to allow observer notification
            processing_active = False
            update_ui_for_processing(False)
            
        threading.Thread(target=delayed_ui_update, daemon=True).start()
        
        EventDispatcher.notify(event_type=EventTopics.PREPROCESSING_END, 
                             sender="preprocessing_handler",
                             message="Preprocessing stopped by user")
    
    def on_cleanup_click(b):
        """Clean preprocessed data"""
        preproc_dir = config.get('data', {}).get('preprocessing', {}).get('output_dir', 'data/preprocessed')
        preproc_path = env_manager.get_path(preproc_dir) if env_manager else Path(preproc_dir)
        
        # Create confirmation dialog
        confirm_box = widgets.VBox([
            widgets.HTML(
                f"""<div style="padding: 10px; background-color: #fff3cd; color: #856404; 
                border-left: 4px solid #856404; border-radius: 4px; margin: 10px 0;">
                <h4 style="margin-top: 0;">⚠️ Confirm Data Cleanup</h4>
                <p>This will delete all preprocessed data in <code>{preproc_dir}</code>.</p>
                <p>Original dataset will not be affected. Do you want to proceed?</p>
                </div>"""
            ),
            widgets.HBox([
                widgets.Button(description="Cancel", button_style="warning"),
                widgets.Button(description="Delete Preprocessed Data", button_style="danger")
            ])
        ])
        
        # Show confirmation dialog
        with ui_components['preprocess_status']:
            clear_output()
            display(confirm_box)
        
        # Setup confirmation buttons
        def on_cancel(b):
            with ui_components['preprocess_status']:
                clear_output()
                display(create_status_indicator("info", "Operation cancelled"))
        
        def on_confirm_delete(b):
            with ui_components['preprocess_status']:
                clear_output()
                display(create_status_indicator("info", "🗑️ Cleaning preprocessed data..."))
                
                try:
                    if preproc_path.exists():
                        # Use dataset utils if available
                        try:
                            from smartcash.utils.dataset.dataset_utils import DatasetUtils
                            utils = DatasetUtils(config, logger=logger)
                            backup_path = utils.backup_directory(preproc_path, suffix="backup_before_delete")
                            if backup_path:
                                display(create_status_indicator("info", f"📦 Backup created: {backup_path}"))
                            shutil.rmtree(preproc_path)
                        except ImportError:
                            # Fallback to direct deletion
                            shutil.rmtree(preproc_path)
                        
                        display(create_status_indicator("success", f"✅ Removed: {preproc_dir}"))
                    else:
                        display(create_status_indicator("info", f"ℹ️ Not found: {preproc_dir}"))
                    
                    # Hide cleanup button and summary after cleanup
                    ui_components['cleanup_button'].layout.display = 'none'
                    ui_components['summary_container'].layout.display = 'none'
                except Exception as e:
                    display(create_status_indicator("error", f"❌ Error: {str(e)}"))
        
        confirm_box.children[1].children[0].on_click(on_cancel)
        confirm_box.children[1].children[1].on_click(on_confirm_delete)
    
    # Add save config button
    save_config_button = widgets.Button(
        description='Save Config',
        button_style='info',
        icon='save'
    )
    save_config_button.on_click(lambda b: save_config())
    
    # Add buttons to UI
    ui_components['save_config_button'] = save_config_button
    
    # Add Save Config button to the button container
    button_container = ui_components['preprocess_button'].parent
    if isinstance(button_container, widgets.HBox):
        # Create new button container with all buttons
        new_button_container = widgets.HBox([
            ui_components['preprocess_button'], 
            ui_components['stop_button'],
            save_config_button
        ])
        
        # Replace the old button container in the UI
        for i, child in enumerate(ui_components['ui'].children):
            if child is button_container:
                new_children = list(ui_components['ui'].children)
                new_children[i] = new_button_container
                ui_components['ui'].children = tuple(new_children)
                break
    
    # Register event handlers
    ui_components['preprocess_button'].on_click(on_preprocess_click)
    ui_components['stop_button'].on_click(on_stop_click)
    ui_components['cleanup_button'].on_click(on_cleanup_click)
    
    # Initialize UI from config
    init_ui()
    
    # Define proper cleanup function
    def cleanup():
        """Clean up observer registrations and resources"""
        nonlocal stop_requested
        stop_requested = True
        
        # Unregister observers
        observer_manager.unregister_group(observer_group)
        
        # Reset UI
        update_ui_for_processing(False)
        
        if logger:
            logger.info("✅ Preprocessing handlers cleaned up")
    
    # Add cleanup function to UI components
    ui_components['cleanup'] = cleanup
    
    return ui_components