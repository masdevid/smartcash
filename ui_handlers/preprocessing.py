"""
File: smartcash/ui_handlers/preprocessing.py
Author: Refactored
Deskripsi: Handler untuk UI preprocessing dataset SmartCash dengan kode minimal dan perbaikan bug.
"""

import threading
import time
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

def setup_preprocessing_handlers(ui_components, config=None):
    """Setup handlers untuk UI preprocessing dataset."""
    # Import necessities
    try:
        from smartcash.utils.logger import get_logger
        from smartcash.handlers.preprocessing import PreprocessingManager
        from smartcash.utils.observer import EventDispatcher, EventTopics
        from smartcash.utils.observer.observer_manager import ObserverManager
        from smartcash.utils.environment_manager import EnvironmentManager
        from smartcash.utils.config_manager import ConfigManager
        
        logger = get_logger("preprocessing")
        env_manager = EnvironmentManager(logger=logger)
        
        # Ensure observer cleanup from previous sessions
        observer_manager = ObserverManager(auto_register=True)
        observer_group = "preprocessing_observers"
        observer_manager.unregister_group(observer_group)
        
        # Make sure we have a valid config
        if not config or not isinstance(config, dict) or len(config) == 0:
            config_manager = ConfigManager.get_instance(logger=logger)
            config = config_manager.get_config()
            
            # If still empty, create default config
            if not config or len(config) == 0:
                config = {
                    'data': {
                        'preprocessing': {
                            'img_size': [640, 640],
                            'num_workers': 4,
                            'normalize_enabled': True,
                            'cache_enabled': True
                        }
                    },
                    'data_dir': 'data'
                }
                config_manager.update_config(config, save=True)
        
    except ImportError as e:
        ui_components['preprocess_status'].append_display_data(
            HTML(f"<p style='color:red'>❌ Error: {str(e)}</p>")
        )
        return ui_components
    
    # Extract UI elements
    ui = {k: ui_components[k] for k in ['preprocess_options', 'validation_options', 'split_selector', 
                                      'preprocess_button', 'stop_button', 'progress_bar', 'current_progress', 
                                      'preprocess_status', 'log_accordion', 'summary_container']}
    
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
                ui['preprocess_options'].children[0].value = cfg['img_size']
            
            # Set checkboxes and workers
            for key, idx in [('normalize_enabled', 1), ('cache_enabled', 2), ('num_workers', 3)]:
                if key in cfg:
                    ui['preprocess_options'].children[idx].value = cfg[key]
    
    def update_ui_for_processing(is_processing):
        """Update UI based on processing state"""
        ui['progress_bar'].layout.visibility = 'visible' if is_processing else 'hidden'
        ui['current_progress'].layout.visibility = 'visible' if is_processing else 'hidden'
        ui['preprocess_button'].disabled = is_processing
        ui['stop_button'].layout.display = 'inline-block' if is_processing else 'none'
        if is_processing:
            ui['log_accordion'].selected_index = 0
    
    def create_status_indicator(status, message):
        """Create a styled status indicator"""
        status_styles = {
            'success': {'icon': '✅', 'color': 'green'},
            'warning': {'icon': '⚠️', 'color': 'orange'},
            'error': {'icon': '❌', 'color': 'red'},
            'info': {'icon': 'ℹ️', 'color': 'blue'}
        }
        
        style = status_styles.get(status, status_styles['info'])
        
        return HTML(f"""
        <div style="margin: 5px 0; padding: 8px 12px; 
                    border-radius: 4px; background-color: #f8f9fa;">
            <span style="color: {style['color']}; font-weight: bold;"> 
                {style['icon']} {message}
            </span>
        </div>
        """)
            
    def update_progress(_, __, progress=0, total=100, message=None, **kwargs):
        """Update progress bar and display messages"""
        if stop_requested:
            return
            
        bar = ui['progress_bar']
        bar.value = int(progress * 100 / total) if total > 0 else 0
        bar.description = f"{int(progress * 100 / total)}%" if total > 0 else "0%"
        if message:
            with ui['preprocess_status']:
                display(create_status_indicator("info", message))
    
    def update_summary(result):
        """Display preprocessing results summary"""
        with ui['summary_container']:
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
            display(HTML(f"<p><b>📂 Output:</b> {config.get('data_dir', 'data')}</p>"))
        
        ui['summary_container'].layout.display = 'block'
        cleanup_button.layout.display = 'inline-block'
    
    # Setup observers
    for event, callback, name in [
        (EventTopics.PREPROCESSING_PROGRESS, update_progress, "ProgressObserver"),
        (EventTopics.VALIDATION_PROGRESS, 
         lambda _, __, progress=0, total=100, **kwargs: 
             setattr(ui['current_progress'], 'value', int(progress * 100 / total) if total > 0 else 0) or
             setattr(ui['current_progress'], 'description', f"{int(progress * 100 / total)}%" if total > 0 else "0%"),
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
    
    # Handler functions
    def run_preprocessing_thread():
        """Execute preprocessing in a separate thread"""
        nonlocal processing_active, stop_requested
        
        try:
            # Get preprocessing parameters
            img_size = ui['preprocess_options'].children[0].value
            normalize = ui['preprocess_options'].children[1].value
            cache = ui['preprocess_options'].children[2].value
            workers = ui['preprocess_options'].children[3].value
            
            # Get validation options and splits
            validate = ui['validation_options'].children[0].value
            fix_issues = ui['validation_options'].children[1].value
            
            # Determine splits to process
            split_map = {
                'All Splits': ['train', 'valid', 'test'],
                'Train Only': ['train'],
                'Validation Only': ['valid'],
                'Test Only': ['test']
            }
            splits = split_map.get(ui['split_selector'].value, ['train', 'valid', 'test'])
            
            # Update config
            config['data']['preprocessing'] = config.get('data', {}).get('preprocessing', {})
            config['data']['preprocessing'].update({
                'img_size': list(img_size),
                'normalize_enabled': normalize,
                'cache_enabled': cache,
                'num_workers': workers
            })
            
            # Notify start
            EventDispatcher.notify(event_type=EventTopics.PREPROCESSING_START, 
                                  sender="preprocessing_handler",
                                  message="Starting preprocessing pipeline")
            
            # Check for stop request periodically
            start_time = time.time()
            
            # Run pipeline
            if not stop_requested:
                result = preprocessing_manager.run_full_pipeline(
                    splits=splits, validate_dataset=validate,
                    fix_issues=fix_issues, augment_data=False, analyze_dataset=True
                )
                
                # Process result
                if result['status'] == 'success':
                    ui['preprocess_status'].append_display_data(
                        create_status_indicator("success", 
                                             f"✅ Preprocessing completed in {result.get('elapsed', 0):.2f} seconds")
                    )
                    update_summary(result)
                    EventDispatcher.notify(event_type=EventTopics.PREPROCESSING_END, 
                                        sender="preprocessing_handler",
                                        result=result)
                else:
                    ui['preprocess_status'].append_display_data(
                        create_status_indicator("error", 
                                             f"❌ Preprocessing failed: {result.get('error', 'Unknown error')}")
                    )
                    EventDispatcher.notify(event_type=EventTopics.PREPROCESSING_ERROR, 
                                        sender="preprocessing_handler",
                                        error=result.get('error', 'Unknown error'))
        except Exception as e:
            ui['preprocess_status'].append_display_data(
                create_status_indicator("error", f"❌ Error: {str(e)}")
            )
            EventDispatcher.notify(event_type=EventTopics.PREPROCESSING_ERROR, 
                                 sender="preprocessing_handler", error=str(e))
        finally:
            processing_active = False
            stop_requested = False
            update_ui_for_processing(False)
    
    # Main handlers
    def on_preprocess_click(b):
        """Start preprocessing"""
        nonlocal processing_active, stop_requested, current_thread
        if processing_active:
            return
        
        processing_active = True
        stop_requested = False
        with ui['preprocess_status']:
            clear_output()
            display(create_status_indicator("info", "🔄 Starting preprocessing..."))
        
        ui['progress_bar'].value = 0
        ui['current_progress'].value = 0
        ui['summary_container'].layout.display = 'none'
        update_ui_for_processing(True)
        
        # Run in thread
        current_thread = threading.Thread(target=run_preprocessing_thread)
        current_thread.daemon = True
        current_thread.start()
    
    def on_stop_click(b):
        """Stop processing"""
        nonlocal processing_active, stop_requested
        stop_requested = True
        with ui['preprocess_status']:
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
        import shutil
        from pathlib import Path
        
        with ui['preprocess_status']:
            clear_output()
            display(create_status_indicator("info", "🗑️ Cleaning preprocessed data..."))
            
            try:
                preproc_dir = config.get('data', {}).get('preprocessing', {}).get('output_dir', 'data/preprocessed')
                preproc_path = env_manager.get_path(preproc_dir) if env_manager else Path(preproc_dir)
                
                if preproc_path.exists():
                    shutil.rmtree(preproc_path)
                    display(create_status_indicator("success", f"✅ Removed: {preproc_dir}"))
                else:
                    display(create_status_indicator("info", f"ℹ️ Not found: {preproc_dir}"))
                
                cleanup_button.layout.display = 'none'
                ui['summary_container'].layout.display = 'none'
            except Exception as e:
                display(create_status_indicator("error", f"❌ Error: {str(e)}"))
    
    # Add cleanup button
    cleanup_button = widgets.Button(
        description='Clean Preprocessed Data',
        button_style='danger',
        icon='trash',
        layout=widgets.Layout(display='none')
    )
    ui_components['ui'].children = list(ui_components['ui'].children) + [cleanup_button]
    ui_components['cleanup_button'] = cleanup_button
    
    # Register event handlers
    ui['preprocess_button'].on_click(on_preprocess_click)
    ui['stop_button'].on_click(on_stop_click)
    cleanup_button.on_click(on_cleanup_click)
    
    # Initialize UI
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
        
        logger.info("✅ Preprocessing handlers cleaned up")
    
    # Add cleanup function to UI components
    ui_components['cleanup'] = cleanup
    
    return ui_components