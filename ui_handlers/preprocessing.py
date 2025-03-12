"""
File: smartcash/ui_handlers/preprocessing.py
Author: Refactored
Deskripsi: Handler untuk UI preprocessing dataset SmartCash yang terintegrasi dengan 
           ui_utils dan menggunakan existing utils lain yang relevan.
"""

import threading
from IPython.display import display, clear_output

def setup_preprocessing_handlers(ui_components, config=None):
    """
    Setup handlers untuk UI preprocessing dataset.
    
    Args:
        ui_components: Dict berisi komponen UI
        config: Dict konfigurasi (opsional)
        
    Returns:
        Dict komponen UI yang sudah disetup dengan handlers
    """
    # Import library yang dibutuhkan
    try:
        from smartcash.utils.logger import get_logger
        from smartcash.handlers.preprocessing import PreprocessingManager
        from smartcash.utils.observer import EventDispatcher, EventTopics
        from smartcash.utils.observer.observer_manager import ObserverManager
        from smartcash.utils.environment_manager import EnvironmentManager
        from smartcash.utils.ui_utils import (
            create_info_alert, create_status_indicator, 
            create_metric_display, update_output_area
        )
        
        logger = get_logger("preprocessing")
        env_manager = EnvironmentManager(logger=logger)
        
        observer_manager = ObserverManager(auto_register=True)
        observer_group = "preprocessing_observers"
        
        # Bersihkan observer lama untuk menghindari memory leak
        observer_manager.unregister_group(observer_group)
        
    except ImportError as e:
        import ipywidgets as widgets
        ui_components['preprocess_status'].append_display_data(
            widgets.HTML(f"<p style='color:red'>❌ Error: {str(e)}</p>")
        )
        return ui_components
    
    # Default config jika tidak disediakan
    if config is None:
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
    
    # Ekstrak komponen UI
    preprocess_options = ui_components['preprocess_options']
    validation_options = ui_components['validation_options']
    split_selector = ui_components['split_selector']
    preprocess_button = ui_components['preprocess_button']
    stop_button = ui_components['stop_button']
    progress_bar = ui_components['progress_bar']
    current_progress = ui_components['current_progress']
    preprocess_status = ui_components['preprocess_status']
    log_accordion = ui_components['log_accordion']
    summary_container = ui_components['summary_container']
    
    # Status flags dan threading
    processing_active = False
    processing_thread = None
    
    # Setup PreprocessingManager
    data_dir = config.get('data_dir', 'data')
    
    # Gunakan environment manager untuk resolve path
    if env_manager:
        data_dir = env_manager.get_path(data_dir)
        
    preprocessing_manager = PreprocessingManager(
        config=config, 
        logger=logger, 
        base_dir=data_dir
    )
    
    # Inisialisasi UI dari config
    def initialize_ui_from_config():
        """Inisialisasi UI dari config."""
        if 'data' in config and 'preprocessing' in config['data']:
            preproc_config = config['data']['preprocessing']
            
            # Update image size
            if 'img_size' in preproc_config:
                try:
                    img_size = preproc_config['img_size']
                    if isinstance(img_size, list) and len(img_size) == 2:
                        preprocess_options.children[0].value = img_size
                except Exception:
                    pass
            
            # Update checkboxes
            for key, idx in [('normalize_enabled', 1), ('cache_enabled', 2)]:
                if key in preproc_config:
                    preprocess_options.children[idx].value = preproc_config[key]
            
            # Update workers
            if 'num_workers' in preproc_config:
                preprocess_options.children[3].value = preproc_config['num_workers']
    
    # Setup observers untuk monitoring
    def setup_observers():
        observer_manager.create_simple_observer(
            event_type=EventTopics.PREPROCESSING_PROGRESS,
            callback=lambda _, __, progress=0, total=100, message=None, **kwargs: update_progress(progress, total, message),
            name="PreprocessingProgressObserver",
            group=observer_group
        )
        
        observer_manager.create_simple_observer(
            event_type=EventTopics.VALIDATION_PROGRESS,
            callback=lambda _, __, progress=0, total=100, message=None, **kwargs: update_current_progress(progress, total, message),
            name="ValidationProgressObserver",
            group=observer_group
        )
        
        observer_manager.create_logging_observer(
            event_types=[
                EventTopics.PREPROCESSING_START,
                EventTopics.PREPROCESSING_END,
                EventTopics.PREPROCESSING_ERROR,
                EventTopics.VALIDATION_EVENT,
                EventTopics.DATASET_VALIDATE
            ],
            logger_name="preprocessing",
            name="PreprocessingLogObserver",
            group=observer_group
        )
    
    # Update progress helpers
    def update_progress(progress, total, message=None):
        progress_bar.value = int(progress * 100 / total) if total > 0 else 0
        progress_bar.description = f"{int(progress * 100 / total)}%" if total > 0 else "0%"
        if message:
            update_output_area(preprocess_status, message, "info")
    
    def update_current_progress(progress, total, message=None):
        current_progress.value = int(progress * 100 / total) if total > 0 else 0
        current_progress.description = f"{int(progress * 100 / total)}%" if total > 0 else "0%"
    
    # Fungsi untuk update UI pada hasil preprocessing
    def update_summary(result):
        from IPython.display import HTML
        
        with summary_container:
            clear_output()
            
            # Heading
            display(HTML("<h3>📊 Preprocessing Summary</h3>"))
            
            # Extract stats
            validation_stats = {}
            analysis_stats = {}
            
            for split in ['train', 'valid', 'test']:
                if split in result.get('validation', {}):
                    validation_stats[split] = result['validation'][split].get('validation_stats', {})
                
                if split in result.get('analysis', {}):
                    analysis_stats[split] = result['analysis'][split].get('analysis', {})
            
            # Validation stats
            if validation_stats:
                display(HTML("<h4>🔍 Validation Results</h4>"))
                
                stat_table = "<table style='width:100%; border-collapse:collapse; margin:10px 0'>"
                stat_table += "<tr style='background:#f2f2f2'><th>Split</th><th>Total</th><th>Valid</th><th>Invalid</th><th>Rate</th></tr>"
                
                for split, stats in validation_stats.items():
                    total = stats.get('total_images', 0)
                    valid = stats.get('valid_images', 0)
                    invalid = total - valid
                    rate = (valid / total * 100) if total > 0 else 0
                    
                    stat_table += f"<tr><td>{split}</td><td>{total}</td><td>{valid}</td><td>{invalid}</td><td>{rate:.1f}%</td></tr>"
                
                stat_table += "</table>"
                display(HTML(stat_table))
            
            # Class distribution
            if analysis_stats:
                display(HTML("<h4>📊 Class Distribution</h4>"))
                
                for split, stats in analysis_stats.items():
                    if 'class_distribution' in stats:
                        class_dist = stats['class_distribution']
                        display(HTML(f"<h5>{split.capitalize()}</h5>"))
                        
                        dist_table = "<table style='width:100%; border-collapse:collapse; margin:10px 0'>"
                        dist_table += "<tr style='background:#f2f2f2'><th>Class</th><th>Count</th><th>Percentage</th></tr>"
                        
                        total = sum(class_dist.values())
                        for cls, count in class_dist.items():
                            percentage = (count / total * 100) if total > 0 else 0
                            dist_table += f"<tr><td>{cls}</td><td>{count}</td><td>{percentage:.1f}%</td></tr>"
                        
                        dist_table += "</table>"
                        display(HTML(dist_table))
            
            # Execution time
            if 'elapsed' in result:
                display(HTML(f"<p><b>⏱️ Execution time:</b> {result['elapsed']:.2f} seconds</p>"))
                
            # Output directory
            display(HTML(f"<p><b>📂 Output:</b> {config.get('data_dir', 'data')}</p>"))
        
        # Show summary
        summary_container.layout.display = 'block'
    
    # Handler untuk tombol preprocessing
    def on_preprocess_click(b):
        nonlocal processing_active, processing_thread
        
        if processing_active:
            return
        
        processing_active = True
        with preprocess_status:
            clear_output()
            display(create_status_indicator("info", "🔄 Starting preprocessing..."))
        
        # Set UI untuk processing
        progress_bar.value = 0
        current_progress.value = 0
        progress_bar.layout.visibility = 'visible'
        current_progress.layout.visibility = 'visible'
        preprocess_button.disabled = True
        stop_button.layout.display = 'inline-block'
        log_accordion.selected_index = 0
        
        # Get preprocessing parameters
        img_size = preprocess_options.children[0].value
        normalize = preprocess_options.children[1].value
        cache = preprocess_options.children[2].value
        workers = preprocess_options.children[3].value
        
        # Update config
        config['data']['preprocessing'] = {
            'img_size': list(img_size),
            'normalize_enabled': normalize,
            'cache_enabled': cache,
            'num_workers': workers
        }
        
        # Get validation options
        validate = validation_options.children[0].value
        fix_issues = validation_options.children[1].value
        
        # Get splits to process
        split_option = split_selector.value
        if split_option == 'All Splits':
            splits_to_process = ['train', 'valid', 'test']
        elif split_option == 'Train Only':
            splits_to_process = ['train']
        elif split_option == 'Validation Only':
            splits_to_process = ['valid']
        else:  # Test Only
            splits_to_process = ['test']
        
        # Define preprocessing function
        def run_preprocessing_thread():
            nonlocal processing_active
            
            try:
                # Notify start
                EventDispatcher.notify(
                    event_type=EventTopics.PREPROCESSING_START,
                    sender="preprocessing_handler",
                    message="Starting preprocessing pipeline"
                )
                
                # Run preprocessing pipeline
                result = preprocessing_manager.run_full_pipeline(
                    splits=splits_to_process,
                    validate_dataset=validate,
                    fix_issues=fix_issues,
                    augment_data=False,
                    analyze_dataset=True
                )
                
                # Process result
                if result['status'] == 'success':
                    # Update UI in main thread
                    import ipywidgets as widgets
                    preprocess_status.append_display_data(
                        create_status_indicator("success", f"✅ Preprocessing completed in {result.get('elapsed', 0):.2f} seconds")
                    )
                    
                    # Update summary
                    update_summary(result)
                    
                    # Show cleanup button
                    cleanup_button.layout.display = 'inline-block'
                    
                    # Notify completion
                    EventDispatcher.notify(
                        event_type=EventTopics.PREPROCESSING_END,
                        sender="preprocessing_handler",
                        result=result,
                        message="Preprocessing completed successfully"
                    )
                else:
                    # Show error
                    preprocess_status.append_display_data(
                        create_status_indicator("error", f"❌ Preprocessing failed: {result.get('error', 'Unknown error')}")
                    )
                    
                    # Notify error
                    EventDispatcher.notify(
                        event_type=EventTopics.PREPROCESSING_ERROR,
                        sender="preprocessing_handler",
                        error=result.get('error', 'Unknown error')
                    )
            except Exception as e:
                # Show error
                preprocess_status.append_display_data(
                    create_status_indicator("error", f"❌ Error: {str(e)}")
                )
                
                # Notify error
                EventDispatcher.notify(
                    event_type=EventTopics.PREPROCESSING_ERROR,
                    sender="preprocessing_handler",
                    error=str(e)
                )
            finally:
                # Reset UI
                processing_active = False
                preprocess_button.disabled = False
                stop_button.layout.display = 'none'
                progress_bar.layout.visibility = 'hidden'
                current_progress.layout.visibility = 'hidden'
        
        # Start preprocessing in thread
        processing_thread = threading.Thread(target=run_preprocessing_thread)
        processing_thread.daemon = True
        processing_thread.start()
    
    # Stop handler
    def on_stop_click(b):
        nonlocal processing_active
        processing_active = False
        
        with preprocess_status:
            display(create_status_indicator("warning", "⚠️ Stopping preprocessing..."))
        
        # Notify
        EventDispatcher.notify(
            event_type=EventTopics.PREPROCESSING_END,
            sender="preprocessing_handler",
            message="Preprocessing stopped by user"
        )
    
    # Clean preprocessed data handler
    def on_cleanup_click(b):
        import shutil
        from pathlib import Path
        
        with preprocess_status:
            clear_output()
            display(create_status_indicator("info", "🗑️ Cleaning preprocessed data..."))
            
            try:
                # Get preprocessed directory
                preprocessed_dir = config.get('data', {}).get('preprocessing', {}).get('output_dir', 'data/preprocessed')
                
                # Use environment manager if available
                if env_manager:
                    preprocessed_dir = env_manager.get_path(preprocessed_dir)
                
                preprocessed_path = Path(preprocessed_dir)
                
                # Check if directory exists
                if preprocessed_path.exists():
                    # Remove directory
                    shutil.rmtree(preprocessed_path)
                    display(create_status_indicator("success", f"✅ Removed preprocessed data: {preprocessed_dir}"))
                else:
                    display(create_status_indicator("info", f"ℹ️ No preprocessed data found at: {preprocessed_dir}"))
                
                # Hide cleanup button
                cleanup_button.layout.display = 'none'
                
                # Hide summary
                summary_container.layout.display = 'none'
                
            except Exception as e:
                display(create_status_indicator("error", f"❌ Error: {str(e)}"))
    
    # Setup cleanup handler
    def cleanup():
        """Fungsi cleanup untuk unregister observer."""
        if observer_manager:
            observer_manager.unregister_group(observer_group)
            logger.info(f"✅ Observer group {observer_group} dibersihkan")
    
    # Register handlers
    preprocess_button.on_click(on_preprocess_click)
    stop_button.on_click(on_stop_click)
    
    # Add cleanup button
    import ipywidgets as widgets
    cleanup_button = widgets.Button(
        description='Clean Preprocessed Data',
        button_style='danger',
        icon='trash',
        layout=widgets.Layout(display='none')
    )
    cleanup_button.on_click(on_cleanup_click)
    
    # Add to container
    ui_components['ui'].children = list(ui_components['ui'].children) + [cleanup_button]
    ui_components['cleanup_button'] = cleanup_button
    
    # Initialize UI
    initialize_ui_from_config()
    
    # Setup observers
    setup_observers()
    
    # Add cleanup function
    ui_components['cleanup'] = cleanup
    
    return ui_components