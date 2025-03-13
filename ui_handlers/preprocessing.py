"""
File: smartcash/ui_handlers/preprocessing.py
Author: Optimasi dan konsolidasi progress tracking
Deskripsi: Handler untuk UI preprocessing dataset SmartCash dengan progress tracking yang lebih baik.
"""

import threading
import time
import shutil
from pathlib import Path
from IPython.display import display, clear_output, HTML

def setup_preprocessing_handlers(ui_components, config=None):
    """Setup handlers untuk UI preprocessing dataset."""
    if not ui_components:
        return {}

    # Import dependencies dengan minimal error handling
    try:
        from smartcash.utils.logger import get_logger
        from smartcash.handlers.preprocessing import PreprocessingManager
        from smartcash.utils.observer import EventTopics, notify
        from smartcash.utils.observer.observer_manager import ObserverManager
        from smartcash.utils.config_manager import get_config_manager
        from smartcash.utils.ui_utils import create_status_indicator
        from smartcash.utils.dataset.dataset_utils import DEFAULT_SPLITS
        
        logger = get_logger("preprocessing")
        config_manager = get_config_manager(logger=logger)
        observer_manager = ObserverManager(auto_register=True)
        observer_group = "preprocessing_observers"
        observer_manager.unregister_group(observer_group)
        
        # Setup atau load config
        if not config or 'preprocessing' not in config.get('data', {}):
            config = config_manager.load_config("configs/preprocessing_config.yaml") or {
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
            config_manager.save_config(config, "configs/preprocessing_config.yaml")
    except ImportError as e:
        # Fallback minimal function
        def create_status_indicator(status, message):
            icon = "✅" if status == "success" else "⚠️" if status == "warning" else "❌" if status == "error" else "ℹ️"
            return HTML(f"<div style='margin:5px 0'>{icon} {message}</div>")
        
        if 'preprocess_status' in ui_components:
            with ui_components['preprocess_status']:
                display(HTML(f"<p style='color:red'>⚠️ Limited functionality: {str(e)}</p>"))
        return ui_components
    
    # Shared state variables
    state = {
        'processing': False,
        'stop_requested': False,
        'preprocessing_manager': None,
        'current_split': None
    }
    
    # Inisialisasi PreprocessingManager secara lazy untuk efisiensi
    def get_preprocessing_manager():
        if not state['preprocessing_manager']:
            state['preprocessing_manager'] = PreprocessingManager(config=config, logger=logger)
        return state['preprocessing_manager']
    
    # Update UI untuk kondisi processing
    def update_ui_for_processing(is_processing):
        # Update progress visibility
        for component in ['progress_bar', 'current_progress']:
            if component in ui_components:
                ui_components[component].layout.visibility = 'visible' if is_processing else 'hidden'
        
        # Update button states
        if 'preprocess_button' in ui_components:
            ui_components['preprocess_button'].disabled = is_processing
        if 'stop_button' in ui_components:
            ui_components['stop_button'].layout.display = 'inline-block' if is_processing else 'none'
        if 'cleanup_button' in ui_components:
            ui_components['cleanup_button'].disabled = is_processing
        
        # Show logs during processing
        if 'log_accordion' in ui_components and is_processing:
            ui_components['log_accordion'].selected_index = 0
    
    # Consolidated progress tracking function
    def update_progress(event_type, sender, progress=0, total=100, message=None, split=None, **kwargs):
        """Unified progress handler untuk semua jenis progress."""
        if state['stop_requested']:
            return
        
        # Tentukan bar yang akan diupdate berdasarkan jenis event
        if event_type == EventTopics.PREPROCESSING_PROGRESS:
            # Overall progress bar
            progress_bar = ui_components.get('progress_bar')
            if progress_bar:
                progress_pct = int(progress * 100 / total) if total > 0 else 0
                progress_bar.value = progress_pct
                progress_bar.description = f"{progress_pct}%"
        elif event_type in [EventTopics.VALIDATION_PROGRESS, EventTopics.AUGMENTATION_PROGRESS]:
            # Current operation progress bar
            current_bar = ui_components.get('current_progress')
            if current_bar:
                progress_pct = int(progress * 100 / total) if total > 0 else 0
                current_bar.value = progress_pct
                current_bar.description = f"{progress_pct}%"
                
                # Increment overall progress slightly with each update
                overall_bar = ui_components.get('progress_bar')
                if overall_bar:
                    # Use a slower increment for overall progress
                    increment = min(5, (100 - overall_bar.value) / 10)
                    overall_bar.value = min(95, overall_bar.value + increment)
        
        # Display message jika ada
        if message and 'preprocess_status' in ui_components:
            with ui_components['preprocess_status']:
                # Add split info if available
                split_info = f" ({split})" if split else ""
                display(create_status_indicator("info", f"{message}{split_info}"))
    
    # Setup observers untuk progress tracking
    if observer_manager:
        # Consolidated progress observer
        for event_type in [EventTopics.PREPROCESSING_PROGRESS, 
                           EventTopics.VALIDATION_PROGRESS, 
                           EventTopics.AUGMENTATION_PROGRESS]:
            observer_manager.create_simple_observer(
                event_type=event_type,
                callback=update_progress,
                name=f"ProgressObserver_{event_type}",
                group=observer_group
            )
        
        # Logging observer for main events
        observer_manager.create_logging_observer(
            event_types=[
                EventTopics.PREPROCESSING_START, 
                EventTopics.PREPROCESSING_END, 
                EventTopics.PREPROCESSING_ERROR, 
                EventTopics.VALIDATION_EVENT,
                EventTopics.VALIDATION_START,
                EventTopics.VALIDATION_END
            ],
            logger_name="preprocessing", 
            name="LogObserver", 
            group=observer_group
        )
    
    # Status display helper
    def display_status(status_type, message):
        if 'preprocess_status' in ui_components:
            with ui_components['preprocess_status']:
                display(create_status_indicator(status_type, message))
    
    # Get preprocessing config from UI
    def get_config_from_ui():
        # Baca parameter dari UI
        opts = ui_components['preprocess_options'].children
        img_size = opts[0].value if len(opts) > 0 else [640, 640]
        normalize = opts[1].value if len(opts) > 1 else True
        cache = opts[2].value if len(opts) > 2 else True
        workers = opts[3].value if len(opts) > 3 else 4
        
        # Baca opsi validasi
        v_opts = ui_components['validation_options'].children
        validate = v_opts[0].value if len(v_opts) > 0 else True
        fix_issues = v_opts[1].value if len(v_opts) > 1 else True
        move_invalid = v_opts[2].value if len(v_opts) > 2 else True
        invalid_dir = v_opts[3].value if len(v_opts) > 3 else 'data/invalid'
        
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
            'output_dir': 'data/preprocessed'
        })
        
        # Save config
        if config_manager:
            config_manager.save_config(config, "configs/preprocessing_config.yaml")
        
        return config
    
    # Display preprocessing results summary
    def update_summary(result):
        if 'summary_container' not in ui_components:
            return
            
        with ui_components['summary_container']:
            clear_output()
            
            # Extract stats
            validation_stats = {
                split: result['validation'].get(split, {}).get('validation_stats', {}) 
                for split in DEFAULT_SPLITS 
                if split in result.get('validation', {})
            }
            
            analysis_stats = {
                split: result['analysis'].get(split, {}).get('analysis', {}) 
                for split in DEFAULT_SPLITS 
                if split in result.get('analysis', {})
            }
            
            # Validation stats table
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
            
            # Class distribution
            if analysis_stats:
                display(HTML("<h4>📊 Class Distribution</h4>"))
                
                for split, stats in analysis_stats.items():
                    if 'class_distribution' in stats:
                        class_dist = stats['class_distribution']
                        display(HTML(f"<h5>{split.capitalize()}</h5>"))
                        
                        dist_table = "<table style='width:100%; border-collapse:collapse'><tr style='background:#f2f2f2'>"
                        dist_table += "<th>Class</th><th>Count</th><th>Percentage</th></tr>"
                        
                        total = sum(class_dist.values())
                        for cls, count in class_dist.items():
                            pct = (count / total * 100) if total > 0 else 0
                            dist_table += f"<tr><td>{cls}</td><td>{count}</td><td>{pct:.1f}%</td></tr>"
                        
                        display(HTML(dist_table + "</table>"))
            
            # Execution info
            if 'elapsed' in result:
                display(HTML(f"<p><b>⏱️ Execution time:</b> {result['elapsed']:.2f} seconds</p>"))
            
            output_dir = config.get('data', {}).get('preprocessing', {}).get('output_dir', 'data/preprocessed')
            display(HTML(f"<p><b>📂 Output:</b> {output_dir}</p>"))
        
        # Show summary & cleanup button
        ui_components['summary_container'].layout.display = 'block'
        ui_components['cleanup_button'].layout.display = 'inline-block'
    
    # Fungsi preprocessing utama
    def process_dataset(splits=None):
        """Function to run preprocessing on selected splits."""
        if splits is None:
            # Parse splits from UI
            split_map = {
                'All Splits': DEFAULT_SPLITS,
                'Train Only': [DEFAULT_SPLITS[0]],
                'Validation Only': [DEFAULT_SPLITS[1]],
                'Test Only': [DEFAULT_SPLITS[2]]
            }
            splits = split_map.get(ui_components['split_selector'].value, DEFAULT_SPLITS)
        
        ui_components['progress_bar'].value = 5  # Show initial progress
        
        try:
            # Get preprocessor manager
            manager = get_preprocessing_manager()
            
            # Notifikasi start
            notify(
                event_type=EventTopics.PREPROCESSING_START,
                sender="preprocessing_handler",
                message="Memulai preprocessing dataset"
            )
            
            # Run preprocessing
            result = manager.run_full_pipeline(
                splits=splits,
                validate_dataset=config['data']['preprocessing']['validation']['enabled'],
                fix_issues=config['data']['preprocessing']['validation']['fix_issues'], 
                augment_data=False,  # Tidak perlu augmentasi di preprocessing stage
                analyze_dataset=True
            )
            
            # Process result
            if result['status'] == 'success':
                ui_components['progress_bar'].value = 100  # Complete progress
                display_status("success", f"✅ Preprocessing selesai dalam {result.get('elapsed', 0):.2f} detik")
                update_summary(result)
            else:
                display_status("error", f"❌ Preprocessing gagal: {result.get('error', 'Unknown error')}")
                
            # Notify completion
            notify(
                event_type=EventTopics.PREPROCESSING_END,
                sender="preprocessing_handler",
                result=result
            )
            
            return result
            
        except Exception as e:
            display_status("error", f"❌ Error: {str(e)}")
            notify(
                event_type=EventTopics.PREPROCESSING_ERROR,
                sender="preprocessing_handler",
                error=str(e)
            )
            return {'status': 'error', 'error': str(e)}
    
    # Main preprocessing thread
    def preprocessing_thread():
        try:
            state['processing'] = True
            state['stop_requested'] = False
            
            # Update config from UI
            get_config_from_ui()
            
            # Execute preprocessing
            process_dataset()
            
        except Exception as e:
            display_status("error", f"❌ Unexpected error: {str(e)}")
        finally:
            state['processing'] = False
            state['stop_requested'] = False
            update_ui_for_processing(False)
    
    # Handler for preprocess button
    def on_preprocess_click(b):
        # Check if already processing
        if state['processing']:
            return
            
        # Check if preprocessed dir exists
        preproc_dir = config.get('data', {}).get('preprocessing', {}).get('output_dir', 'data/preprocessed')
        preproc_path = Path(preproc_dir)
        
        if preproc_path.exists() and any(preproc_path.iterdir()):
            # Show confirmation dialog
            confirm_box = widgets.VBox([
                widgets.HTML(
                    f"""<div style="padding:10px; background-color:#fff3cd; color:#856404; 
                    border-left:4px solid #856404; border-radius:4px; margin:10px 0;">
                    <h4 style="margin-top:0;">⚠️ Preprocessed data already exists</h4>
                    <p>Directory <code>{preproc_dir}</code> already contains preprocessed data.</p>
                    <p>Continuing will overwrite existing data. Do you want to proceed?</p>
                    </div>"""
                ),
                widgets.HBox([
                    widgets.Button(description="Cancel", button_style="danger"),
                    widgets.Button(description="Proceed", button_style="primary")
                ])
            ])
            
            with ui_components['preprocess_status']:
                clear_output()
                display(confirm_box)
            
            # Setup confirmation button handlers
            def on_cancel(b):
                with ui_components['preprocess_status']:
                    clear_output()
                    display(create_status_indicator("info", "Operation cancelled"))
                    
            def on_proceed(b):
                with ui_components['preprocess_status']:
                    clear_output()
                    display(create_status_indicator("info", "🔄 Proceeding with preprocessing..."))
                
                # Reset UI elements
                ui_components['progress_bar'].value = 0
                ui_components['current_progress'].value = 0
                ui_components['summary_container'].layout.display = 'none'
                ui_components['cleanup_button'].layout.display = 'none'
                update_ui_for_processing(True)
                
                # Start preprocessing thread
                thread = threading.Thread(target=preprocessing_thread)
                thread.daemon = True
                thread.start()
            
            confirm_box.children[1].children[0].on_click(on_cancel)
            confirm_box.children[1].children[1].on_click(on_proceed)
        else:
            # No confirmation needed, start directly
            ui_components['progress_bar'].value = 0
            ui_components['current_progress'].value = 0
            ui_components['summary_container'].layout.display = 'none'
            ui_components['cleanup_button'].layout.display = 'none'
            update_ui_for_processing(True)
            
            # Start preprocessing thread
            thread = threading.Thread(target=preprocessing_thread)
            thread.daemon = True
            thread.start()
    
    # Handler for stop button
    def on_stop_click(b):
        state['stop_requested'] = True
        display_status("warning", "⚠️ Stopping preprocessing...")
        
        # Update UI after short delay
        def delayed_ui_update():
            time.sleep(0.5)
            state['processing'] = False
            update_ui_for_processing(False)
        
        threading.Thread(target=delayed_ui_update, daemon=True).start()
        
        # Notify stop
        notify(
            event_type=EventTopics.PREPROCESSING_END,
            sender="preprocessing_handler",
            message="Preprocessing stopped by user"
        )
    
    # Handler for cleanup button
    def on_cleanup_click(b):
        preproc_dir = config.get('data', {}).get('preprocessing', {}).get('output_dir', 'data/preprocessed')
        preproc_path = Path(preproc_dir)
        
        # Create confirmation dialog
        confirm_box = widgets.VBox([
            widgets.HTML(
                f"""<div style="padding:10px; background-color:#fff3cd; color:#856404; 
                border-left:4px solid #856404; border-radius:4px; margin:10px 0;">
                <h4 style="margin-top:0;">⚠️ Confirm Data Cleanup</h4>
                <p>This will delete all preprocessed data in <code>{preproc_dir}</code>.</p>
                <p>Original dataset will not be affected. Do you want to proceed?</p>
                </div>"""
            ),
            widgets.HBox([
                widgets.Button(description="Cancel", button_style="warning"),
                widgets.Button(description="Delete Preprocessed Data", button_style="danger")
            ])
        ])
        
        with ui_components['preprocess_status']:
            clear_output()
            display(confirm_box)
        
        # Button handlers
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
                    # Try to backup before delete
                    try:
                        from smartcash.utils.dataset.dataset_utils import DatasetUtils
                        utils = DatasetUtils(config, logger=logger)
                        backup_path = utils.backup_directory(preproc_path, suffix="backup_before_delete")
                        if backup_path:
                            display_status("info", f"📦 Backup created: {backup_path}")
                    except ImportError:
                        pass
                    
                    # Delete preprocessing directory
                    shutil.rmtree(preproc_path)
                    display_status("success", f"✅ Removed: {preproc_dir}")
                    
                    # Hide cleanup button and summary
                    ui_components['cleanup_button'].layout.display = 'none'
                    ui_components['summary_container'].layout.display = 'none'
                else:
                    display_status("info", f"ℹ️ Not found: {preproc_dir}")
            except Exception as e:
                display_status("error", f"❌ Error: {str(e)}")
        
        # Register button handlers
        confirm_box.children[1].children[0].on_click(on_cancel)
        confirm_box.children[1].children[1].on_click(on_confirm_delete)
    
    # Handler for Save Config button
    def on_save_config_click(b):
        get_config_from_ui()  # Updates and saves config
        display_status("success", "✅ Configuration saved")
    
    # Setup UI from config
    def init_ui():
        """Initialize UI elements from config."""
        if 'data' in config and 'preprocessing' in config['data']:
            cfg = config['data']['preprocessing']
            
            # Update preprocess options
            opts = ui_components['preprocess_options'].children
            if len(opts) >= 4:
                if 'img_size' in cfg and isinstance(cfg['img_size'], list) and len(cfg['img_size']) == 2:
                    opts[0].value = cfg['img_size']
                    
                # Update normalize, cache, workers
                for key, idx in [('normalize_enabled', 1), ('cache_enabled', 2), ('num_workers', 3)]:
                    if key in cfg:
                        opts[idx].value = cfg[key]
            
            # Update validation options
            if 'validation' in cfg:
                v_opts = ui_components['validation_options'].children
                if len(v_opts) >= 4:
                    val_cfg = cfg['validation']
                    for key, idx in [('enabled', 0), ('fix_issues', 1), ('move_invalid', 2)]:
                        if key in val_cfg:
                            v_opts[idx].value = val_cfg[key]
                    
                    if 'invalid_dir' in val_cfg:
                        v_opts[3].value = val_cfg['invalid_dir']
    
    # Add Save Config button
    try:
        import ipywidgets as widgets
        save_config_button = widgets.Button(
            description='Save Config',
            button_style='info',
            icon='save'
        )
        save_config_button.on_click(on_save_config_click)
        
        # Add to UI components
        ui_components['save_config_button'] = save_config_button
        
        # Find suitable button container or create one
        button_container = None
        for child in ui_components['ui'].children:
            if isinstance(child, widgets.HBox) and hasattr(child, 'children'):
                if any(btn is ui_components['preprocess_button'] for btn in child.children):
                    button_container = child
                    break
        
        if button_container:
            # Add to existing container
            new_buttons = list(button_container.children) + [save_config_button]
            button_container.children = tuple(new_buttons)
    except ImportError:
        pass  # Can't add the button without ipywidgets
    
    # Register event handlers
    ui_components['preprocess_button'].on_click(on_preprocess_click)
    ui_components['stop_button'].on_click(on_stop_click)
    ui_components['cleanup_button'].on_click(on_cleanup_click)
    
    # Initialize UI from config
    init_ui()
    
    # Cleanup function
    def cleanup():
        """Clean up resources when cell is rerun or notebook is closed."""
        state['stop_requested'] = True
        
        if observer_manager:
            observer_manager.unregister_group(observer_group)
        
        update_ui_for_processing(False)
        
        if logger:
            logger.info("✅ Preprocessing handlers cleaned up")
    
    # Add cleanup to UI components
    ui_components['cleanup'] = cleanup
    
    return ui_components