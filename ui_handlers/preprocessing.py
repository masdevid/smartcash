"""
File: smartcash/ui_handlers/preprocessing.py
Author: Alfrida Sabar (optimized)
Deskripsi: Handler untuk UI preprocessing dataset SmartCash dengan pendekatan robust dan modular.
"""

import asyncio  # Added for async support
import time
from pathlib import Path
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import shutil

def setup_preprocessing_handlers(ui_components, config=None):
    """Setup handlers untuk UI preprocessing dataset."""
    if not ui_components:
        return {}

    # Import dependencies
    try:
        from smartcash.utils.logger import get_logger
        from smartcash.handlers.preprocessing import PreprocessingManager
        from smartcash.utils.observer import EventTopics, EventDispatcher
        from smartcash.utils.observer.observer_manager import ObserverManager
        from smartcash.utils.environment_manager import EnvironmentManager
        from smartcash.utils.config_manager import get_config_manager
        from smartcash.utils.ui_utils import create_status_indicator
        
        logger = get_logger("preprocessing")
        env_manager = EnvironmentManager(logger=logger)
        observer_manager = ObserverManager(auto_register=True)
        observer_group = "preprocessing_observers"
        observer_manager.unregister_group(observer_group)
        config_manager = get_config_manager(logger=logger)
        
        # Setup or load config
        if not config or len(config) == 0 or 'preprocessing' not in config.get('data', {}):
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
        # Fallback minimal status indicator
        def create_status_indicator(status, message):
            icon = "✅" if status == "success" else "⚠️" if status == "warning" else "❌" if status == "error" else "ℹ️"
            return HTML(f"<div style='margin:5px 0'>{icon} {message}</div>")
        
        if 'preprocess_status' in ui_components:
            with ui_components['preprocess_status']:
                display(HTML(f"<p style='color:red'>⚠️ Limited functionality: {str(e)}</p>"))
        return ui_components
    
    # State variables
    processing_active = False
    stop_requested = False
    
    # Initialize PreprocessingManager
    data_dir = env_manager.get_path(config.get('data_dir', 'data')) if env_manager else config.get('data_dir', 'data')
    preprocessing_manager = PreprocessingManager(config=config, logger=logger, base_dir=data_dir)
    
    # Setup UI from config
    def init_ui():
        if 'data' in config and 'preprocessing' in config['data']:
            cfg = config['data']['preprocessing']
            
            # Update preprocess_options
            opts = ui_components['preprocess_options'].children
            if len(opts) >= 4:
                if 'img_size' in cfg and isinstance(cfg['img_size'], list) and len(cfg['img_size']) == 2:
                    opts[0].value = cfg['img_size']
                for key, idx in [('normalize_enabled', 1), ('cache_enabled', 2), ('num_workers', 3)]:
                    if key in cfg: opts[idx].value = cfg[key]
            
            # Update validation_options
            if 'validation' in cfg:
                v_opts = ui_components['validation_options'].children
                if len(v_opts) >= 4:
                    for key, idx in [('enabled', 0), ('fix_issues', 1), ('move_invalid', 2)]:
                        if key in cfg['validation']: v_opts[idx].value = cfg['validation'][key]
                    if 'invalid_dir' in cfg['validation']:
                        v_opts[3].value = cfg['validation']['invalid_dir']
    
    # Update UI for processing state
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
    
    # Progress observer callback
    def update_progress(_, __, progress=0, total=100, message=None, **kwargs):
        if stop_requested:
            return
        
        if 'progress_bar' in ui_components:
            bar = ui_components['progress_bar']
            progress_pct = int(progress * 100 / total) if total > 0 else 0
            bar.value = progress_pct
            bar.description = f"{progress_pct}%"
        
        if message and 'preprocess_status' in ui_components:
            with ui_components['preprocess_status']:
                display(create_status_indicator("info", message))
    
    # Display preprocessing results summary
    def update_summary(result):
        if 'summary_container' not in ui_components:
            return
            
        with ui_components['summary_container']:
            clear_output()
            
            # Extract stats
            validation_stats = {
                split: result['validation'].get(split, {}).get('validation_stats', {}) 
                for split in ['train', 'valid', 'test'] 
                if split in result.get('validation', {})
            }
            
            analysis_stats = {
                split: result['analysis'].get(split, {}).get('analysis', {}) 
                for split in ['train', 'valid', 'test'] 
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
    
    # Setup observers
    if observer_manager:
        for event, callback, name in [
            (EventTopics.PREPROCESSING_PROGRESS, update_progress, "ProgressObserver"),
            (EventTopics.VALIDATION_PROGRESS, 
             lambda _, __, progress=0, total=100, **kwargs: 
                 setattr(ui_components['current_progress'], 'value', int(progress * 100 / total) if total > 0 else 0) or
                 setattr(ui_components['current_progress'], 'description', f"{int(progress * 100 / total)}%" if total > 0 else "0%"),
             "CurrentProgressObserver")
        ]:
            observer_manager.create_simple_observer(
                event_type=event, 
                callback=callback, 
                name=name, 
                group=observer_group
            )
        
        # Add logging observer
        observer_manager.create_logging_observer(
            event_types=[
                EventTopics.PREPROCESSING_START, 
                EventTopics.PREPROCESSING_END, 
                EventTopics.PREPROCESSING_ERROR, 
                EventTopics.VALIDATION_EVENT
            ],
            logger_name="preprocessing", 
            name="LogObserver", 
            group=observer_group
        )
    
    # Confirmation dialog for overwriting existing data
    def check_preprocessed_exists():
        preproc_dir = config.get('data', {}).get('preprocessing', {}).get('output_dir', 'data/preprocessed')
        preproc_path = env_manager.get_path(preproc_dir) if env_manager else Path(preproc_dir)
        
        if preproc_path.exists() and any(preproc_path.iterdir()):
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
            
            confirm_box.children[1].children[0].on_click(lambda b: display_status("info", "Operation cancelled"))
            confirm_box.children[1].children[1].on_click(lambda b: (
                display_status("info", "🔄 Proceeding with preprocessing..."),
                start_preprocessing()
            ))
            
            return True
        return False
    
    # Status display helper
    def display_status(status_type, message):
        if 'preprocess_status' in ui_components:
            with ui_components['preprocess_status']:
                clear_output()
                display(create_status_indicator(status_type, message))
    
    # Get preprocessing config from UI
    def get_config_from_ui():
        # Get preprocessing parameters
        opts = ui_components['preprocess_options'].children
        img_size = opts[0].value if len(opts) > 0 else [640, 640]
        normalize = opts[1].value if len(opts) > 1 else True
        cache = opts[2].value if len(opts) > 2 else True
        workers = opts[3].value if len(opts) > 3 else 4
        
        # Get validation options
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
        
        return config
    
    # Save current configuration 
    def save_config():
        updated_config = get_config_from_ui()
        if config_manager:
            config_manager.save_config(updated_config, "configs/preprocessing_config.yaml", sync_to_drive=True)
            display_status("success", "✅ Configuration saved to configs/preprocessing_config.yaml")
    
    async def start_preprocessing():
        nonlocal processing_active, stop_requested  # Ensure these variables are accessible
        
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
            if 'EventDispatcher' in globals():
                EventDispatcher.notify(
                    event_type=EventTopics.PREPROCESSING_START, 
                    sender="preprocessing_handler",
                    message="Starting preprocessing pipeline"
                )
            
            if not stop_requested:
                # Run pipeline asynchronously
                result = await asyncio.to_thread(
                    preprocessing_manager.run_full_pipeline,
                    splits=splits, 
                    validate_dataset=config['data']['preprocessing']['validation']['enabled'],
                    fix_issues=config['data']['preprocessing']['validation']['fix_issues'], 
                    augment_data=False, 
                    analyze_dataset=True
                )
                
                # Process result
                if result['status'] == 'success':
                    display_status("success", f"✅ Preprocessing completed in {result.get('elapsed', 0):.2f} seconds")
                    update_summary(result)
                    if 'EventDispatcher' in globals():
                        EventDispatcher.notify(
                            event_type=EventTopics.PREPROCESSING_END, 
                            sender="preprocessing_handler",
                            result=result
                        )
                else:
                    display_status("error", f"❌ Preprocessing failed: {result.get('error', 'Unknown error')}")
                    if 'EventDispatcher' in globals():
                        EventDispatcher.notify(
                            event_type=EventTopics.PREPROCESSING_ERROR, 
                            sender="preprocessing_handler",
                            error=result.get('error', 'Unknown error')
                        )
        except Exception as e:
            display_status("error", f"❌ Error: {str(e)}")
            if 'EventDispatcher' in globals():
                EventDispatcher.notify(
                    event_type=EventTopics.PREPROCESSING_ERROR, 
                    sender="preprocessing_handler", 
                    error=str(e)
                )
        finally:
            processing_active = False  # Ensure this is always set
            stop_requested = False
            update_ui_for_processing(False)

    async def on_preprocess_click(b):
        nonlocal processing_active  # Ensure this variable is accessible
        
        if check_preprocessed_exists():
            pass  # Confirmation dialog will handle starting if confirmed
        else:
            if processing_active:
                return
            
            processing_active = True  # Initialize the variable
            ui_components['progress_bar'].value = 0
            ui_components['current_progress'].value = 0
            ui_components['summary_container'].layout.display = 'none'
            ui_components['cleanup_button'].layout.display = 'none'
            update_ui_for_processing(True)
            display_status("info", "🔄 Starting preprocessing...")
            await start_preprocessing()
    
    # Handler for stop button
    def on_stop_click(b):
        nonlocal stop_requested, processing_active
        stop_requested = True
        display_status("warning", "⚠️ Stopping preprocessing...")
        
        # Immediate UI update
        processing_active = False
        update_ui_for_processing(False)
        
        if 'EventDispatcher' in globals():
            EventDispatcher.notify(
                event_type=EventTopics.PREPROCESSING_END, 
                sender="preprocessing_handler",
                message="Preprocessing stopped by user"
            )
    
    # Handler for cleanup button
    def on_cleanup_click(b):
        preproc_dir = config.get('data', {}).get('preprocessing', {}).get('output_dir', 'data/preprocessed')
        preproc_path = env_manager.get_path(preproc_dir) if env_manager else Path(preproc_dir)
        
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
        
        # Cancel button handler
        confirm_box.children[1].children[0].on_click(
            lambda b: display_status("info", "Operation cancelled")
        )
        
        # Confirm delete button handler
        def on_confirm_delete(b):
            display_status("info", "🗑️ Cleaning preprocessed data...")
            
            try:
                if preproc_path.exists():
                    # Create backup
                    try:
                        from smartcash.utils.dataset.dataset_utils import DatasetUtils
                        utils = DatasetUtils(config, logger=logger)
                        backup_path = utils.backup_directory(preproc_path, suffix="backup_before_delete")
                        if backup_path:
                            with ui_components['preprocess_status']:
                                display(create_status_indicator("info", f"📦 Backup created: {backup_path}"))
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
                
        confirm_box.children[1].children[1].on_click(on_confirm_delete)
    
    # Add Save Config button
    save_config_button = widgets.Button(
        description='Save Config',
        button_style='info',
        icon='save'
    )
    save_config_button.on_click(lambda b: save_config())
    
    # Add Save Config button to UI components
    ui_components['save_config_button'] = save_config_button
    
    # Find suitable button container and add save_config_button
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
    else:
        # Create new container
        new_button_container = widgets.HBox([
            ui_components['preprocess_button'], 
            ui_components['stop_button'],
            save_config_button
        ])
        
        # Find location to insert
        for i, child in enumerate(ui_components['ui'].children):
            if hasattr(child, 'children') and ui_components['preprocess_button'] in child.children:
                new_children = list(ui_components['ui'].children)
                new_children[i] = new_button_container
                ui_components['ui'].children = tuple(new_children)
                break
    
    # Register event handlers
    ui_components['preprocess_button'].on_click(lambda b: asyncio.create_task(on_preprocess_click(b)))
    ui_components['stop_button'].on_click(on_stop_click)
    ui_components['cleanup_button'].on_click(on_cleanup_click)
    
    # Initialize UI from config
    init_ui()
    
    # Cleanup function
    def cleanup():
        nonlocal stop_requested
        stop_requested = True
        
        if observer_manager:
            observer_manager.unregister_group(observer_group)
        
        update_ui_for_processing(False)
        
        if logger:
            logger.info("✅ Preprocessing handlers cleaned up")
    
    ui_components['cleanup'] = cleanup
    
    return ui_components