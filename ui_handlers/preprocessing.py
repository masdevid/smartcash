"""
File: smartcash/ui_handlers/preprocessing.py
Author: Alfrida Sabar (refactored)
Deskripsi: Handler untuk UI preprocessing dataset SmartCash dengan pendekatan simpel dan robust
"""

import threading
import time
import shutil
from pathlib import Path
from IPython.display import display, clear_output, HTML

from smartcash.utils.ui_utils import create_status_indicator

def setup_preprocessing_handlers(ui_components, config=None):
    """Setup handlers untuk UI preprocessing dataset."""
    # Inisialisasi dependencies
    logger = None
    preprocessing_manager = None
    observer_manager = None
    config_manager = None
    dataset_utils = None
    
    # Flag untuk tracking state
    is_processing = False
    stop_requested = False
    
    try:
        from smartcash.utils.logger import get_logger
        from smartcash.handlers.preprocessing import PreprocessingManager
        from smartcash.utils.observer import EventTopics, EventDispatcher
        from smartcash.utils.observer.observer_manager import ObserverManager
        from smartcash.utils.config_manager import get_config_manager
        from smartcash.utils.dataset.dataset_utils import DatasetUtils
        
        logger = get_logger("preprocessing")
        observer_manager = ObserverManager(auto_register=True)
        config_manager = get_config_manager(logger=logger)
        
        # Clean up existing observers
        observer_manager.unregister_group("preprocessing_observers")
        
        # Load config
        if not config or 'preprocessing' not in config:
            config = config_manager.load_config("configs/preprocessing_config.yaml")
        
        # Initialize managers
        preprocessing_manager = PreprocessingManager(config=config, logger=logger)
        dataset_utils = DatasetUtils(config=config, logger=logger)
        
    except ImportError as e:
        with ui_components['preprocess_status']:
            display(create_status_indicator("warning", f"⚠️ Beberapa modul tidak tersedia: {str(e)}"))
    
    # Helper untuk update progress dalam UI
    def update_progress(progress=0, total=100, message=None, progress_type="main"):
        """Update UI progress dengan tangguh terhadap error."""
        try:
            if progress_type == "main":
                # Update main progress bar
                progress_bar = ui_components.get('progress_bar')
                if progress_bar is not None:
                    progress_pct = min(100, int(progress * 100 / total) if total > 0 else 0)
                    progress_bar.value = progress_pct
                    progress_bar.description = f"{progress_pct}%"
            else:
                # Update current operation progress
                current_bar = ui_components.get('current_progress')
                if current_bar is not None:
                    progress_pct = min(100, int(progress * 100 / total) if total > 0 else 0)
                    current_bar.value = progress_pct
                    current_bar.description = f"{progress_pct}%"
            
            # Display message if provided
            if message and message.strip():
                with ui_components['preprocess_status']:
                    display(create_status_indicator("info", message))
        except Exception as e:
            if logger:
                logger.warning(f"⚠️ Error updating progress: {str(e)}")
    
    # Register observer untuk progress updates
    if observer_manager:
        # Register observers untuk tipe event yang berbeda
        observer_manager.create_simple_observer(
            event_type=EventTopics.PREPROCESSING_PROGRESS,
            callback=lambda _, __, progress=0, total=100, message=None, **kwargs: 
                update_progress(progress, total, message, "main"),
            name="PreprocessingProgressObserver",
            group="preprocessing_observers"
        )
        
        observer_manager.create_simple_observer(
            event_type=EventTopics.VALIDATION_PROGRESS,
            callback=lambda _, __, progress=0, total=100, message=None, **kwargs: 
                update_progress(progress, total, message, "current"),
            name="ValidationProgressObserver",
            group="preprocessing_observers"
        )
        
        # Observer untuk pesan log
        observer_manager.create_logging_observer(
            event_types=[
                EventTopics.PREPROCESSING_START,
                EventTopics.PREPROCESSING_END,
                EventTopics.PREPROCESSING_ERROR
            ],
            log_level="info",
            name="PreprocessingLogObserver",
            group="preprocessing_observers"
        )
    
    # Helper untuk update UI state
    def update_ui_state(processing=False):
        """Update UI berdasarkan state processing."""
        nonlocal is_processing
        is_processing = processing
        
        if 'preprocess_button' in ui_components:
            ui_components['preprocess_button'].disabled = processing
        
        if 'stop_button' in ui_components:
            ui_components['stop_button'].layout.display = 'inline-block' if processing else 'none'
        
        if 'cleanup_button' in ui_components:
            ui_components['cleanup_button'].disabled = processing
        
        # Show/hide progress
        for comp in ['progress_bar', 'current_progress']:
            if comp in ui_components:
                ui_components[comp].layout.visibility = 'visible' if processing else 'hidden'
        
        # Show logs
        if processing and 'log_accordion' in ui_components:
            ui_components['log_accordion'].selected_index = 0
    
    # Update config dari UI
    def get_config_from_ui():
        """Extract config dari UI components."""
        try:
            # Basic options
            if 'preprocess_options' in ui_components and len(ui_components['preprocess_options'].children) >= 4:
                opts = ui_components['preprocess_options'].children
                img_size = opts[0].value
                normalize = opts[1].value
                cache = opts[2].value
                workers = opts[3].value
                
                # Validation options
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
                
                # Save config if manager available
                if config_manager:
                    config_manager.save_config(config, "configs/preprocessing_config.yaml")
                    
                    with ui_components['preprocess_status']:
                        display(create_status_indicator("info", "📝 Konfigurasi diperbarui"))
        except Exception as e:
            if logger:
                logger.warning(f"⚠️ Error getting config from UI: {str(e)}")
    
    # Handler untuk preprocessing
    def on_preprocess_click(b):
        """Handler untuk tombol preprocessing."""
        nonlocal stop_requested
        
        if is_processing:
            return
        
        # Reset state
        stop_requested = False
        
        # Check if output dir exists and has content
        preproc_dir = config.get('data', {}).get('preprocessing', {}).get('output_dir', 'data/preprocessed')
        preproc_path = Path(preproc_dir)
        
        if preproc_path.exists() and any(preproc_path.iterdir()):
            # Show confirmation
            confirm_html = f"""
            <div style="padding:10px; background-color:#fff3cd; color:#856404; 
                 border-left:4px solid #856404; border-radius:4px; margin:10px 0;">
                <h4 style="margin-top:0;">⚠️ Preprocessed data sudah ada</h4>
                <p>Directory <code>{preproc_dir}</code> sudah berisi data.</p>
                <p>Melanjutkan akan menimpa data yang ada. Lanjutkan?</p>
            </div>
            """
            
            # Buat buttons
            confirm_button = ui_components.get('_confirm_button')
            cancel_button = ui_components.get('_cancel_button')
            
            if not confirm_button or not cancel_button:
                import ipywidgets as widgets
                
                confirm_button = widgets.Button(
                    description="Lanjutkan",
                    button_style="primary",
                    layout=widgets.Layout(margin='5px')
                )
                
                cancel_button = widgets.Button(
                    description="Batal",
                    button_style="danger",
                    layout=widgets.Layout(margin='5px')
                )
                
                ui_components['_confirm_button'] = confirm_button
                ui_components['_cancel_button'] = cancel_button
            
            # Setup confirmation dialog
            with ui_components['preprocess_status']:
                clear_output()
                display(HTML(confirm_html))
                display(ui_components['_cancel_button'])
                display(ui_components['_confirm_button'])
            
            # Button handlers
            ui_components['_cancel_button'].on_click(
                lambda b: confirm_cancelled()
            )
            
            ui_components['_confirm_button'].on_click(
                lambda b: start_preprocessing()
            )
        else:
            # No confirmation needed
            start_preprocessing()
    
    # Helper functions untuk confirmation
    def confirm_cancelled():
        """Handle pembatalan konfirmasi."""
        with ui_components['preprocess_status']:
            clear_output()
            display(create_status_indicator("info", "ℹ️ Operasi dibatalkan"))
    
    # Function utama untuk preprocessing
    def start_preprocessing():
        """Mulai proses preprocessing."""
        nonlocal is_processing, stop_requested
        
        # Update UI for processing
        update_ui_state(True)
        
        # Reset UI elements
        ui_components['progress_bar'].value = 0
        ui_components['current_progress'].value = 0
        ui_components['summary_container'].layout.display = 'none'
        
        with ui_components['preprocess_status']:
            clear_output()
            display(create_status_indicator("info", "🚀 Memulai preprocessing dataset..."))
        
        # Get split selection
        split_map = {
            'All Splits': ['train', 'valid', 'test'],
            'Train Only': ['train'],
            'Validation Only': ['valid'],
            'Test Only': ['test']
        }
        splits = split_map.get(ui_components['split_selector'].value, ['train', 'valid', 'test'])
        
        # Update config from UI
        get_config_from_ui()
        
        # Start processing in a thread
        def processing_thread():
            try:
                # Notify preprocessing start
                if observer_manager:
                    EventDispatcher.notify(
                        event_type=EventTopics.PREPROCESSING_START,
                        sender="preprocessing_handler",
                        message=f"Memulai preprocessing untuk splits: {', '.join(splits)}"
                    )
                
                # Run full pipeline
                validate = config['data']['preprocessing']['validation']['enabled']
                fix_issues = config['data']['preprocessing']['validation']['fix_issues']
                
                start_time = time.time()
                for i, split in enumerate(splits):
                    if stop_requested:
                        break
                        
                    with ui_components['preprocess_status']:
                        display(create_status_indicator("info", f"🔄 Memproses split: {split} ({i+1}/{len(splits)})"))
                    
                    # Update progress
                    ui_components['progress_bar'].value = (i / len(splits)) * 100
                    
                # Run full pipeline if manager available
                if preprocessing_manager and not stop_requested:
                    # Log start
                    with ui_components['preprocess_status']:
                        display(create_status_indicator("info", "🔄 Menjalankan pipeline preprocessing..."))
                    
                    # Run pipeline
                    result = preprocessing_manager.run_full_pipeline(
                        splits=splits,
                        validate_dataset=validate,
                        fix_issues=fix_issues,
                        augment_data=False,
                        analyze_dataset=True
                    )
                    
                    # Process result
                    elapsed = time.time() - start_time
                    
                    if result['status'] == 'success':
                        # Update progress to 100%
                        ui_components['progress_bar'].value = 100
                        
                        with ui_components['preprocess_status']:
                            display(create_status_indicator("success", f"✅ Preprocessing selesai dalam {elapsed:.2f} detik"))
                        
                        # Show summary
                        update_summary(result)
                        
                        # Notify completion
                        if observer_manager:
                            EventDispatcher.notify(
                                event_type=EventTopics.PREPROCESSING_END,
                                sender="preprocessing_handler",
                                result=result,
                                elapsed=elapsed
                            )
                    else:
                        with ui_components['preprocess_status']:
                            display(create_status_indicator("error", f"❌ Preprocessing gagal: {result.get('error', 'Unknown error')}"))
                        
                        # Notify error
                        if observer_manager:
                            EventDispatcher.notify(
                                event_type=EventTopics.PREPROCESSING_ERROR,
                                sender="preprocessing_handler",
                                error=result.get('error', 'Unknown error')
                            )
                else:
                    with ui_components['preprocess_status']:
                        display(create_status_indicator("error", "❌ PreprocessingManager tidak tersedia"))
            except Exception as e:
                with ui_components['preprocess_status']:
                    display(create_status_indicator("error", f"❌ Error: {str(e)}"))
                
                # Notify error
                if observer_manager:
                    EventDispatcher.notify(
                        event_type=EventTopics.PREPROCESSING_ERROR,
                        sender="preprocessing_handler",
                        error=str(e)
                    )
            finally:
                # Reset state
                update_ui_state(False)
        
        # Start thread
        thread = threading.Thread(target=processing_thread)
        thread.daemon = True
        thread.start()
    
    # Update summary after preprocessing
    def update_summary(result):
        """Update summary display with preprocessing results."""
        if 'summary_container' not in ui_components:
            return
        
        with ui_components['summary_container']:
            clear_output()
            
            # Extract stats
            validation_stats = {}
            analysis_stats = {}
            
            for split in ['train', 'valid', 'test']:
                if split in result.get('validation', {}):
                    validation_stats[split] = result['validation'][split].get('validation_stats', {})
                
                if split in result.get('analysis', {}):
                    analysis_stats[split] = result['analysis'][split].get('analysis', {})
            
            # Show validation stats
            if validation_stats:
                display(HTML("<h3>📊 Hasil Preprocessing</h3><h4>🔍 Validation</h4>"))
                
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
                display(HTML("<h4>📊 Distribusi Kelas</h4>"))
                
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
                display(HTML(f"<p><b>⏱️ Waktu eksekusi:</b> {result['elapsed']:.2f} detik</p>"))
            
            output_dir = config.get('data', {}).get('preprocessing', {}).get('output_dir', 'data/preprocessed')
            display(HTML(f"<p><b>📂 Output:</b> {output_dir}</p>"))
        
        # Show summary and cleanup button
        ui_components['summary_container'].layout.display = 'block'
        ui_components['cleanup_button'].layout.display = 'inline-block'
    
    # Handler for stop button
    def on_stop_click(b):
        """Handler untuk tombol stop."""
        nonlocal stop_requested
        stop_requested = True
        
        with ui_components['preprocess_status']:
            display(create_status_indicator("warning", "⚠️ Menghentikan preprocessing..."))
        
        # Notify stop
        if observer_manager:
            EventDispatcher.notify(
                event_type=EventTopics.PREPROCESSING_END,
                sender="preprocessing_handler",
                message="Preprocessing dihentikan oleh user"
            )
    
    # Handler for cleanup button
    def on_cleanup_click(b):
        """Handler untuk tombol cleanup."""
        if is_processing:
            return
        
        preproc_dir = config.get('data', {}).get('preprocessing', {}).get('output_dir', 'data/preprocessed')
        
        # Create confirmation dialog
        confirm_html = f"""
        <div style="padding:10px; background-color:#fff3cd; color:#856404; 
             border-left:4px solid #856404; border-radius:4px; margin:10px 0;">
            <h4 style="margin-top:0;">⚠️ Konfirmasi Pembersihan</h4>
            <p>Menghapus semua data preprocessed di <code>{preproc_dir}</code>.</p>
            <p>Data asli tidak akan terpengaruh. Lanjutkan?</p>
        </div>
        """
        
        # Create buttons
        import ipywidgets as widgets
        
        confirm_button = widgets.Button(
            description="Hapus Data",
            button_style="danger",
            layout=widgets.Layout(margin='5px')
        )
        
        cancel_button = widgets.Button(
            description="Batal",
            button_style="warning",
            layout=widgets.Layout(margin='5px')
        )
        
        # Display confirmation
        with ui_components['preprocess_status']:
            clear_output()
            display(HTML(confirm_html))
            display(cancel_button)
            display(confirm_button)
        
        # Button handlers
        def on_cancel(b):
            with ui_components['preprocess_status']:
                clear_output()
                display(create_status_indicator("info", "Operasi dibatalkan"))
        
        def on_confirm(b):
            with ui_components['preprocess_status']:
                clear_output()
                display(create_status_indicator("info", "🗑️ Membersihkan data..."))
            
            try:
                preproc_path = Path(preproc_dir)
                
                if preproc_path.exists():
                    # Try backup first
                    try:
                        if dataset_utils:
                            backup_path = dataset_utils.backup_directory(
                                preproc_path, 
                                suffix="backup_before_delete"
                            )
                            
                            with ui_components['preprocess_status']:
                                display(create_status_indicator("info", f"📦 Backup dibuat: {backup_path}"))
                    except Exception as e:
                        with ui_components['preprocess_status']:
                            display(create_status_indicator("warning", f"⚠️ Tidak bisa membuat backup: {str(e)}"))
                    
                    # Delete directory
                    shutil.rmtree(preproc_path)
                    
                    with ui_components['preprocess_status']:
                        display(create_status_indicator("success", f"✅ Data preprocessed berhasil dihapus"))
                    
                    # Hide summary and cleanup button
                    ui_components['summary_container'].layout.display = 'none'
                    ui_components['cleanup_button'].layout.display = 'none'
                else:
                    with ui_components['preprocess_status']:
                        display(create_status_indicator("warning", f"⚠️ Direktori tidak ditemukan: {preproc_dir}"))
            except Exception as e:
                with ui_components['preprocess_status']:
                    display(create_status_indicator("error", f"❌ Error: {str(e)}"))
        
        # Register handlers
        cancel_button.on_click(on_cancel)
        confirm_button.on_click(on_confirm)
    
    # Initialize UI dari config
    def init_ui():
        """Initialize UI dari config yang ada."""
        try:
            if 'data' in config and 'preprocessing' in config['data']:
                preproc_config = config['data']['preprocessing']
                
                # Preprocess options
                if 'preprocess_options' in ui_components and len(ui_components['preprocess_options'].children) >= 4:
                    opts = ui_components['preprocess_options'].children
                    
                    if 'img_size' in preproc_config:
                        opts[0].value = preproc_config['img_size']
                    
                    if 'normalize_enabled' in preproc_config:
                        opts[1].value = preproc_config['normalize_enabled']
                    
                    if 'cache_enabled' in preproc_config:
                        opts[2].value = preproc_config['cache_enabled']
                    
                    if 'num_workers' in preproc_config:
                        opts[3].value = preproc_config['num_workers']
                
                # Validation options
                if 'validation' in preproc_config and 'validation_options' in ui_components and len(ui_components['validation_options'].children) >= 4:
                    v_opts = ui_components['validation_options'].children
                    val_config = preproc_config['validation']
                    
                    if 'enabled' in val_config:
                        v_opts[0].value = val_config['enabled']
                    
                    if 'fix_issues' in val_config:
                        v_opts[1].value = val_config['fix_issues']
                    
                    if 'move_invalid' in val_config:
                        v_opts[2].value = val_config['move_invalid']
                    
                    if 'invalid_dir' in val_config:
                        v_opts[3].value = val_config['invalid_dir']
            
            # Check if preprocessed directory exists and contains files
            output_dir = config.get('data', {}).get('preprocessing', {}).get('output_dir', 'data/preprocessed')
            output_path = Path(output_dir)
            
            if output_path.exists() and any(output_path.iterdir()):
                ui_components['cleanup_button'].layout.display = 'inline-block'
            else:
                ui_components['cleanup_button'].layout.display = 'none'
                
            with ui_components['preprocess_status']:
                display(create_status_indicator(
                    "info", 
                    f"ℹ️ Siap untuk preprocessing dataset dari {config.get('data_dir', 'data')} ke {output_dir}"
                ))
                
        except Exception as e:
            if logger:
                logger.warning(f"⚠️ Error initializing UI: {str(e)}")
    
    # Register UI handlers
    ui_components['preprocess_button'].on_click(on_preprocess_click)
    ui_components['stop_button'].on_click(on_stop_click)
    ui_components['cleanup_button'].on_click(on_cleanup_click)
    
    # Initialize UI
    init_ui()
    
    # Function for cleanup
    def cleanup():
        """Cleanup resources when notebook cell is rerun."""
        if observer_manager:
            observer_manager.unregister_group("preprocessing_observers")
    
    # Add cleanup to ui_components
    ui_components['cleanup'] = cleanup
    
    return ui_components