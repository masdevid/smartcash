"""
File: smartcash/ui_handlers/preprocessing.py
Author: Alfrida Sabar (refactored)
Deskripsi: Handler untuk UI preprocessing dataset SmartCash tanpa threading dan selaras dengan manager.py
"""

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
    
    try:
        from smartcash.utils.logger import get_logger
        from smartcash.handlers.preprocessing import PreprocessingManager
        from smartcash.utils.observer import EventTopics
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
        
        # Initialize manager directly without thread
        preprocessing_manager = PreprocessingManager(config=config, logger=logger)
        dataset_utils = DatasetUtils(config=config, logger=logger)
        
    except ImportError as e:
        with ui_components['preprocess_status']:
            display(create_status_indicator("warning", f"⚠️ Beberapa modul tidak tersedia: {str(e)}"))
    
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
    
    # Update summary with preprocessing results
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
    
    # Handler untuk start preprocessing
    def start_preprocessing():
        """Proses utama preprocessing langsung tanpa thread."""
        nonlocal is_processing
        
        # Update UI for processing
        update_ui_state(True)
        
        # Reset UI elements
        ui_components['progress_bar'].value = 0
        ui_components['current_progress'].value = 0
        ui_components['summary_container'].layout.display = 'none'
        
        # Get split selection
        split_map = {
            'All Splits': ['train', 'valid', 'test'],
            'Train Only': ['train'],
            'Validation Only': ['valid'],
            'Test Only': ['test']
        }
        splits = split_map.get(ui_components['split_selector'].value, ['train', 'valid', 'test'])
        
        with ui_components['preprocess_status']:
            clear_output()
            display(create_status_indicator("info", f"🚀 Memulai preprocessing untuk splits: {', '.join(splits)}"))
        
        # Register observers directly - match with those in manager.py
        if observer_manager:
            # Setup logging observer exactly like in manager.py
            observer_manager.create_logging_observer(
                event_types=[EventTopics.PREPROCESSING_START, EventTopics.PREPROCESSING_END, EventTopics.PREPROCESSING_ERROR],
                log_level="debug",
                name="PreprocessingLogObserver",
                group="preprocessing_observers"
            )
            
            # Progress observer
            observer_manager.create_simple_observer(
                event_type=EventTopics.PREPROCESSING_PROGRESS,
                callback=lambda _, __, progress=0, total=100, **kwargs: update_progress_ui(progress, total),
                name="PreprocessingProgressObserver",
                group="preprocessing_observers"
            )
        
        try:
            # Run preprocessing directly
            if preprocessing_manager:
                validate = True
                fix_issues = True
                
                if 'validation_options' in ui_components and len(ui_components['validation_options'].children) >= 2:
                    validate = ui_components['validation_options'].children[0].value
                    fix_issues = ui_components['validation_options'].children[1].value
                
                # Run full pipeline directly without thread
                result = preprocessing_manager.run_full_pipeline(
                    splits=splits,
                    validate_dataset=validate,
                    fix_issues=fix_issues,
                    augment_data=False,  # No augmentation as per requirement
                    analyze_dataset=True
                )
                
                # Process result
                if result['status'] == 'success':
                    # Update progress to 100%
                    ui_components['progress_bar'].value = 100
                    
                    with ui_components['preprocess_status']:
                        display(create_status_indicator("success", f"✅ Preprocessing selesai dalam {result['elapsed']:.2f} detik"))
                    
                    # Show summary
                    update_summary(result)
                else:
                    with ui_components['preprocess_status']:
                        display(create_status_indicator("error", f"❌ Preprocessing gagal: {result.get('error', 'Unknown error')}"))
            else:
                with ui_components['preprocess_status']:
                    display(create_status_indicator("error", "❌ PreprocessingManager tidak tersedia"))
        except Exception as e:
            with ui_components['preprocess_status']:
                display(create_status_indicator("error", f"❌ Error: {str(e)}"))
        finally:
            # Reset UI state
            update_ui_state(False)
    
    # Helper function for progress updates
    def update_progress_ui(progress, total):
        """Update progress bars."""
        if 'progress_bar' in ui_components:
            progress_pct = min(100, int(progress * 100 / total) if total > 0 else 0)
            ui_components['progress_bar'].value = progress_pct
            ui_components['progress_bar'].description = f"{progress_pct}%"
    
    # Handler untuk preprocessing button
    def on_preprocess_click(b):
        """Handler untuk tombol preprocessing."""
        if is_processing:
            return
        
        # Check if output dir exists and has content
        preproc_dir = config.get('data', {}).get('preprocessing', {}).get('output_dir', 'data/preprocessed')
        preproc_path = Path(preproc_dir)
        
        if preproc_path.exists() and any(preproc_path.iterdir()):
            # Show confirmation dialog
            confirm_html = f"""
            <div style="padding:10px; background-color:#fff3cd; color:#856404; 
                 border-left:4px solid #856404; border-radius:4px; margin:10px 0;">
                <h4 style="margin-top:0;">⚠️ Preprocessed data sudah ada</h4>
                <p>Directory <code>{preproc_dir}</code> sudah berisi data.</p>
                <p>Melanjutkan akan menimpa data yang ada. Lanjutkan?</p>
            </div>
            """
            
            # Create buttons for confirmation
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
            
            # Define button handlers
            def on_cancel(b):
                with ui_components['preprocess_status']:
                    clear_output()
                    display(create_status_indicator("info", "ℹ️ Operasi dibatalkan"))
            
            def on_confirm(b):
                with ui_components['preprocess_status']:
                    clear_output()
                # Update config from UI
                get_config_from_ui()
                # Run preprocessing
                start_preprocessing()
            
            # Setup confirmation dialog
            with ui_components['preprocess_status']:
                clear_output()
                display(HTML(confirm_html))
                display(cancel_button)
                display(confirm_button)
            
            # Register button handlers
            cancel_button.on_click(on_cancel)
            confirm_button.on_click(on_confirm)
        else:
            # No confirmation needed, start directly
            get_config_from_ui()
            start_preprocessing()
    
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
        
        # Define handlers
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
                    # Try to backup before delete
                    try:
                        if dataset_utils:
                            backup_path = dataset_utils.backup_directory(
                                preproc_path, 
                                suffix="backup_before_delete"
                            )
                            
                            if backup_path:
                                with ui_components['preprocess_status']:
                                    display(create_status_indicator("info", f"📦 Backup dibuat: {backup_path}"))
                    except Exception as e:
                        with ui_components['preprocess_status']:
                            display(create_status_indicator("warning", f"⚠️ Tidak bisa membuat backup: {str(e)}"))
                    
                    # Delete preprocessing directory
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
        
        # Display confirmation
        with ui_components['preprocess_status']:
            clear_output()
            display(HTML(confirm_html))
            display(cancel_button)
            display(confirm_button)
        
        # Register button handlers
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