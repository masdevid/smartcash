"""
File: smartcash/ui_handlers/preprocessing.py
Author: Optimasi dengan expanded logging
Deskripsi: Handler untuk UI preprocessing dataset SmartCash dengan logging yang lebih verbose dan event tracking.
"""

import threading, time, shutil, asyncio
from pathlib import Path
from IPython.display import display, clear_output, HTML

def setup_preprocessing_handlers(ui_components, config=None):
    """Setup handlers untuk UI preprocessing dataset dengan logging yang lebih detail."""
    if not ui_components:
        print("Error: ui_components is None")
        return {}
        
    # Verify critical components exist
    missing_components = []
    for key in ['preprocess_button', 'stop_button', 'preprocess_status']:
        if key not in ui_components or ui_components[key] is None:
            missing_components.append(key)
    
    if missing_components:
        print(f"Error: Missing UI components: {', '.join(missing_components)}")
        return ui_components  # Return unchanged

    # Import dependencies dengan fallback siap pakai
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
        
        # Clean up existing observers
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
            log_event("info", "📝 Konfigurasi default dibuat untuk preprocessing")
    except ImportError as e:
        # Fallback minimal function
        def create_status_indicator(status, message):
            icon = "✅" if status == "success" else "⚠️" if status == "warning" else "❌" if status == "error" else "ℹ️"
            return HTML(f"<div style='margin:5px 0'>{icon} {message}</div>")
        
        if 'preprocess_status' in ui_components:
            with ui_components['preprocess_status']:
                display(HTML(f"<p style='color:red'>⚠️ Fungsi terbatas - package tidak lengkap: {str(e)}</p>"))
        return ui_components
    
    # Shared state variables
    state = {
        'processing': False,
        'stop_requested': False,
        'preprocessing_manager': None,
        'current_split': None,
        'start_time': 0,
        'last_log_time': 0,
        'progress_updates': 0
    }
    
    # Helper untuk logging ke output UI
    def log_event(status, message):
        """Log pesan ke status output dengan format konsisten."""
        try:
            if 'preprocess_status' in ui_components and ui_components['preprocess_status'] is not None:
                timestamp = time.strftime("%H:%M:%S")
                with ui_components['preprocess_status']:
                    display(HTML(
                        f"<div style='padding:5px; margin:3px 0; border-left:3px solid "
                        f"{'#28a745' if status == 'success' else '#dc3545' if status == 'error' else '#ffc107' if status == 'warning' else '#17a2b8'}'>"
                        f"<span style='color:#6c757d; font-size:0.8em'>[{timestamp}]</span> {message}"
                        f"</div>"
                    ))
        except Exception as e:
            # Fallback to console if widget display fails
            print(f"[{status.upper()}] {message} (Log error: {str(e)})")
    
    # Initialize PreprocessingManager secara lazy
    def get_preprocessing_manager():
        if not state['preprocessing_manager']:
            # Get paths
            data_dir = config.get('data_dir', 'data')
            output_dir = config.get('data', {}).get('preprocessing', {}).get('output_dir', 'data/preprocessed')
            
            log_event("info", f"📂 Inisialisasi preprocessing dari: <b>{data_dir}</b> ke <b>{output_dir}</b>")
            
            state['preprocessing_manager'] = PreprocessingManager(config=config, logger=logger)
            log_event("success", "✅ PreprocessingManager berhasil diinisialisasi")
        return state['preprocessing_manager']
    
    # Update UI untuk kondisi processing
    def update_ui_for_processing(is_processing):
        """Update UI untuk status processing dengan safety checks."""
        try:
            # Update progress visibility
            for component in ['progress_bar', 'current_progress']:
                if component in ui_components and ui_components[component] is not None and hasattr(ui_components[component], 'layout'):
                    ui_components[component].layout.visibility = 'visible' if is_processing else 'hidden'
            
            # Update button states
            if 'preprocess_button' in ui_components and ui_components['preprocess_button'] is not None:
                ui_components['preprocess_button'].disabled = is_processing
                
            if 'stop_button' in ui_components and ui_components['stop_button'] is not None and hasattr(ui_components['stop_button'], 'layout'):
                ui_components['stop_button'].layout.display = 'inline-block' if is_processing else 'none'
                
            if 'cleanup_button' in ui_components and ui_components['cleanup_button'] is not None:
                ui_components['cleanup_button'].disabled = is_processing
            
            # Show logs during processing
            if 'log_accordion' in ui_components and ui_components['log_accordion'] is not None and is_processing:
                ui_components['log_accordion'].selected_index = 0
        except Exception as e:
            # Log error but continue
            print(f"Error updating UI state: {str(e)}")

    
    # Progress tracking function with better verbosity
    def update_progress(event_type, sender, progress=0, total=100, message=None, split=None, **kwargs):
        """Enhanced progress handler dengan verbose logging dan safety checks."""
        try:
            if state['stop_requested']:
                return
                
            current_time = time.time()
            if event_type == EventTopics.PREPROCESSING_PROGRESS:
                # Overall progress bar
                progress_bar = ui_components.get('progress_bar')
                if progress_bar is not None:
                    progress_pct = int(progress * 100 / total) if total > 0 else 0
                    progress_bar.value = progress_pct
                    progress_bar.description = f"{progress_pct}%"
                    
                    # Log periodic updates (not too frequent)
                    if current_time - state['last_log_time'] > 2.0 or progress_pct % 20 == 0:
                        elapsed = current_time - state['start_time']
                        log_event("info", f"⏱️ Progress: <b>{progress_pct}%</b> ({progress}/{total}) setelah {elapsed:.1f}s")
                        state['last_log_time'] = current_time
            elif event_type in [EventTopics.VALIDATION_PROGRESS, EventTopics.AUGMENTATION_PROGRESS]:
                # Current operation progress bar
                current_bar = ui_components.get('current_progress')
                if current_bar is not None:
                    progress_pct = int(progress * 100 / total) if total > 0 else 0
                    current_bar.value = progress_pct
                    current_bar.description = f"{progress_pct}%"
                    
                    # Log more detailed updates for current operations
                    state['progress_updates'] += 1
                    if state['progress_updates'] % 5 == 0 or progress_pct % 25 == 0:
                        split_info = f"({split})" if split else ""
                        task_type = "validasi" if event_type == EventTopics.VALIDATION_PROGRESS else "augmentasi"
                        log_event("info", f"🔄 {task_type.capitalize()} {split_info}: <b>{progress_pct}%</b> ({progress}/{total})")
                    
                    # Increment overall progress
                    overall_bar = ui_components.get('progress_bar')
                    if overall_bar is not None:
                        # Use a slower increment for overall progress
                        increment = min(5, (100 - overall_bar.value) / 10)
                        overall_bar.value = min(95, overall_bar.value + increment)
            
            # Always display specific operation messages
            if message and message.strip():
                with ui_components['preprocess_status']:
                    split_info = f" ({split})" if split else ""
                    display(create_status_indicator("info", f"{message}{split_info}"))
        except Exception as e:
            # Fallback to console
            print(f"Error updating progress: {str(e)}")

    
    # Setup observers untuk enhanced progress tracking
    if observer_manager:
        # Setup detailed progress observer
        for event_type in [EventTopics.PREPROCESSING_PROGRESS, 
                          EventTopics.VALIDATION_PROGRESS, 
                          EventTopics.AUGMENTATION_PROGRESS]:
            observer_manager.create_simple_observer(
                event_type=event_type,
                callback=update_progress,
                name=f"ProgressObserver_{event_type}",
                group=observer_group
            )
        
        # Setup specific event observers
        for event_type, icon, desc in [
            (EventTopics.PREPROCESSING_START, "🚀", "Preprocessing dimulai"),
            (EventTopics.PREPROCESSING_END, "✅", "Preprocessing selesai"),
            (EventTopics.PREPROCESSING_ERROR, "❌", "Error preprocessing"),
            (EventTopics.VALIDATION_START, "🔍", "Validasi dimulai"),
            (EventTopics.VALIDATION_END, "✓", "Validasi selesai")
        ]:
            observer_manager.create_simple_observer(
                event_type=event_type,
                callback=lambda event_type, sender, message=None, **kwargs: 
                    log_event(
                        "success" if "END" in str(event_type) else
                        "error" if "ERROR" in str(event_type) else "info", 
                        f"{icon} {message or desc}"
                    ),
                name=f"EventObserver_{event_type}",
                group=observer_group
            )
    
    # Get preprocessing config from UI
    def get_config_from_ui():
        """Baca parameter dari UI dan update config."""
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
            log_event("info", "📝 Konfigurasi preprocessing diperbarui dan disimpan")
        
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
            
            # Log summary completion
            log_event("success", f"✅ Preprocessing selesai dengan {sum(len(stats.get('class_distribution', {})) for split, stats in analysis_stats.items() if 'class_distribution' in stats)} kelas")
        
        # Show summary & cleanup button
        ui_components['summary_container'].layout.display = 'block'
        ui_components['cleanup_button'].layout.display = 'inline-block'
    
    # Heartbeat untuk progress updates
    async def progress_heartbeat():
        """Send periodic progress updates ketika tidak ada event dari observers."""
        while state['processing'] and not state['stop_requested']:
            current_time = time.time()
            elapsed = current_time - state['start_time']
            
            # Tambahkan log tiap 5 detik jika tidak ada update dari progress observer
            if current_time - state['last_log_time'] > 5.0:
                split = state['current_split'] or 'current'
                progress_bar = ui_components.get('progress_bar')
                progress_pct = progress_bar.value if progress_bar else 0
                
                log_event("info", f"⏱️ Masih memproses {split}: {progress_pct}% setelah {elapsed:.1f}s")
                state['last_log_time'] = current_time
                
                # Increment progress bar sedikit untuk menunjukkan masih berjalan
                if progress_bar and progress_pct < 95:
                    progress_bar.value += 1
                
            await asyncio.sleep(5.0)
    
    # Main preprocessing function
    async def process_dataset(splits=None):
        """Async function untuk run preprocessing dengan monitoring yang lebih baik."""
        if splits is None:
            # Parse splits from UI
            split_map = {
                'All Splits': DEFAULT_SPLITS,
                'Train Only': [DEFAULT_SPLITS[0]],
                'Validation Only': [DEFAULT_SPLITS[1]],
                'Test Only': [DEFAULT_SPLITS[2]]
            }
            splits = split_map.get(ui_components['split_selector'].value, DEFAULT_SPLITS)
        
        state['start_time'] = time.time()
        state['last_log_time'] = state['start_time']
        state['progress_updates'] = 0
        
        # Print the splits that will be processed
        log_event("info", f"🔧 Memproses split: <b>{', '.join(splits)}</b>")
        
        # Init progress
        ui_components['progress_bar'].value = 5
        
        # Start heartbeat task for continuous progress updates
        heartbeat_task = asyncio.create_task(progress_heartbeat())
        
        try:
            # Get preprocessor manager
            manager = get_preprocessing_manager()
            
            # Notifikasi start
            notify(
                event_type=EventTopics.PREPROCESSING_START,
                sender="preprocessing_handler",
                message=f"Memulai preprocessing untuk {len(splits)} split"
            )
            
            # Create a thread for the actual processing
            result_event = threading.Event()
            result_container = {}
            
            def run_preprocessing():
                try:
                    for i, split in enumerate(splits):
                        state['current_split'] = split
                        log_event("info", f"🔄 Memproses split: <b>{split}</b> ({i+1}/{len(splits)})")
                    
                    validate = config['data']['preprocessing']['validation']['enabled']
                    fix_issues = config['data']['preprocessing']['validation']['fix_issues']
                    
                    result = manager.run_full_pipeline(
                        splits=splits,
                        validate_dataset=validate,
                        fix_issues=fix_issues, 
                        augment_data=False,
                        analyze_dataset=True
                    )
                    result_container['result'] = result
                except Exception as e:
                    result_container['error'] = str(e)
                    log_event("error", f"❌ Error: {str(e)}")
                finally:
                    result_event.set()
            
            # Start processing thread
            log_event("info", "🚀 Menjalankan preprocessing pipeline...")
            processing_thread = threading.Thread(target=run_preprocessing)
            processing_thread.daemon = True
            processing_thread.start()
            
            # Wait for processing to complete
            while not result_event.is_set() and not state['stop_requested']:
                await asyncio.sleep(0.1)
            
            # Cancel heartbeat
            heartbeat_task.cancel()
            
            if state['stop_requested']:
                log_event("warning", "⚠️ Preprocessing dihentikan oleh user")
                ui_components['progress_bar'].value = 0
                return {'status': 'stopped'}
            
            # Process result
            if 'error' in result_container:
                log_event("error", f"❌ Preprocessing gagal: {result_container['error']}")
                notify(
                    event_type=EventTopics.PREPROCESSING_ERROR,
                    sender="preprocessing_handler",
                    error=result_container['error']
                )
                return {'status': 'error', 'error': result_container['error']}
            
            result = result_container['result']
            if result['status'] == 'success':
                ui_components['progress_bar'].value = 100
                elapsed = time.time() - state['start_time']
                log_event("success", f"✅ Preprocessing selesai dalam {elapsed:.2f} detik")
                update_summary(result)
                
                # Notify completion
                notify(
                    event_type=EventTopics.PREPROCESSING_END,
                    sender="preprocessing_handler",
                    result=result
                )
            else:
                log_event("error", f"❌ Preprocessing gagal: {result.get('error', 'Unknown error')}")
                notify(
                    event_type=EventTopics.PREPROCESSING_ERROR,
                    sender="preprocessing_handler",
                    error=result.get('error', 'Unknown error')
                )
            
            return result
            
        except Exception as e:
            log_event("error", f"❌ Unexpected error: {str(e)}")
            notify(
                event_type=EventTopics.PREPROCESSING_ERROR,
                sender="preprocessing_handler",
                error=str(e)
            )
            return {'status': 'error', 'error': str(e)}
    
    # Main preprocessing thread
    async def preprocessing_thread():
        try:
            state['processing'] = True
            state['stop_requested'] = False
            
            # Update config from UI
            get_config_from_ui()
            
            # Execute preprocessing
            await process_dataset()
            
        except Exception as e:
            log_event("error", f"❌ Unexpected error: {str(e)}")
        finally:
            state['processing'] = False
            state['stop_requested'] = False
            update_ui_for_processing(False)

    # Run async function in thread to avoid blocking
    def run_async_preprocessing():
        """Wrapper to run async preprocessing in a thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(preprocessing_thread())
        finally:
            loop.close()
    
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
                    <p>Directory <code>{preproc_dir}</code> sudah berisi data.</p>
                    <p>Melanjutkan akan menimpa data yang ada. Lanjutkan?</p>
                    </div>"""
                ),
                widgets.HBox([
                    widgets.Button(description="Batal", button_style="danger"),
                    widgets.Button(description="Lanjutkan", button_style="primary")
                ])
            ])
            
            with ui_components['preprocess_status']:
                clear_output()
                display(confirm_box)
            
            # Setup confirmation button handlers
            def on_cancel(b):
                with ui_components['preprocess_status']:
                    clear_output()
                    log_event("info", "Operasi dibatalkan")
                    
            def on_proceed(b):
                with ui_components['preprocess_status']:
                    clear_output()
                    log_event("info", "🔄 Memulai preprocessing...")
                
                # Reset UI elements
                ui_components['progress_bar'].value = 0
                ui_components['current_progress'].value = 0
                ui_components['summary_container'].layout.display = 'none'
                ui_components['cleanup_button'].layout.display = 'none'
                update_ui_for_processing(True)
                
                # Start preprocessing thread
                threading.Thread(target=run_async_preprocessing, daemon=True).start()
            
            confirm_box.children[1].children[0].on_click(on_cancel)
            confirm_box.children[1].children[1].on_click(on_proceed)
        else:
            # No confirmation needed, start directly
            ui_components['progress_bar'].value = 0
            ui_components['current_progress'].value = 0
            ui_components['summary_container'].layout.display = 'none'
            ui_components['cleanup_button'].layout.display = 'none'
            update_ui_for_processing(True)
            
            with ui_components['preprocess_status']:
                clear_output()
                log_event("info", "🚀 Memulai preprocessing dataset...")
            
            # Start preprocessing thread
            threading.Thread(target=run_async_preprocessing, daemon=True).start()
    
    # Handler for stop button
    def on_stop_click(b):
        state['stop_requested'] = True
        log_event("warning", "⚠️ Menghentikan preprocessing...")
        
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
            message="Preprocessing dihentikan oleh user"
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
                <h4 style="margin-top:0;">⚠️ Konfirmasi Pembersihan Data</h4>
                <p>Menghapus semua data preprocessed di <code>{preproc_dir}</code>.</p>
                <p>Dataset asli tidak akan terpengaruh. Lanjutkan?</p>
                </div>"""
            ),
            widgets.HBox([
                widgets.Button(description="Batal", button_style="warning"),
                widgets.Button(description="Hapus Data", button_style="danger")
            ])
        ])
        
        with ui_components['preprocess_status']:
            clear_output()
            display(confirm_box)
        
        # Button handlers
        def on_cancel(b):
            with ui_components['preprocess_status']:
                clear_output()
                log_event("info", "Operasi dibatalkan")
        
        def on_confirm_delete(b):
            with ui_components['preprocess_status']:
                clear_output()
                log_event("info", "🗑️ Membersihkan data preprocessed...")
            
            try:
                if preproc_path.exists():
                    # Try to backup before delete
                    try:
                        from smartcash.utils.dataset.dataset_utils import DatasetUtils
                        utils = DatasetUtils(config, logger=logger)
                        backup_path = utils.backup_directory(preproc_path, suffix="backup_before_delete")
                        if backup_path:
                            log_event("info", f"📦 Backup dibuat: {backup_path}")
                    except ImportError:
                        pass
                    
                    # Delete preprocessing directory
                    shutil.rmtree(preproc_path)
                    log_event("success", f"✅ Dihapus: {preproc_dir}")
                    
                    # Hide cleanup button and summary
                    ui_components['cleanup_button'].layout.display = 'none'
                    ui_components['summary_container'].layout.display = 'none'
                else:
                    log_event("info", f"ℹ️ Direktori tidak ditemukan: {preproc_dir}")
            except Exception as e:
                log_event("error", f"❌ Error: {str(e)}")
        
        # Register button handlers
        confirm_box.children[1].children[0].on_click(on_cancel)
        confirm_box.children[1].children[1].on_click(on_confirm_delete)
    
    # Handler for Save Config button
    def on_save_config_click(b):
        get_config_from_ui()  # Updates and saves config
        log_event("success", "✅ Konfigurasi tersimpan")
    
    # Setup UI from config
    def init_ui():
        """Initialize UI elements from config with robust error handling."""
        try:
            if 'data' in config and 'preprocessing' in config['data']:
                cfg = config['data']['preprocessing']
                
                # Update preprocess options with safety checks
                if 'preprocess_options' in ui_components and ui_components['preprocess_options'] is not None:
                    opts = ui_components['preprocess_options'].children if hasattr(ui_components['preprocess_options'], 'children') else []
                    
                    if len(opts) >= 4:
                        if 'img_size' in cfg and isinstance(cfg['img_size'], list) and len(cfg['img_size']) == 2:
                            opts[0].value = cfg['img_size']
                            
                        # Update normalize, cache, workers
                        for key, idx in [('normalize_enabled', 1), ('cache_enabled', 2), ('num_workers', 3)]:
                            if key in cfg and idx < len(opts):
                                opts[idx].value = cfg[key]
                
                # Update validation options with safety checks
                if 'validation' in cfg and 'validation_options' in ui_components and ui_components['validation_options'] is not None:
                    v_opts = ui_components['validation_options'].children if hasattr(ui_components['validation_options'], 'children') else []
                    
                    if len(v_opts) >= 4:
                        val_cfg = cfg['validation']
                        for key, idx in [('enabled', 0), ('fix_issues', 1), ('move_invalid', 2)]:
                            if key in val_cfg and idx < len(v_opts):
                                v_opts[idx].value = val_cfg[key]
                        
                        if 'invalid_dir' in val_cfg and 3 < len(v_opts):
                            v_opts[3].value = val_cfg['invalid_dir']
            
            # Display directories
            data_dir = config.get('data_dir', 'data')
            output_dir = config.get('data', {}).get('preprocessing', {}).get('output_dir', 'data/preprocessed')
            
            if 'preprocess_status' in ui_components and ui_components['preprocess_status'] is not None:
                with ui_components['preprocess_status']:
                    log_event("info", f"📁 Data: <b>{data_dir}</b> → Preprocessing: <b>{output_dir}</b>")
                
            # Check if preprocessed directory already exists
            preproc_path = Path(output_dir)
            if preproc_path.exists() and any(preproc_path.iterdir()):
                if 'cleanup_button' in ui_components and ui_components['cleanup_button'] is not None:
                    ui_components['cleanup_button'].layout.display = 'inline-block'
                if 'preprocess_status' in ui_components and ui_components['preprocess_status'] is not None:
                    with ui_components['preprocess_status']:
                        log_event("info", f"💡 Preprocessed data terdeteksi: {output_dir}")
            elif 'cleanup_button' in ui_components and ui_components['cleanup_button'] is not None:
                ui_components['cleanup_button'].layout.display = 'none'
        except Exception as e:
            # Log error but continue
            if 'preprocess_status' in ui_components and ui_components['preprocess_status'] is not None:
                with ui_components['preprocess_status']:
                    log_event("error", f"❌ Error initializing UI: {str(e)}")
            print(f"Error in init_ui: {str(e)}")  # Fallback console log