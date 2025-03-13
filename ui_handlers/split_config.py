"""
File: smartcash/ui_handlers/split_config.py
Author: Refactor
Deskripsi: Handler untuk UI konfigurasi split dataset SmartCash (optimized).
"""

import threading
from IPython.display import display, HTML, clear_output
from pathlib import Path


def setup_split_config_handlers(ui_components, config=None):
    """Setup handlers untuk UI konfigurasi split dataset."""
    # Import dependencies yang diperlukan
    try:
        from smartcash.utils.logger import get_logger
        from smartcash.handlers.dataset import DatasetManager
        from smartcash.utils.observer import notify
        from smartcash.utils.observer.observer_manager import ObserverManager
        from smartcash.utils.config_manager import get_config_manager
        from smartcash.utils.dataset.dataset_utils import DatasetUtils, DEFAULT_SPLITS
        from smartcash.utils.ui_utils import create_status_indicator
        
        logger = get_logger("split_config")
        dataset_manager = DatasetManager(config, logger=logger)
        observer_manager = ObserverManager(auto_register=True)
        config_manager = get_config_manager(logger=logger)
        dataset_utils = DatasetUtils(config=config, logger=logger)
        
        # Bersihkan observer grup
        observer_group = "split_observers"
        observer_manager.unregister_group(observer_group)
    except ImportError as e:
        from smartcash.utils.ui_utils import create_status_indicator
        with ui_components['split_status']:
            display(create_status_indicator("warning", f"Beberapa modul tidak tersedia: {str(e)}"))
        return ui_components
    
    # Fungsi untuk mendapatkan statistik dataset saat ini
    def update_current_stats_display():
        """Update tampilan statistik dataset saat ini."""
        with ui_components['current_stats']:
            clear_output(wait=True)
            
            try:
                # Gunakan DatasetUtils untuk mendapatkan statistik
                stats = dataset_utils.get_split_statistics(DEFAULT_SPLITS)
                
                # Hitung total dan persentase
                total_images = sum(split.get('images', 0) for split in stats.values())
                
                # Create stats HTML
                html = """
                <h3 style="margin-top:0; color:#2c3e50">📊 Current Dataset Statistics</h3>
                <div style="display:flex; flex-wrap:wrap; gap:15px; margin-bottom:10px">
                """
                
                for split, data in stats.items():
                    images = data.get('images', 0)
                    labels = data.get('labels', 0)
                    status = data.get('status', 'valid')
                    percentage = (images / total_images * 100) if total_images > 0 else 0
                    
                    # Pick color based on status
                    color = "#3498db" if status == 'valid' else "#e74c3c"
                    
                    html += f"""
                    <div style="flex:1; min-width:150px; border:1px solid {color}; border-radius:5px; padding:10px; background-color:#f8f9fa">
                        <h4 style="margin-top:0; color:{color}">{split.capitalize()}</h4>
                        <p style="margin:5px 0; color:#2c3e50"><strong>Images:</strong> {images}</p>
                        <p style="margin:5px 0; color:#2c3e50"><strong>Labels:</strong> {labels}</p>
                        <p style="margin:5px 0; color:#2c3e50"><strong>Percentage:</strong> {percentage:.1f}%</p>
                    </div>
                    """
                
                html += "</div>"
                
                if total_images == 0:
                    html += """
                    <div style="padding:10px; background-color:#fff3cd; border-left:4px solid #856404; color:#856404">
                        <p style="margin:0">⚠️ Dataset tidak ditemukan atau kosong. Silakan download dataset terlebih dahulu.</p>
                    </div>
                    """
                
                display(HTML(html))
                
            except Exception as e:
                logger.error(f"❌ Error saat mendapatkan statistik dataset: {str(e)}")
                display(HTML(f"""
                    <div style="padding:10px; background-color:#f8d7da; border-left:4px solid #721c24; color:#721c24">
                        <p style="margin:0">❌ Error saat mendapatkan statistik dataset: {str(e)}</p>
                    </div>
                """))
    
    # Function untuk update UI dari config
    def update_ui_from_config():
        """Update UI sliders dan options dari config yang ada."""
        if 'data' in config:
            # Update split ratios
            split_ratios = config['data'].get('split_ratios', {'train': 0.7, 'valid': 0.15, 'test': 0.15})
            
            ui_components['split_sliders'][0].value = split_ratios.get('train', 0.7) * 100
            ui_components['split_sliders'][1].value = split_ratios.get('valid', 0.15) * 100
            ui_components['split_sliders'][2].value = split_ratios.get('test', 0.15) * 100
            
            # Update stratified checkbox
            ui_components['stratified'].value = config['data'].get('stratified_split', True)
            
            # Update advanced options
            if 'random_seed' in config['data']:
                ui_components['advanced_options'].children[0].value = config['data']['random_seed']
    
    # Function to display split results
    def display_split_results(stats, result_stats=None):
        """Display the results after splitting."""
        with ui_components['stats_output']:
            clear_output()
            
            # Ensure stats_output is visible
            ui_components['stats_output'].layout.display = 'block'
            
            # Create HTML summary
            html = f"""
            <h3 style="margin-top:0; color:#2c3e50">✅ Split Results</h3>
            <div style="display:flex; flex-wrap:wrap; gap:15px; margin-bottom:15px">
            """
            
            # Show stats for each split
            for split, data in stats.items():
                images = data.get('images', 0)
                labels = data.get('labels', 0)
                status = data.get('status', 'valid')
                
                color = "#2ecc71" if status == 'valid' else "#e74c3c"
                
                html += f"""
                <div style="flex:1; min-width:150px; border:1px solid {color}; border-radius:5px; padding:10px; background-color:#f8f9fa">
                    <h4 style="margin-top:0; color:{color}">{split.capitalize()}</h4>
                    <p style="margin:5px 0; color:#2c3e50"><strong>Images:</strong> {images}</p>
                    <p style="margin:5px 0; color:#2c3e50"><strong>Labels:</strong> {labels}</p>
                </div>
                """
            
            html += "</div>"
            
            # Add additional result information if available
            if result_stats:
                if 'moved_files' in result_stats:
                    html += f"""
                    <div style="padding:10px; background-color:#d1ecf1; border-left:4px solid #0c5460; color:#0c5460; margin-bottom:15px">
                        <p style="margin:0">ℹ️ Files moved: {result_stats['moved_files']}</p>
                    </div>
                    """
                
                if 'class_distribution' in result_stats:
                    html += """
                    <div style="padding:10px; background-color:#d4edda; border-left:4px solid #155724; color:#155724">
                        <p style="margin:0">✅ Class distribution balanced across splits</p>
                    </div>
                    """
            
            display(HTML(html))
    
    # Function for applying split configuration
    def on_apply_split(b):
        """Handler for the Apply Split button."""
        # Get values from UI
        train_pct = ui_components['split_sliders'][0].value
        val_pct = ui_components['split_sliders'][1].value
        test_pct = ui_components['split_sliders'][2].value
        stratified = ui_components['stratified'].value
        
        # Get advanced options
        advanced = ui_components['advanced_options']
        random_seed = advanced.children[0].value
        force_resplit = advanced.children[1].value
        preserve_structure = advanced.children[2].value
        backup_dataset = advanced.children[3].value
        backup_dir = advanced.children[4].value
        
        # Start split thread to avoid blocking UI
        def split_thread():
            try:
                with ui_components['split_status']:
                    clear_output()
                    display(create_status_indicator("info", "🔄 Memulai proses split dataset..."))
                
                # Normalize percentages to ensure they sum to 100%
                total = train_pct + val_pct + test_pct
                if abs(total - 100.0) > 0.001:
                    norm_train = train_pct * 100 / total
                    norm_val = val_pct * 100 / total
                    norm_test = test_pct * 100 / total
                    
                    with ui_components['split_status']:
                        display(create_status_indicator(
                            "warning", 
                            f"⚠️ Persentase disesuaikan: Train {norm_train:.1f}%, Valid {norm_val:.1f}%, Test {norm_test:.1f}%"
                        ))
                else:
                    norm_train, norm_val, norm_test = train_pct, val_pct, test_pct
                
                # Create ratios dict and update config
                split_ratios = {
                    'train': norm_train / 100.0,
                    'valid': norm_val / 100.0,
                    'test': norm_test / 100.0
                }
                
                # Update the config
                config['data']['split_ratios'] = split_ratios
                config['data']['stratified_split'] = stratified
                config['data']['random_seed'] = random_seed
                
                # Notify of split start
                notify(
                    event_type="dataset.split.start",
                    sender="split_config_handler",
                    train_pct=norm_train,
                    val_pct=norm_val,
                    test_pct=norm_test,
                    stratified=stratified
                )
                
                with ui_components['split_status']:
                    # Backup if requested
                    if backup_dataset:
                        display(create_status_indicator("info", f"📦 Membuat backup dataset ke {backup_dir}..."))
                        
                        try:
                            dataset_utils.backup_directory(
                                config['data'].get('dir', 'data'),
                                suffix=f"presplit_{threading._thread.get_ident()}"
                            )
                        except Exception as e:
                            display(create_status_indicator("warning", f"⚠️ Gagal backup dataset: {str(e)}"))
                    
                    # Execute split
                    display(create_status_indicator("info", "🔄 Menjalankan split dataset..."))
                    
                    result = dataset_manager.split_dataset(
                        ratios=split_ratios,
                        stratified=stratified,
                        random_seed=random_seed,
                        force_resplit=force_resplit,
                        preserve_structure=preserve_structure
                    )
                    
                    if result and result.get('success', False):
                        display(create_status_indicator(
                            "success", 
                            f"✅ Split berhasil dengan rasio: Train {norm_train:.1f}%, Valid {norm_val:.1f}%, Test {norm_test:.1f}%"
                        ))
                        
                        # Update stats after split
                        try:
                            new_stats = dataset_utils.get_split_statistics(DEFAULT_SPLITS)
                            display_split_results(new_stats, result.get('stats', {}))
                        except Exception as e:
                            logger.error(f"Error getting stats: {str(e)}")
                            display_split_results({'train': {}, 'valid': {}, 'test': {}})
                        
                        # Save config after successful split
                        config_manager.save_config(config)
                        
                        # Notification event
                        notify(
                            event_type="dataset.split.end",
                            sender="split_config_handler",
                            success=True,
                            stats=result.get('stats', {})
                        )
                    else:
                        display(create_status_indicator(
                            "warning", 
                            f"⚠️ Split selesai dengan issues: {result.get('message', 'unknown error')}"
                        ))
                        
                        # Notification event
                        notify(
                            event_type="dataset.split.end",
                            sender="split_config_handler",
                            success=False,
                            message=result.get('message', 'unknown error')
                        )
            except Exception as e:
                with ui_components['split_status']:
                    display(create_status_indicator("error", f"❌ Error: {str(e)}"))
                
                # Notification event
                notify(
                    event_type="dataset.split.error",
                    sender="split_config_handler",
                    error=str(e)
                )
        
        # Start thread
        thread = threading.Thread(target=split_thread)
        thread.daemon = True
        thread.start()
    
    # Handler for reset button
    def on_reset_defaults(b):
        """Reset UI to default values."""
        # Default values
        default_config = {
            'data': {
                'split_ratios': {'train': 0.7, 'valid': 0.15, 'test': 0.15},
                'stratified_split': True,
                'random_seed': 42
            }
        }
        
        # Update config with defaults
        for key, value in default_config['data'].items():
            config['data'][key] = value
        
        # Update UI
        update_ui_from_config()
        
        with ui_components['split_status']:
            clear_output()
            display(create_status_indicator("info", "🔄 Reset ke pengaturan default"))
    
    # Handler for save config button
    def on_save_config(b):
        """Save current configuration."""
        # Get current values from UI
        train_pct = ui_components['split_sliders'][0].value
        val_pct = ui_components['split_sliders'][1].value
        test_pct = ui_components['split_sliders'][2].value
        stratified = ui_components['stratified'].value
        random_seed = ui_components['advanced_options'].children[0].value
        
        # Update config
        config['data']['split_ratios'] = {
            'train': train_pct / 100.0,
            'valid': val_pct / 100.0,
            'test': test_pct / 100.0
        }
        config['data']['stratified_split'] = stratified
        config['data']['random_seed'] = random_seed
        
        # Save config
        config_manager.save_config(config)
        
        with ui_components['split_status']:
            clear_output()
            display(create_status_indicator("success", "✅ Konfigurasi berhasil disimpan"))
    
    # Register handlers
    ui_components['split_button'].on_click(on_apply_split)
    ui_components['reset_button'].on_click(on_reset_defaults)
    ui_components['save_button'].on_click(on_save_config)
    
    # Register observer untuk progress
    observer_manager.create_logging_observer(
        event_types=["dataset.split.start", "dataset.split.end", "dataset.split.error"],
        logger_name="split_config",
        name="SplitLogObserver",
        group=observer_group
    )
    
    # Function untuk cleanup
    def cleanup():
        """Cleanup resources."""
        observer_manager.unregister_group(observer_group)
        logger.info("✅ Observer untuk split dataset telah dibersihkan")
    
    # Tambahkan cleanup ke ui_components
    ui_components['cleanup'] = cleanup
    
    # Inisialisasi tampilan
    update_ui_from_config()
    update_current_stats_display()
    
    return ui_components