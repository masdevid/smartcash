"""
File: smartcash/ui_handlers/preprocessing.py
Author: Alfrida Sabar (refactored)
Deskripsi: Handler untuk UI preprocessing dataset SmartCash.
"""

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import os, sys, time
from pathlib import Path

def setup_preprocessing_handlers(ui_components, config=None):
    """Setup handlers untuk UI preprocessing dataset."""
    # Default config jika tidak disediakan
    if config is None:
        config = {
            'data': {
                'preprocessing': {
                    'img_size': [640, 640],
                    'cache_dir': '.cache/smartcash',
                    'num_workers': 4,
                    'augmentation_enabled': True,
                    'normalize_enabled': True,
                    'cache_enabled': True,
                    'cache': {
                        'max_size_gb': 1.0,
                        'ttl_hours': 24,
                        'auto_cleanup': True,
                        'cleanup_interval_mins': 30
                    }
                }
            },
            'data_dir': 'data'
        }
    
    # Setup akses ke komponen UI
    preprocess_options = ui_components['preprocess_options']
    cache_settings = ui_components['cache_settings']
    preprocess_button = ui_components['preprocess_button']
    preprocess_progress = ui_components['preprocess_progress']
    preprocess_status = ui_components['preprocess_status']
    
    # Setup logger dan DatasetManager jika tersedia
    dataset_manager = None
    logger = None
    try:
        from smartcash.utils.logger import get_logger
        from smartcash.handlers.dataset import DatasetManager
        from smartcash.utils.observer import EventDispatcher, EventTopics
        
        logger = get_logger("preprocessing")
        dataset_manager = DatasetManager(config, logger=logger)
    except ImportError as e:
        print(f"ℹ️ Beberapa modul tidak tersedia: {str(e)}")
    
    # Handler untuk tombol preprocessing
    def on_preprocess_click(b):
        # Expand logs accordion untuk menampilkan progress
        for widget_id, widget in ui_components.items():
            if widget_id == 'preprocess_status' and hasattr(widget, 'parent') and isinstance(widget.parent, widgets.Accordion):
                widget.parent.selected_index = 0
            elif widget_id == 'log_accordion' and isinstance(widget, widgets.Accordion):
                widget.selected_index = 0
        
        with preprocess_status:
            clear_output()
            display(create_status_indicator("info", "🔄 Memulai preprocessing dataset..."))
            
            try:
                # Ambil preprocessing options dari form
                img_size = preprocess_options.children[0].value
                normalize = preprocess_options.children[1].value
                cache = preprocess_options.children[2].value
                workers = preprocess_options.children[3].value
                
                # Ambil cache settings
                max_size_gb = cache_settings.children[0].value
                ttl_hours = cache_settings.children[1].value
                auto_cleanup = cache_settings.children[2].value
                
                # Update config
                if config and 'data' in config and 'preprocessing' in config['data']:
                    config['data']['preprocessing']['img_size'] = list(img_size)
                    config['data']['preprocessing']['normalize_enabled'] = normalize
                    config['data']['preprocessing']['cache_enabled'] = cache
                    config['data']['preprocessing']['num_workers'] = workers
                    config['data']['preprocessing']['cache']['max_size_gb'] = max_size_gb
                    config['data']['preprocessing']['cache']['ttl_hours'] = ttl_hours
                    config['data']['preprocessing']['cache']['auto_cleanup'] = auto_cleanup
                
                # Notification event
                if 'EventDispatcher' in locals():
                    EventDispatcher.notify(
                        event_type=EventTopics.PREPROCESSING_START,
                        sender="preprocessing_handler",
                        img_size=img_size,
                        normalize=normalize,
                        cache=cache,
                        workers=workers
                    )
                
                # Tampilkan progress bar
                preprocess_progress.layout.visibility = 'visible'
                preprocess_progress.value = 0
                
                # Gunakan DatasetManager jika tersedia
                if dataset_manager:
                    display(create_status_indicator("info", "⚙️ Menggunakan DatasetManager untuk preprocessing..."))
                    
                    try:
                        # Setup observer untuk tracking progress
                        if 'EventDispatcher' in locals():
                            def update_progress(sender, progress, total, message=None, **kwargs):
                                preprocess_progress.max = total
                                preprocess_progress.value = progress
                                preprocess_progress.description = f"{int(progress * 100 / total)}%"
                                
                                if message:
                                    display(create_status_indicator("info", message))
                            
                            EventDispatcher.register(EventTopics.PREPROCESSING_PROGRESS, update_progress)
                        
                        # Jalankan preprocessing
                        result = dataset_manager.preprocess_dataset(
                            img_size=img_size,
                            normalize=normalize,
                            cache=cache,
                            num_workers=workers,
                            max_cache_size_gb=max_size_gb,
                            cache_ttl_hours=ttl_hours,
                            auto_cleanup=auto_cleanup
                        )
                        
                        # Unregister observer
                        if 'EventDispatcher' in locals():
                            EventDispatcher.unregister(EventTopics.PREPROCESSING_PROGRESS, update_progress)
                        
                        # Tampilkan hasil
                        if result and result.get('success', False):
                            display(create_status_indicator(
                                "success", 
                                f"✅ Preprocessing selesai: {result.get('num_images', 0)} gambar"
                            ))
                            
                            # Tampilkan statistik preprocessing
                            if 'stats' in result:
                                stats = result['stats']
                                stats_html = f"""
                                <div style="background-color: #f8f9fa; padding: 10px; color: black; border-radius: 5px; margin-top: 10px;">
                                    <h4>📊 Preprocessing Stats</h4>
                                    <ul>
                                        <li><b>Images processed:</b> {stats.get('num_images', 'N/A')}</li>
                                        <li><b>Average processing time:</b> {stats.get('avg_time_ms', 'N/A')} ms/image</li>
                                        <li><b>Cache size:</b> {stats.get('cache_size_mb', 'N/A')} MB</li>
                                        <li><b>Image size:</b> {img_size[0]}x{img_size[1]}</li>
                                    </ul>
                                </div>
                                """
                                display(HTML(stats_html))
                        else:
                            display(create_status_indicator(
                                "warning", 
                                f"⚠️ Preprocessing selesai dengan beberapa issues: {result.get('message', 'unknown error')}"
                            ))
                    
                    except Exception as e:
                        display(create_status_indicator("error", f"❌ Error dari DatasetManager: {str(e)}"))
                
                else:
                    # Simulasi preprocessing jika DatasetManager tidak tersedia
                    display(create_status_indicator("info", "ℹ️ DatasetManager tidak tersedia, melakukan simulasi preprocessing..."))
                    
                    # Simulasi progres
                    total_steps = 5
                    for i in range(total_steps + 1):
                        preprocess_progress.value = int(i * 100 / total_steps)
                        preprocess_progress.description = f"{int(i * 100 / total_steps)}%"
                        
                        # Pesan berdasarkan tahap
                        if i == 1:
                            display(create_status_indicator("info", "🔍 Scanning dataset..."))
                        elif i == 2:
                            display(create_status_indicator("info", "📏 Resizing images..."))
                        elif i == 3:
                            display(create_status_indicator("info", "🧮 Normalizing pixel values..."))
                        elif i == 4:
                            display(create_status_indicator("info", "💾 Setting up cache..."))
                        
                        time.sleep(0.5)  # Simulasi delay
                    
                    # Tampilkan sukses
                    display(create_status_indicator(
                        "success", 
                        f"✅ Preprocessing selesai dengan {workers} workers, img_size={img_size}"
                    ))
                
                # Notification event
                if 'EventDispatcher' in locals():
                    EventDispatcher.notify(
                        event_type=EventTopics.PREPROCESSING_END,
                        sender="preprocessing_handler",
                        img_size=img_size,
                        normalize=normalize,
                        cache=cache,
                        workers=workers
                    )
                
            except Exception as e:
                display(create_status_indicator("error", f"❌ Error: {str(e)}"))
                
                # Notification event
                if 'EventDispatcher' in locals():
                    EventDispatcher.notify(
                        event_type=EventTopics.PREPROCESSING_ERROR,
                        sender="preprocessing_handler",
                        error=str(e)
                    )
            
            finally:
                # Sembunyikan progress bar
                preprocess_progress.layout.visibility = 'hidden'
    
    # Register handler
    preprocess_button.on_click(on_preprocess_click)
    
    # Inisialisasi dari config
    if config and 'data' in config and 'preprocessing' in config['data']:
        # Update input fields dari config
        preproc_config = config['data']['preprocessing']
        
        # Setup image size slider
        if 'img_size' in preproc_config and isinstance(preproc_config['img_size'], list) and len(preproc_config['img_size']) == 2:
            preprocess_options.children[0].value = preproc_config['img_size']
        
        # Setup checkboxes
        preprocess_options.children[1].value = preproc_config.get('normalize_enabled', True)
        preprocess_options.children[2].value = preproc_config.get('cache_enabled', True)
        
        # Setup worker slider
        preprocess_options.children[3].value = preproc_config.get('num_workers', 4)
        
        # Setup cache settings
        if 'cache' in preproc_config:
            cache_config = preproc_config['cache']
            cache_settings.children[0].value = cache_config.get('max_size_gb', 1.0)
            cache_settings.children[1].value = cache_config.get('ttl_hours', 24)
            cache_settings.children[2].value = cache_config.get('auto_cleanup', True)
    
    return ui_components

def create_status_indicator(status, message):
    """Buat indikator status dengan styling konsisten."""
    status_styles = {
        'success': {'icon': '✅', 'color': 'green'},
        'warning': {'icon': '⚠️', 'color': 'orange'},
        'error': {'icon': '❌', 'color': 'red'},
        'info': {'icon': 'ℹ️', 'color': 'blue'}
    }
    
    style = status_styles.get(status, status_styles['info'])
    
    status_html = f"""
    <div style="margin: 5px 0; padding: 8px 12px; 
                border-radius: 4px; background-color: #f8f9fa;">
        <span style="color: {style['color']}; font-weight: bold;"> 
            {style['icon']} {message}
        </span>
    </div>
    """
    
    return HTML(status_html)