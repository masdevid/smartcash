"""
File: smartcash/ui_handlers/preprocessing.py
Author: Alfrida Sabar (refactored)
Deskripsi: Handler untuk UI preprocessing dataset SmartCash yang terintegrasi dengan PreprocessingManager.
"""

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import os, sys
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
                    'normalize_enabled': True,
                    'cache_enabled': True
                }
            },
            'data_dir': 'data'
        }
    
    # Setup akses ke komponen UI
    preprocess_options = ui_components['preprocess_options']
    preprocess_button = ui_components['preprocess_button']
    preprocess_progress = ui_components['preprocess_progress']
    preprocess_status = ui_components['preprocess_status']
    
    # Setup logger dan PreprocessingManager jika tersedia
    preprocessing_manager = None
    logger = None
    
    try:
        from smartcash.utils.logger import get_logger
        from smartcash.handlers.preprocessing import PreprocessingManager
        from smartcash.utils.observer import EventDispatcher, EventTopics
        
        logger = get_logger("preprocessing")
        preprocessing_manager = PreprocessingManager(config=config, logger=logger)
        
        if logger:
            logger.info("✅ PreprocessingManager berhasil diinisialisasi")
    except ImportError as e:
        print(f"ℹ️ Beberapa modul tidak tersedia: {str(e)}")
    
    # Handler untuk tombol preprocessing
    def on_preprocess_click(b):
        # Expand logs accordion untuk menampilkan progress
        if 'log_accordion' in ui_components:
            ui_components['log_accordion'].selected_index = 0
        
        with preprocess_status:
            clear_output()
            display(create_status_indicator("info", "🔄 Memulai preprocessing dataset..."))
            
            try:
                # Ambil preprocessing options dari form
                img_size = preprocess_options.children[0].value
                normalize = preprocess_options.children[1].value
                enable_cache = preprocess_options.children[2].value
                workers = preprocess_options.children[3].value
                
                # Update config
                if config and 'data' in config and 'preprocessing' in config['data']:
                    config['data']['preprocessing']['img_size'] = list(img_size)
                    config['data']['preprocessing']['normalize_enabled'] = normalize
                    config['data']['preprocessing']['cache_enabled'] = enable_cache
                    config['data']['preprocessing']['num_workers'] = workers
                
                # Tampilkan progress bar
                preprocess_progress.layout.visibility = 'visible'
                preprocess_progress.value = 0
                
                # Setup observer untuk progress tracking
                if 'EventDispatcher' in locals():
                    def update_progress(sender, progress, total, message=None, **kwargs):
                        preprocess_progress.max = total
                        preprocess_progress.value = progress
                        preprocess_progress.description = f"{int(progress * 100 / total)}%" if total > 0 else "0%"
                        
                        if message:
                            display(create_status_indicator("info", message))
                    
                    EventDispatcher.register(EventTopics.PREPROCESSING_PROGRESS, update_progress)
                
                # Gunakan PreprocessingManager 
                if preprocessing_manager:
                    display(create_status_indicator("info", "⚙️ Menggunakan PreprocessingManager untuk preprocessing..."))
                    
                    try:
                        # Jalankan preprocessing pipeline
                        result = preprocessing_manager.run_full_pipeline(
                            splits=['train', 'valid', 'test'],
                            validate_dataset=True,
                            fix_issues=False,
                            augment_data=False,
                            analyze_dataset=True
                        )
                        
                        # Unregister observer jika ada
                        if 'EventDispatcher' in locals():
                            EventDispatcher.unregister(EventTopics.PREPROCESSING_PROGRESS, update_progress)
                        
                        # Tampilkan hasil
                        if result and result.get('status') == 'success':
                            display(create_status_indicator(
                                "success", 
                                f"✅ Preprocessing pipeline selesai dalam {result.get('elapsed', 0):.2f} detik"
                            ))
                            
                            # Tampilkan statistik preprocessing jika tersedia
                            validation_stats = result.get('validation', {}).get('train', {}).get('validation_stats', {})
                            analysis_stats = result.get('analysis', {}).get('train', {}).get('analysis', {})
                            
                            if validation_stats or analysis_stats:
                                stats_html = f"""
                                <div style="background-color: #f8f9fa; padding: 10px; color: black; border-radius: 5px; margin-top: 10px;">
                                    <h4>📊 Preprocessing Stats</h4>
                                    <ul>
                                """
                                
                                if validation_stats:
                                    valid_percent = (validation_stats.get('valid_images', 0) / validation_stats.get('total_images', 1) * 100) if validation_stats.get('total_images', 0) > 0 else 0
                                    stats_html += f"""
                                        <li><b>Total images:</b> {validation_stats.get('total_images', 'N/A')}</li>
                                        <li><b>Valid images:</b> {validation_stats.get('valid_images', 'N/A')} ({valid_percent:.1f}%)</li>
                                    """
                                
                                if analysis_stats and 'class_balance' in analysis_stats:
                                    imbalance = analysis_stats['class_balance'].get('imbalance_score', 0)
                                    stats_html += f"""
                                        <li><b>Class imbalance score:</b> {imbalance:.2f}/10</li>
                                    """
                                
                                stats_html += f"""
                                        <li><b>Image size:</b> {img_size[0]}x{img_size[1]}</li>
                                    </ul>
                                </div>
                                """
                                display(HTML(stats_html))
                        else:
                            display(create_status_indicator(
                                "warning", 
                                f"⚠️ Preprocessing selesai dengan status: {result.get('status', 'unknown')}"
                            ))
                    
                    except Exception as e:
                        display(create_status_indicator("error", f"❌ Error dari PreprocessingManager: {str(e)}"))
                
                else:
                    # Pesan error jika PreprocessingManager tidak tersedia
                    display(create_status_indicator("error", "❌ PreprocessingManager tidak tersedia"))
            
            except Exception as e:
                display(create_status_indicator("error", f"❌ Error: {str(e)}"))
            
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