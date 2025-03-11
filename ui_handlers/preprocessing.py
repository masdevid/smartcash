"""Preprocessing UI handler untuk SmartCash dataset."""

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

def setup_preprocessing_handlers(ui_components, config=None):
    """Setup handlers untuk preprocessing dataset."""
    config = config or {
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
    
    try:
        from smartcash.utils.logger import get_logger
        from smartcash.handlers.preprocessing import PreprocessingManager
        from smartcash.utils.observer.observer_manager import ObserverManager
        from smartcash.utils.observer import EventTopics
        
        logger = get_logger("preprocessing")
        preprocessing_manager = PreprocessingManager(config=config, logger=logger)
        observer_manager = ObserverManager(auto_register=True)
        preprocessors_group = "preprocessing_observers"
        
        # Progress tracking observer
        observer_manager.create_simple_observer(
            event_type=EventTopics.PREPROCESSING_PROGRESS,
            callback=lambda event_type, sender, progress=0, total=100, message=None, **kwargs: 
                _update_progress_ui(ui_components, progress, total, message),
            name="PreprocessingProgressObserver",
            group=preprocessors_group
        )
        
        # Pipeline end observer
        observer_manager.create_simple_observer(
            event_type=EventTopics.PREPROCESSING_END,
            callback=lambda event_type, sender, results=None, **kwargs: 
                logger.success(f"✅ Pipeline selesai dalam {results.get('elapsed', 0):.2f} detik") 
                if results else None,
            name="PipelineEndObserver",
            group=preprocessors_group
        )
    except ImportError as e:
        print(f"ℹ️ Modul tidak tersedia: {str(e)}")
        preprocessing_manager, observer_manager, logger = None, None, None
    
    def on_preprocess_click(b):
        with ui_components['preprocess_status']:
            clear_output()
            display(_create_status_indicator("info", "🔄 Memulai preprocessing dataset..."))
            
            try:
                img_size, normalize = _get_preprocessing_config(ui_components, config)
                
                if preprocessing_manager:
                    display(_create_status_indicator("info", "⚙️ Preprocessing dimulai..."))
                    
                    result = preprocessing_manager.run_full_pipeline(
                        splits=['train', 'valid', 'test'],
                        validate_dataset=True,
                        fix_issues=False,
                        augment_data=False,
                        analyze_dataset=True
                    )
                    
                    _display_preprocessing_results(result, img_size)
                else:
                    display(_create_status_indicator("error", "❌ PreprocessingManager tidak tersedia"))
            
            except Exception as e:
                display(_create_status_indicator("error", f"❌ Error: {str(e)}"))
    
    def _update_progress_ui(ui_components, progress, total, message=None):
        """Update UI progress."""
        ui_components['preprocess_progress'].value = int(progress * 100 / total) if total > 0 else 0
        ui_components['preprocess_progress'].description = f"{int(progress * 100 / total)}%" if total > 0 else "0%"
    
    def _get_preprocessing_config(ui_components, config):
        """Ekstrak konfigurasi preprocessing dari UI."""
        preprocess_options = ui_components['preprocess_options']
        img_size = preprocess_options.children[0].value
        normalize = preprocess_options.children[1].value
        
        # Update config
        if config and 'data' in config and 'preprocessing' in config['data']:
            config['data']['preprocessing']['img_size'] = list(img_size)
            config['data']['preprocessing']['normalize_enabled'] = normalize
        
        return img_size, normalize
    
    def _display_preprocessing_results(result, img_size):
        """Tampilkan hasil preprocessing."""
        if result and result.get('status') == 'success':
            display(_create_status_indicator(
                "success", 
                f"✅ Preprocessing selesai dalam {result.get('elapsed', 0):.2f} detik"
            ))
            
            # Tampilkan statistik
            validation_stats = result.get('validation', {}).get('train', {}).get('validation_stats', {})
            analysis_stats = result.get('analysis', {}).get('train', {}).get('analysis', {})
            
            if validation_stats or analysis_stats:
                stats_html = _generate_stats_html(validation_stats, analysis_stats, img_size)
                display(HTML(stats_html))
        else:
            display(_create_status_indicator(
                "warning", 
                f"⚠️ Preprocessing selesai dengan status: {result.get('status', 'unknown')}"
            ))
    
    def _generate_stats_html(validation_stats, analysis_stats, img_size):
        """Generate HTML untuk statistik preprocessing."""
        stats_html = """
        <div style="background-color: #f8f9fa; padding: 10px; color: black; border-radius: 5px; margin-top: 10px;">
            <h4>📊 Preprocessing Stats</h4>
            <ul>
        """
        
        # Tambahkan statistik validasi
        if validation_stats:
            valid_percent = (validation_stats.get('valid_images', 0) / validation_stats.get('total_images', 1) * 100) if validation_stats.get('total_images', 0) > 0 else 0
            stats_html += f"""
                <li><b>Total images:</b> {validation_stats.get('total_images', 'N/A')}</li>
                <li><b>Valid images:</b> {validation_stats.get('valid_images', 'N/A')} ({valid_percent:.1f}%)</li>
            """
        
        # Tambahkan statistik analisis
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
        
        return stats_html
    
    # Pasang handler
    ui_components['preprocess_button'].on_click(on_preprocess_click)
    
    return ui_components

def _create_status_indicator(status, message):
    """Buat indikator status."""
    styles = {
        'success': {'icon': '✅', 'color': 'green'},
        'warning': {'icon': '⚠️', 'color': 'orange'},
        'error': {'icon': '❌', 'color': 'red'},
        'info': {'icon': 'ℹ️', 'color': 'blue'}
    }
    
    style = styles.get(status, styles['info'])
    
    return HTML(f"""
    <div style="margin: 5px 0; padding: 8px 12px; 
                border-radius: 4px; background-color: #f8f9fa;">
        <span style="color: {style['color']}; font-weight: bold;"> 
            {style['icon']} {message}
        </span>
    </div>
    """)