"""
File: smartcash/ui_handlers/preprocessing.py
Author: Alfrida Sabar (refactored)
Deskripsi: Handler untuk UI preprocessing dataset SmartCash dengan implementasi ObserverManager
           dan perbaikan untuk fungsi cleanup.
"""

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import os, sys
from pathlib import Path

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

def setup_preprocessing_handlers(ui_components, config=None):
    """Setup handlers untuk UI preprocessing dataset."""
    # Default config jika tidak disediakan
    if config is None:
        config = {
            'data': {
                'preprocessing': {
                    'img_size': [640, 640],
                    'num_workers': 4,
                    'normalize_enabled': True,
                    'cache_enabled': True,
                    'output_dir': 'data/preprocessed'
                }
            },
            'data_dir': 'data'
        }
    
    # Setup akses ke komponen UI
    preprocess_options = ui_components['preprocess_options']
    preprocess_button = ui_components['preprocess_button']
    preprocess_progress = ui_components['preprocess_progress']
    preprocess_status = ui_components['preprocess_status']
    log_accordion = ui_components['log_accordion']
    
    # Tambahkan tombol cleanup
    cleanup_button = widgets.Button(
        description='Clean Preprocessed Data',
        button_style='danger',
        icon='trash',
        layout=widgets.Layout(display='none')
    )
    ui_components['cleanup_button'] = cleanup_button
    
    # Setup logger, PreprocessingManager, dan ObserverManager
    preprocessing_manager = None
    logger = None
    observer_manager = None
    
    try:
        from smartcash.utils.logger import get_logger
        from smartcash.handlers.preprocessing import PreprocessingManager
        from smartcash.utils.observer import EventDispatcher, EventTopics
        from smartcash.utils.observer.observer_manager import ObserverManager
        
        logger = get_logger("preprocessing")
        preprocessing_manager = PreprocessingManager(config=config, logger=logger)
        observer_manager = ObserverManager(auto_register=True)
        
    except ImportError as e:
        print(f"ℹ️ Beberapa modul tidak tersedia: {str(e)}")
    
    # Kelompok observer untuk preprocessing
    preprocessing_observers_group = "preprocessing_observers"
    
    # Fungsi untuk mendapatkan direktori output preprocessing
    def get_preprocessing_output_dir():
        """Mendapatkan direktori output preprocessing dari konfigurasi."""
        if config and 'data' in config and 'preprocessing' in config['data'] and 'output_dir' in config['data']['preprocessing']:
            return config['data']['preprocessing']['output_dir']
        return os.path.join(config.get('data_dir', 'data'), 'preprocessed')
    
    # Fungsi untuk cek apakah dataset sudah dipreprocess
    def check_preprocessed_dataset():
        """Cek apakah dataset sudah dipreprocess."""
        output_dir = get_preprocessing_output_dir()
        splits = ['train', 'valid', 'test']
        
        # Cek apakah direktori train/valid/test memiliki gambar yang sudah dipreprocess
        for split in splits:
            split_path = Path(output_dir) / split
            if not split_path.exists() or not any((split_path / 'images').glob('*')):
                return False
        
        return True
    
    # Fungsi untuk membersihkan data preprocessed
    def on_cleanup_click(b):
        with preprocess_status:
            clear_output()
            display(create_status_indicator("warning", "🗑️ Membersihkan data preprocessing..."))
            
            try:
                # Gunakan direktori output preprocessing
                output_dir = get_preprocessing_output_dir()
                splits = ['train', 'valid', 'test']
                
                # Hapus direktori gambar dan label untuk setiap split
                files_deleted = 0
                for split in splits:
                    split_path = Path(output_dir) / split
                    if split_path.exists():
                        for subdir in ['images', 'labels']:
                            full_subdir = split_path / subdir
                            if full_subdir.exists():
                                # Hapus file dalam direktori
                                for file_path in full_subdir.glob('*'):
                                    if file_path.is_file():
                                        file_path.unlink()
                                        files_deleted += 1
                
                display(create_status_indicator("success", 
                    f"✅ Data preprocessing berhasil dibersihkan ({files_deleted} file dihapus)"))
                
                # Sembunyikan tombol cleanup jika semua file sudah dihapus
                if not check_preprocessed_dataset():
                    cleanup_button.layout.display = 'none'
                
            except Exception as e:
                display(create_status_indicator("error", f"❌ Error: {str(e)}"))
    
    # Tambahkan handler ke tombol cleanup
    cleanup_button.on_click(on_cleanup_click)
    
    # Fungsi untuk update progress UI
    def update_progress_callback(event_type, sender, progress=0, total=100, message=None, **kwargs):
        # Update progress bar
        preprocess_progress.value = int(progress * 100 / total) if total > 0 else 0
        preprocess_progress.description = f"{int(progress * 100 / total)}%" if total > 0 else "0%"
        
        # Display message jika ada
        if message:
            with preprocess_status:
                display(create_status_indicator("info", message))
    
    # Setup observer untuk preprocessing progress jika observer_manager tersedia
    if observer_manager:
        try:
            # Buat progress observer
            progress_observer = observer_manager.create_simple_observer(
                event_type=EventTopics.PREPROCESSING_PROGRESS,
                callback=update_progress_callback,
                name="PreprocessingProgressObserver",
                group=preprocessing_observers_group
            )
            
            # Buat logger observer untuk event preprocessing
            logger_observer = observer_manager.create_logging_observer(
                event_types=[
                    EventTopics.PREPROCESSING_START,
                    EventTopics.PREPROCESSING_END,
                    EventTopics.PREPROCESSING_ERROR
                ],
                log_level="info",
                name="PreprocessingLoggerObserver",
                format_string="{event_type}: {message}",
                include_timestamp=True,
                logger_name="preprocessing",
                group=preprocessing_observers_group
            )
            
            if logger:
                logger.info("✅ Observer untuk preprocessing telah dikonfigurasi")
            
        except Exception as e:
            if logger:
                logger.error(f"❌ Error saat setup observer: {str(e)}")
    
    # Handler untuk tombol preprocessing
    def on_preprocess_click(b):
        # Disable tombol preprocessing saat sedang berjalan
        preprocess_button.disabled = True
        
        # Expand logs accordion untuk menampilkan progress
        log_accordion.selected_index = 0
        
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
                
                # Pastikan direktori output preprocessing ada dalam config
                if 'data' not in config:
                    config['data'] = {}
                if 'preprocessing' not in config['data']:
                    config['data']['preprocessing'] = {}
                if 'output_dir' not in config['data']['preprocessing']:
                    config['data']['preprocessing']['output_dir'] = os.path.join(config.get('data_dir', 'data'), 'preprocessed')
                
                # Tampilkan progress bar
                preprocess_progress.layout.visibility = 'visible'
                preprocess_progress.value = 0
                
                # Gunakan PreprocessingManager 
                if preprocessing_manager:
                    display(create_status_indicator("info", "⚙️ Menggunakan PreprocessingManager untuk preprocessing..."))
                    
                    try:
                        # Perbarui konfigurasi PreprocessingManager
                        preprocessing_manager.config = config
                        
                        # Jalankan preprocessing pipeline
                        result = preprocessing_manager.run_full_pipeline(
                            splits=['train', 'valid', 'test'],
                            validate_dataset=True,
                            fix_issues=False,
                            augment_data=False,
                            analyze_dataset=True
                        )
                        
                        # Tampilkan hasil
                        if result and result.get('status') == 'success':
                            display(create_status_indicator(
                                "success", 
                                f"✅ Preprocessing pipeline selesai dalam {result.get('elapsed', 0):.2f} detik"
                            ))
                            
                            # Tampilkan statistik preprocessing jika tersedia
                            validation_stats = result.get('validation', {}).get('train', {}).get('validation_stats', {})
                            analysis_stats = result.get('analysis', {}).get('train', {}).get('analysis', {})
                            
                            # Panel summary di bawah logs accordion
                            summary_html = f"""
                            <div style="background-color: #f8f9fa; padding: 10px; color: black; border-radius: 5px; margin-top: 10px;">
                                <h4>📊 Preprocessing Summary</h4>
                                <ul>
                            """
                            
                            if validation_stats:
                                valid_percent = (validation_stats.get('valid_images', 0) / validation_stats.get('total_images', 1) * 100) if validation_stats.get('total_images', 0) > 0 else 0
                                summary_html += f"""
                                    <li><b>Total images:</b> {validation_stats.get('total_images', 'N/A')}</li>
                                    <li><b>Valid images:</b> {validation_stats.get('valid_images', 'N/A')} ({valid_percent:.1f}%)</li>
                                """
                            
                            if analysis_stats and 'class_balance' in analysis_stats:
                                imbalance = analysis_stats['class_balance'].get('imbalance_score', 0)
                                summary_html += f"""
                                    <li><b>Class imbalance score:</b> {imbalance:.2f}/10</li>
                                """
                            
                            summary_html += f"""
                                    <li><b>Image size:</b> {img_size[0]}x{img_size[1]}</li>
                                    <li><b>Output directory:</b> {config['data']['preprocessing']['output_dir']}</li>
                                </ul>
                            </div>
                            """
                            
                            # Tambahkan summary di bawah log accordion
                            log_accordion.children[0].append_display(HTML(summary_html))
                            
                            # Tampilkan tombol cleanup
                            cleanup_button.layout.display = ''
                            
                            # Verifikasi hasil preprocessing
                            output_dir = config['data']['preprocessing']['output_dir']
                            if check_preprocessed_dataset():
                                display(create_status_indicator("success", 
                                    f"✅ Dataset berhasil dipreprocess dan tersedia di {output_dir}"))
                            else:
                                display(create_status_indicator("warning", 
                                    "⚠️ Preprocessing selesai tetapi beberapa file mungkin gagal dibuat"))
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
                # Enable kembali tombol preprocessing
                preprocess_button.disabled = False
                
                # Cek apakah ada data preprocessed untuk menampilkan tombol cleanup
                if check_preprocessed_dataset():
                    cleanup_button.layout.display = ''
    
    # Fungsi cleanup untuk unregister observer
    def cleanup():
        if observer_manager:
            try:
                observer_manager.unregister_group(preprocessing_observers_group)
                if logger:
                    logger.info("✅ Observer untuk preprocessing telah dibersihkan")
            except Exception as e:
                if logger:
                    logger.error(f"❌ Error saat membersihkan observer: {str(e)}")
    
    # Register handler
    preprocess_button.on_click(on_preprocess_click)
    
    # Tambahkan fungsi cleanup ke komponen UI
    ui_components['cleanup'] = cleanup
    
    # Tambahkan tombol cleanup ke UI
    main_container = ui_components['ui']
    main_container.children = list(main_container.children) + [cleanup_button]
    
    # Inisialisasi UI
    # Cek apakah dataset sudah dipreprocess
    if check_preprocessed_dataset():
        cleanup_button.layout.display = ''
    
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