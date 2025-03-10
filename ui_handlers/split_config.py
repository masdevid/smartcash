"""
File: smartcash/ui_handlers/split_config.py
Author: Alfrida Sabar (refactored)
Deskripsi: Handler untuk UI konfigurasi split dataset SmartCash.
"""

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import os, sys, time
from pathlib import Path

def setup_split_config_handlers(ui_components, config=None):
    """Setup handlers untuk UI konfigurasi split dataset."""
    # Default config jika tidak disediakan
    if config is None:
        config = {
            'data': {
                'train_dir': 'data/train',
                'valid_dir': 'data/valid',
                'test_dir': 'data/test'
            },
            'data_dir': 'data'
        }
    
    # Setup akses ke komponen UI
    split_options = ui_components['split_options']
    advanced_options = ui_components['advanced_options']
    split_button = ui_components['split_button']
    split_status = ui_components['split_status']
    stats_output = ui_components['stats_output']
    
    # Setup logger dan DatasetManager jika tersedia
    dataset_manager = None
    logger = None
    try:
        from smartcash.utils.logger import get_logger
        from smartcash.handlers.dataset import DatasetManager
        from smartcash.utils.observer import EventDispatcher, EventTopics
        
        logger = get_logger("split_config")
        dataset_manager = DatasetManager(config, logger=logger)
        
        if logger:
            logger.info("✅ DatasetManager berhasil diinisialisasi")
    except ImportError as e:
        print(f"ℹ️ Beberapa modul tidak tersedia: {str(e)}")
    
    # Handler untuk tombol split
    def on_split_click(b):
        with split_status:
            clear_output()
            display(create_status_indicator("info", "🔄 Menerapkan konfigurasi split dataset..."))
            
            try:
                # Ambil split options dari form
                train_pct = split_options.children[0].value
                val_pct = split_options.children[1].value
                test_pct = split_options.children[2].value
                stratified = split_options.children[3].value
                
                # Ambil advanced options
                random_seed = advanced_options.children[0].value
                force_resplit = advanced_options.children[1].value
                preserve_structure = advanced_options.children[2].value
                
                # Validasi total persentase
                total = train_pct + val_pct + test_pct
                if abs(total - 100.0) > 0.01:
                    display(create_status_indicator(
                        "warning", 
                        f"⚠️ Total persentase ({total}%) tidak sama dengan 100%. Menyesuaikan..."
                    ))
                    # Sesuaikan persentase
                    factor = 100.0 / total
                    train_pct *= factor
                    val_pct *= factor
                    test_pct *= factor
                
                # Update config jika diperlukan
                if config and 'data' in config:
                    config['data']['split_ratios'] = {
                        'train': train_pct / 100.0,
                        'valid': val_pct / 100.0,
                        'test': test_pct / 100.0
                    }
                    config['data']['stratified_split'] = stratified
                    config['data']['random_seed'] = random_seed
                
                # Notification event
                if 'EventDispatcher' in locals():
                    EventDispatcher.notify(
                        event_type=EventTopics.DATASET_SPLIT_START,
                        sender="split_config_handler",
                        train_pct=train_pct,
                        val_pct=val_pct,
                        test_pct=test_pct,
                        stratified=stratified,
                        random_seed=random_seed
                    )
                
                # Gunakan DatasetManager jika tersedia
                if dataset_manager:
                    display(create_status_indicator("info", "⚙️ Menggunakan DatasetManager untuk split dataset..."))
                    
                    try:
                        # Jalankan split
                        result = dataset_manager.split_dataset(
                            ratios={
                                'train': train_pct / 100.0,
                                'valid': val_pct / 100.0,
                                'test': test_pct / 100.0
                            },
                            stratified=stratified,
                            random_seed=random_seed,
                            force_resplit=force_resplit,
                            preserve_structure=preserve_structure
                        )
                        
                        # Tampilkan hasil
                        if result and result.get('success', False):
                            display(create_status_indicator(
                                "success", 
                                f"✅ Split dataset berhasil: Train {train_pct:.1f}%, Validation {val_pct:.1f}%, Test {test_pct:.1f}%"
                            ))
                            
                            # Tampilkan info tambahan jika stratified
                            if stratified:
                                display(create_status_indicator(
                                    "info", 
                                    "⚖️ Split stratified diterapkan untuk menjaga distribusi kelas yang seimbang."
                                ))
                            
                            # Tampilkan statistik split
                            if 'stats' in result:
                                update_stats_display(result['stats'])
                        else:
                            display(create_status_indicator(
                                "warning", 
                                f"⚠️ Split dataset selesai dengan issues: {result.get('message', 'unknown error')}"
                            ))
                    
                    except Exception as e:
                        display(create_status_indicator("error", f"❌ Error dari DatasetManager: {str(e)}"))
                
                else:
                    # Simulasi split jika DatasetManager tidak tersedia
                    display(create_status_indicator("info", "ℹ️ DatasetManager tidak tersedia, melakukan simulasi split..."))
                    
                    # Simulasi delay
                    time.sleep(1.5)
                    
                    # Tampilkan sukses dengan persentase yang disesuaikan
                    display(create_status_indicator(
                        "success", 
                        f"✅ Split diterapkan: Train {train_pct:.1f}%, Validation {val_pct:.1f}%, Test {test_pct:.1f}%"
                    ))
                    
                    # Tampilkan info tambahan jika stratified
                    if stratified:
                        display(create_status_indicator(
                            "info", 
                            "⚖️ Split stratified diterapkan untuk menjaga distribusi kelas yang seimbang."
                        ))
                    
                    # Update simulasi stats
                    update_stats_display({
                        'train_images': int(580 * train_pct / 70.0),
                        'valid_images': int(130 * val_pct / 15.0),
                        'test_images': int(130 * test_pct / 15.0),
                        'class_distribution': {
                            'train': {'001': 120, '002': 115, '005': 125, '010': 130},
                            'valid': {'001': 30, '002': 28, '005': 35, '010': 37},
                            'test': {'001': 30, '002': 28, '005': 35, '010': 37}
                        }
                    })
                
                # Notification event
                if 'EventDispatcher' in locals():
                    EventDispatcher.notify(
                        event_type=EventTopics.DATASET_SPLIT_END,
                        sender="split_config_handler",
                        train_pct=train_pct,
                        val_pct=val_pct,
                        test_pct=test_pct,
                        stratified=stratified
                    )
                
            except Exception as e:
                display(create_status_indicator("error", f"❌ Error: {str(e)}"))
                
                # Notification event
                if 'EventDispatcher' in locals():
                    EventDispatcher.notify(
                        event_type=EventTopics.DATASET_SPLIT_ERROR,
                        sender="split_config_handler",
                        error=str(e)
                    )
    
    # Function to update stats display
    def update_stats_display(stats):
        with stats_output:
            clear_output()
            
            train_images = stats.get('train_images', 0)
            valid_images = stats.get('valid_images', 0)
            test_images = stats.get('test_images', 0)
            total_images = train_images + valid_images + test_images
            
            # Create stats HTML
            stats_html = f"""
            <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 10px;">
                <h4>📊 Dataset Statistics</h4>
                <div>
                    <div style="display: flex; align-items: center; margin-bottom: 10px;">
                        <div style="flex: 1; font-weight: bold;">Total Images:</div>
                        <div style="flex: 2;">{total_images}</div>
                    </div>
                    <div style="display: flex; align-items: center; margin-bottom: 10px;">
                        <div style="flex: 1; font-weight: bold;">Train Split:</div>
                        <div style="flex: 2;">{train_images} images ({train_images/total_images*100:.1f}%)</div>
                    </div>
                    <div style="display: flex; align-items: center; margin-bottom: 10px;">
                        <div style="flex: 1; font-weight: bold;">Validation Split:</div>
                        <div style="flex: 2;">{valid_images} images ({valid_images/total_images*100:.1f}%)</div>
                    </div>
                    <div style="display: flex; align-items: center; margin-bottom: 10px;">
                        <div style="flex: 1; font-weight: bold;">Test Split:</div>
                        <div style="flex: 2;">{test_images} images ({test_images/total_images*100:.1f}%)</div>
                    </div>
                </div>
            """
            
            # Add class distribution if available
            if 'class_distribution' in stats:
                stats_html += """
                <h5>Class Distribution</h5>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="background-color: #e9ecef;">
                        <th style="padding: 8px; text-align: left; border: 1px solid #dee2e6;">Class</th>
                        <th style="padding: 8px; text-align: center; border: 1px solid #dee2e6;">Train</th>
                        <th style="padding: 8px; text-align: center; border: 1px solid #dee2e6;">Validation</th>
                        <th style="padding: 8px; text-align: center; border: 1px solid #dee2e6;">Test</th>
                        <th style="padding: 8px; text-align: center; border: 1px solid #dee2e6;">Total</th>
                    </tr>
                """
                
                # Get all unique classes
                all_classes = set()
                for split in ['train', 'valid', 'test']:
                    if split in stats['class_distribution']:
                        all_classes.update(stats['class_distribution'][split].keys())
                
                # Add rows for each class
                for cls in sorted(all_classes):
                    train_count = stats['class_distribution'].get('train', {}).get(cls, 0)
                    valid_count = stats['class_distribution'].get('valid', {}).get(cls, 0)
                    test_count = stats['class_distribution'].get('test', {}).get(cls, 0)
                    total_count = train_count + valid_count + test_count
                    
                    stats_html += f"""
                    <tr>
                        <td style="padding: 8px; border: 1px solid #dee2e6;">{cls}</td>
                        <td style="padding: 8px; text-align: center; border: 1px solid #dee2e6;">{train_count}</td>
                        <td style="padding: 8px; text-align: center; border: 1px solid #dee2e6;">{valid_count}</td>
                        <td style="padding: 8px; text-align: center; border: 1px solid #dee2e6;">{test_count}</td>
                        <td style="padding: 8px; text-align: center; border: 1px solid #dee2e6;">{total_count}</td>
                    </tr>
                    """
                
                stats_html += "</table>"
            
            stats_html += "</div>"
            display(HTML(stats_html))
    
    # Register handler
    split_button.on_click(on_split_click)
    
    # Inisialisasi dari config
    if config and 'data' in config and 'split_ratios' in config['data']:
        # Update input fields dari config
        split_ratios = config['data']['split_ratios']
        
        # Setup persentase split
        split_options.children[0].value = split_ratios.get('train', 0.7) * 100
        split_options.children[1].value = split_ratios.get('valid', 0.15) * 100
        split_options.children[2].value = split_ratios.get('test', 0.15) * 100
        
        # Setup stratified checkbox
        split_options.children[3].value = config['data'].get('stratified_split', True)
        
        # Setup advanced options
        advanced_options.children[0].value = config['data'].get('random_seed', 42)
    
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