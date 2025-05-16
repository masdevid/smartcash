"""
File: smartcash/ui/dataset/augmentation/components/augmentation_options.py
Deskripsi: Komponen UI untuk opsi augmentasi dataset
"""

import ipywidgets as widgets
from typing import Dict, Any, List, Optional

def create_combined_options(config: Dict[str, Any] = None) -> widgets.VBox:
    """
    Buat komponen UI gabungan untuk opsi augmentasi dataset tanpa tab.
    
    Args:
        config: Konfigurasi aplikasi
        
    Returns:
        Widget VBox berisi opsi augmentasi yang digabungkan
    """
    from smartcash.ui.utils.constants import COLORS, ICONS
    from smartcash.common.config.manager import get_config_manager
    from smartcash.common.logger import get_logger
    
    logger = get_logger('augmentation')
    
    try:
        # Dapatkan konfigurasi augmentasi
        config_manager = get_config_manager()
        if config_manager is None:
            aug_config = {}
        else:
            aug_config = config_manager.get_module_config('augmentation')
        
        # Gunakan config yang diberikan jika ada
        if config is not None and isinstance(config, dict):
            if 'augmentation' in config:
                aug_config = config.get('augmentation', {})
        
        # Pastikan aug_config memiliki struktur yang benar
        if not aug_config or not isinstance(aug_config, dict):
            aug_config = {}
        if 'augmentation' not in aug_config:
            aug_config['augmentation'] = {}
    except Exception as e:
        aug_config = {}
        aug_config['augmentation'] = {}
    
    # Daftar jenis augmentasi yang tersedia
    available_types = [
        'combined',  # Kombinasi posisi dan pencahayaan (direkomendasikan)
        'position',  # Variasi posisi seperti rotasi, flipping, dan scaling
        'lighting'   # Variasi pencahayaan seperti brightness, contrast dan HSV
    ]
    
    # Daftar split yang tersedia
    available_splits = ['train', 'valid', 'test']
    
    # Opsi dasar
    aug_enabled = widgets.Checkbox(
        value=aug_config.get('augmentation', {}).get('enabled', True),
        description='Aktifkan Augmentasi',
        indent=False,
        layout=widgets.Layout(width='auto')
    )
    
    # Jumlah variasi per gambar
    num_variations = widgets.IntSlider(
        value=aug_config.get('augmentation', {}).get('num_variations', 2),
        min=1,
        max=10,
        step=1,
        description='Jumlah Variasi:',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        layout=widgets.Layout(width='95%')
    )
    
    # Jenis augmentasi (multi-select) dengan deskripsi
    aug_types_options = [
        ('Combined: Kombinasi posisi dan pencahayaan (direkomendasikan)', 'combined'),
        ('Position: Variasi posisi seperti rotasi, flipping, dan scaling', 'position'),
        ('Lighting: Variasi pencahayaan seperti brightness, contrast dan HSV', 'lighting')
    ]
    
    # Dapatkan nilai types dari konfigurasi atau gunakan default
    aug_types_value = aug_config.get('augmentation', {}).get('types', ['combined'])
    
    # Validasi nilai aug_types terhadap opsi yang tersedia
    available_values = [opt[1] for opt in aug_types_options]
    valid_aug_types = [t for t in aug_types_value if t in available_values]
    
    # Jika tidak ada nilai valid, gunakan default
    if not valid_aug_types:
        valid_aug_types = ['combined']
    
    # Pastikan nilai adalah tuple untuk SelectMultiple
    aug_types = widgets.SelectMultiple(
        options=aug_types_options,
        value=valid_aug_types,
        description='Jenis:',
        disabled=False,
        layout=widgets.Layout(width='95%', height='100px')
    )
    
    # Target split
    target_split = widgets.Dropdown(
        options=available_splits,
        value=aug_config.get('augmentation', {}).get('target_split', 'train'),
        description='Target Split:',
        disabled=False,
        layout=widgets.Layout(width='95%')
    )
    
    # Prefix output
    output_prefix = widgets.Text(
        value=aug_config.get('augmentation', {}).get('output_prefix', 'aug'),
        placeholder='Prefix untuk file hasil augmentasi',
        description='Output Prefix:',
        disabled=False,
        layout=widgets.Layout(width='95%')
    )
    
    # Target jumlah per kelas
    target_count = widgets.IntSlider(
        value=aug_config.get('augmentation', {}).get('target_count', 1000),
        min=100,
        max=5000,
        step=100,
        description='Target per Kelas:',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        layout=widgets.Layout(width='95%')
    )
    
    # Balancing kelas
    balance_classes = widgets.Checkbox(
        value=aug_config.get('augmentation', {}).get('balance_classes', True),
        description='Balancing Kelas',
        indent=False,
        layout=widgets.Layout(width='auto')
    )
    
    # Pindahkan ke preprocessed
    move_to_preprocessed = widgets.Checkbox(
        value=aug_config.get('augmentation', {}).get('move_to_preprocessed', True),
        description='Pindahkan ke Preprocessed',
        indent=False,
        layout=widgets.Layout(width='auto')
    )
    
    # Validasi hasil
    validate_results = widgets.Checkbox(
        value=aug_config.get('augmentation', {}).get('validate_results', True),
        description='Validasi Hasil',
        indent=False,
        layout=widgets.Layout(width='auto')
    )
    
    # Resume augmentasi
    resume = widgets.Checkbox(
        value=aug_config.get('augmentation', {}).get('resume', False),
        description='Resume Augmentasi',
        indent=False,
        layout=widgets.Layout(width='auto')
    )
    
    # Jumlah workers
    num_workers = widgets.IntSlider(
        value=aug_config.get('augmentation', {}).get('num_workers', 4),
        min=1,
        max=16,
        step=1,
        description='Jumlah Workers:',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        layout=widgets.Layout(width='95%')
    )
    
    # Informasi tentang split
    split_info = widgets.HTML(
        f"""
        <div style="padding: 5px; color: {COLORS['dark']};">
            <p><b>{ICONS['info']} Informasi Split:</b></p>
            <ul>
                <li><b>train</b>: Augmentasi pada data training (rekomendasi)</li>
                <li><b>valid</b>: Augmentasi pada data validasi (jarang diperlukan)</li>
                <li><b>test</b>: Augmentasi pada data testing (tidak direkomendasikan)</li>
            </ul>
        </div>
        """
    )
    
    # Layout gabungan dalam satu card full width
    combined_box = widgets.VBox([
        widgets.HTML(f"<h5 style='color: {COLORS['dark']}; margin: 5px 0;'>{ICONS['settings']} Opsi Dasar</h5>"),
        widgets.HBox([
            widgets.VBox([
                widgets.HBox([aug_enabled, balance_classes], layout=widgets.Layout(justify_content='space-between')),
                widgets.HBox([move_to_preprocessed, validate_results], layout=widgets.Layout(justify_content='space-between')),
                widgets.HBox([resume], layout=widgets.Layout(justify_content='flex-start')),
                num_variations,
                target_count,
                num_workers,
                output_prefix
            ], layout=widgets.Layout(width='50%')),
            widgets.VBox([
                widgets.HTML(f"<h5 style='color: {COLORS['dark']}; margin: 5px 0;'>{ICONS['augmentation']} Jenis Augmentasi & Split</h5>"),
                aug_types,
                widgets.HBox([
                    target_split,
                    split_info
                ], layout=widgets.Layout(justify_content='space-between', align_items='center'))
            ], layout=widgets.Layout(width='50%'))
        ], layout=widgets.Layout(width='100%'))
    ], layout=widgets.Layout(padding='10px', border='1px solid #ddd', width='100%'))
    
    return combined_box

def create_augmentation_options(config: Dict[str, Any] = None) -> widgets.VBox:
    """
    Buat komponen UI untuk opsi augmentasi dataset.
    
    Args:
        config: Konfigurasi aplikasi
        
    Returns:
        Widget VBox berisi opsi augmentasi
    """
    from smartcash.ui.utils.constants import COLORS, ICONS
    from smartcash.common.config.manager import get_config_manager
    from smartcash.common.logger import get_logger
    
    logger = get_logger('augmentation')
    logger.info(f"🔧 Membuat komponen opsi augmentasi")
    
    try:
        # Dapatkan konfigurasi augmentasi
        config_manager = get_config_manager()
        if config_manager is None:
            logger.warning(f"⚠️ ConfigManager tidak tersedia, menggunakan konfigurasi default")
            aug_config = {}
        else:
            aug_config = config_manager.get_module_config('augmentation')
            logger.info(f"📋 Konfigurasi augmentasi berhasil dimuat")
        
        # Gunakan config yang diberikan jika ada
        if config is not None and isinstance(config, dict):
            if 'augmentation' in config:
                aug_config = config.get('augmentation', {})
                logger.info(f"📋 Menggunakan konfigurasi yang diberikan")
        
        # Pastikan aug_config memiliki struktur yang benar
        if not aug_config or not isinstance(aug_config, dict):
            logger.warning(f"⚠️ Konfigurasi tidak valid, menggunakan konfigurasi default")
            aug_config = {}
        if 'augmentation' not in aug_config:
            aug_config['augmentation'] = {}
    except Exception as e:
        logger.error(f"❌ Error saat memuat konfigurasi: {str(e)}")
        import traceback
        logger.error(f"🔍 Traceback: {traceback.format_exc()}")
        aug_config = {}
        aug_config['augmentation'] = {}
    
    # Daftar jenis augmentasi yang tersedia
    available_types = [
        'combined',  # Kombinasi posisi dan pencahayaan (direkomendasikan)
        'position',  # Variasi posisi seperti rotasi, flipping, dan scaling
        'lighting'   # Variasi pencahayaan seperti brightness, contrast dan HSV
    ]
    
    # Daftar split yang tersedia
    available_splits = ['train', 'valid', 'test']
    
    # Opsi dasar
    aug_enabled = widgets.Checkbox(
        value=aug_config.get('augmentation', {}).get('enabled', True),
        description='Aktifkan Augmentasi',
        indent=False,
        layout=widgets.Layout(width='auto')
    )
    
    # Jumlah variasi per gambar
    num_variations = widgets.IntSlider(
        value=aug_config.get('augmentation', {}).get('num_variations', 2),
        min=1,
        max=10,
        step=1,
        description='Jumlah Variasi:',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        layout=widgets.Layout(width='95%')
    )
    
    # Jenis augmentasi (multi-select) dengan deskripsi
    aug_types_options = [
        ('Combined: Kombinasi posisi dan pencahayaan (direkomendasikan)', 'combined'),
        ('Position: Variasi posisi seperti rotasi, flipping, dan scaling', 'position'),
        ('Lighting: Variasi pencahayaan seperti brightness, contrast dan HSV', 'lighting')
    ]
    
    # Dapatkan nilai types dari konfigurasi atau gunakan default
    aug_types_value = aug_config.get('augmentation', {}).get('types', ['combined'])
    
    # Validasi nilai aug_types terhadap opsi yang tersedia
    available_values = [opt[1] for opt in aug_types_options]
    valid_aug_types = [t for t in aug_types_value if t in available_values]
    
    # Jika tidak ada nilai valid, gunakan default
    if not valid_aug_types:
        valid_aug_types = ['combined']
    
    # Pastikan nilai adalah tuple untuk SelectMultiple
    aug_types = widgets.SelectMultiple(
        options=aug_types_options,
        value=tuple(valid_aug_types),
        description='Jenis Augmentasi:',
        disabled=False,
        layout=widgets.Layout(width='95%', height='120px')
    )
    
    # Target split
    target_split = widgets.Dropdown(
        options=available_splits,
        value=aug_config.get('augmentation', {}).get('target_split', 'train'),
        description='Target Split:',
        disabled=False,
        layout=widgets.Layout(width='95%')
    )
    
    # Prefix output
    output_prefix = widgets.Text(
        value=aug_config.get('augmentation', {}).get('output_prefix', 'aug'),
        placeholder='Prefix untuk file hasil augmentasi',
        description='Output Prefix:',
        disabled=False,
        layout=widgets.Layout(width='95%')
    )
    
    # Target jumlah per kelas
    target_count = widgets.IntSlider(
        value=aug_config.get('augmentation', {}).get('target_count', 1000),
        min=100,
        max=5000,
        step=100,
        description='Target per Kelas:',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        layout=widgets.Layout(width='95%')
    )
    
    # Balancing kelas
    balance_classes = widgets.Checkbox(
        value=aug_config.get('augmentation', {}).get('balance_classes', True),
        description='Balancing Kelas',
        indent=False,
        layout=widgets.Layout(width='auto')
    )
    
    # Pindahkan ke preprocessed
    move_to_preprocessed = widgets.Checkbox(
        value=aug_config.get('augmentation', {}).get('move_to_preprocessed', True),
        description='Pindahkan ke Preprocessed',
        indent=False,
        layout=widgets.Layout(width='auto')
    )
    
    # Validasi hasil
    validate_results = widgets.Checkbox(
        value=aug_config.get('augmentation', {}).get('validate_results', True),
        description='Validasi Hasil',
        indent=False,
        layout=widgets.Layout(width='auto')
    )
    
    # Resume augmentasi
    resume = widgets.Checkbox(
        value=aug_config.get('augmentation', {}).get('resume', False),
        description='Resume Augmentasi',
        indent=False,
        layout=widgets.Layout(width='auto')
    )
    
    # Jumlah workers
    num_workers = widgets.IntSlider(
        value=aug_config.get('augmentation', {}).get('num_workers', 4),
        min=1,
        max=16,
        step=1,
        description='Jumlah Workers:',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        layout=widgets.Layout(width='95%')
    )
    
    # Layout opsi dasar
    basic_options = widgets.VBox([
        widgets.HBox([aug_enabled, balance_classes], layout=widgets.Layout(justify_content='space-between')),
        widgets.HBox([move_to_preprocessed, validate_results], layout=widgets.Layout(justify_content='space-between')),
        widgets.HBox([resume], layout=widgets.Layout(justify_content='flex-start')),
        num_variations,
        target_count,
        num_workers,
        output_prefix
    ], layout=widgets.Layout(padding='10px', border='1px solid #ddd', width='100%'))
    
    # Informasi tentang split
    split_info = widgets.HTML(
        f"""
        <div style="padding: 5px; color: {COLORS['dark']};">
            <p><b>{ICONS['info']} Informasi Split:</b></p>
            <ul>
                <li><b>train</b>: Augmentasi pada data training (rekomendasi)</li>
                <li><b>valid</b>: Augmentasi pada data validasi (jarang diperlukan)</li>
                <li><b>test</b>: Augmentasi pada data testing (tidak direkomendasikan)</li>
            </ul>
        </div>
        """
    )
    
    # Layout jenis augmentasi dengan split info yang terintegrasi
    augmentation_types_box = widgets.VBox([
        widgets.HTML(f"<h5 style='color: {COLORS['dark']}; margin: 5px 0;'>{ICONS['augmentation']} Jenis Augmentasi & Split</h5>"),
        aug_types,
        widgets.HBox([
            target_split,
            split_info
        ], layout=widgets.Layout(justify_content='space-between', align_items='center'))
    ], layout=widgets.Layout(padding='10px', border='1px solid #ddd', width='100%'))
    
    # Tab untuk opsi dasar dan jenis augmentasi (dipertahankan untuk kompatibilitas)
    tabs = widgets.Tab(children=[basic_options, augmentation_types_box])
    tabs.set_title(0, f"{ICONS['settings']} Opsi Dasar")
    tabs.set_title(1, f"{ICONS['augmentation']} Jenis Augmentasi")
    
    # Container utama
    try:
        container = widgets.VBox([
            tabs
        ], layout=widgets.Layout(margin='10px 0'))
        
        logger.info(f"✅ Komponen opsi augmentasi berhasil dibuat")
        return container
    except Exception as e:
        logger.error(f"❌ Error saat membuat container UI: {str(e)}")
        import traceback
        logger.error(f"🔍 Traceback: {traceback.format_exc()}")
        
        # Buat container fallback sederhana
        fallback_container = widgets.VBox([
            widgets.HTML(f"<h3 style='color: red;'>⚠️ Error saat membuat komponen opsi augmentasi</h3>"),
            widgets.HTML(f"<p>Detail error: {str(e)}</p>")
        ])
        return fallback_container
