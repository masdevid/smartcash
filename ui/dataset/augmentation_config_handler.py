"""
File: smartcash/ui/dataset/augmentation_config_handler.py
Deskripsi: Handler konfigurasi augmentasi dengan dukungan balancing kelas dan sumber preprocessed
"""

from typing import Dict, Any, Optional
import os, yaml, copy
from pathlib import Path
from IPython.display import display

def update_config_from_ui(ui_components: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Update konfigurasi dari UI components."""
    logger = ui_components.get('logger')
    # PERBAIKAN: Deep copy untuk mencegah modifikasi tidak sengaja
    config = copy.deepcopy(config) if config else {}
    
    # Jika tidak ada 'augmentation' di config, buat baru
    if 'augmentation' not in config:
        config['augmentation'] = {}
    
    # Map UI types to config types
    type_map = {'Combined (Recommended)': 'combined', 'Position Variations': 'position', 
                'Lighting Variations': 'lighting', 'Extreme Rotation': 'extreme_rotation'}
    
    # Get augmentation types from UI
    aug_types = [type_map.get(t, 'combined') for t in ui_components['aug_options'].children[0].value]
    
    # Dapatkan jumlah workers dari UI jika tersedia
    num_workers = 4  # Default value
    if len(ui_components['aug_options'].children) > 5:
        num_workers = ui_components['aug_options'].children[5].value
    
    # Cek opsi balancing kelas (opsi baru)
    target_balance = False
    if len(ui_components['aug_options'].children) > 6:
        target_balance = ui_components['aug_options'].children[6].value
    
    # Update config dengan nilai baru dari UI
    config['augmentation'].update({
        'enabled': True,
        'types': aug_types,
        'num_variations': ui_components['aug_options'].children[1].value,
        'output_prefix': ui_components['aug_options'].children[2].value,
        'process_bboxes': ui_components['aug_options'].children[3].value if len(ui_components['aug_options'].children) > 3 else True,
        'validate_results': ui_components['aug_options'].children[4].value if len(ui_components['aug_options'].children) > 4 else True,
        'num_workers': num_workers,
        'target_balance': target_balance,
        'resume': False
    })
    
    # Update preprocessing default lokasi untuk source_dir
    if 'preprocessing' not in config:
        config['preprocessing'] = {}
    
    # Pastikan source_dir tersedia di config
    if 'preprocessed_dir' not in config['preprocessing']:
        config['preprocessing']['preprocessed_dir'] = 'data/preprocessed'
    
    # Pastikan file_prefix untuk penamaan file preprocessed tersedia
    if 'file_prefix' not in config['preprocessing']:
        config['preprocessing']['file_prefix'] = 'rp'
    
    # PERBAIKAN: Simpan ke ui_components untuk memastikan persistensi
    ui_components['config'] = config
    
    return config

def save_augmentation_config(config: Dict[str, Any], config_path: str = "configs/augmentation_config.yaml") -> bool:
    """Simpan konfigurasi augmentasi ke file."""
    logger = None
    if 'logger' in config and callable(getattr(config.get('logger', None), 'info', None)):
        logger = config.get('logger')
    
    try:
        # PERBAIKAN: Deep copy untuk mencegah modifikasi tidak sengaja
        save_config = copy.deepcopy(config)
        
        # Pastikan direktori config ada
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        
        # PERBAIKAN: Cek jika file sudah ada, merge dengan config yang ada
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                existing_config = yaml.safe_load(f) or {}
                
            # Merge existing config dengan config baru
            merged_config = copy.deepcopy(existing_config)
            
            # Update config dengan augmentation settings baru
            if 'augmentation' in save_config:
                if 'augmentation' not in merged_config:
                    merged_config['augmentation'] = {}
                merged_config['augmentation'].update(save_config['augmentation'])
            
            # Update preprocessing jika ada
            if 'preprocessing' in save_config:
                if 'preprocessing' not in merged_config:
                    merged_config['preprocessing'] = {}
                merged_config['preprocessing'].update(save_config['preprocessing'])
                
            # Gunakan config yang sudah dimerge
            save_config = merged_config
        
        # Coba gunakan ConfigManager untuk konsistensi
        try:
            from smartcash.common.config import get_config_manager
            config_manager = get_config_manager()
            
            # Update config di manager
            for section in ['augmentation', 'preprocessing']:
                if section in save_config:
                    for key, value in save_config[section].items():
                        config_manager.set(f"{section}.{key}", value)
            
            # Simpan ke file
            config_manager.save_config(config_path)
            
            if logger: logger.info(f"✅ Konfigurasi disimpan ke {config_path} via ConfigManager")
        except (ImportError, AttributeError):
            # Fallback: simpan langsung ke file
            with open(config_path, 'w') as f:
                yaml.dump(save_config, f, default_flow_style=False)
            
            if logger: logger.info(f"✅ Konfigurasi disimpan ke {config_path} (fallback)")
        
        # PERBAIKAN: Cegah duplikasi ke Google Drive jika path sama
        try:
            from smartcash.common.environment import get_environment_manager
            env_manager = get_environment_manager()
            
            if env_manager.is_drive_mounted:
                drive_config_path = str(env_manager.drive_path / 'configs' / Path(config_path).name)
                
                # Cek apakah path sama dengan realpath untuk mencegah error pada symlink
                if os.path.realpath(config_path) == os.path.realpath(drive_config_path):
                    if logger: logger.info(f"🔄 File lokal dan drive identik: {config_path}, melewati salinan")
                else:
                    # Buat direktori drive jika belum ada
                    os.makedirs(Path(drive_config_path).parent, exist_ok=True)
                    
                    # Simpan juga ke drive untuk backup
                    with open(drive_config_path, 'w') as f:
                        yaml.dump(save_config, f, default_flow_style=False)
                    
                    if logger: logger.info(f"📤 Konfigurasi disimpan ke drive: {drive_config_path}")
        except Exception as e:
            if logger: logger.debug(f"ℹ️ Tidak dapat menyimpan ke drive: {str(e)}")
        
        return True
    except Exception as e:
        if logger: logger.error(f"❌ Error saat menyimpan konfigurasi: {str(e)}")
        return False

def load_augmentation_config(config_path: str = "configs/augmentation_config.yaml", ui_components: Dict[str, Any] = None) -> Dict[str, Any]:
    """Load konfigurasi augmentasi dari file."""
    logger = None
    if ui_components and 'logger' in ui_components:
        logger = ui_components.get('logger')
    
    # PERBAIKAN: Cek apakah ada config tersimpan di ui_components
    if ui_components and 'config' in ui_components and ui_components['config']:
        if logger: logger.info("ℹ️ Menggunakan konfigurasi dari UI components")
        return ui_components['config']
    
    try:
        # PERBAIKAN: Cek konfigurasi dari Google Drive terlebih dahulu 
        try:
            from smartcash.common.environment import get_environment_manager
            env_manager = get_environment_manager()
            
            if env_manager.is_drive_mounted:
                drive_config_path = str(env_manager.drive_path / 'configs' / Path(config_path).name)
                
                # Cek apakah path sama dengan realpath untuk mencegah error pada symlink
                if os.path.realpath(config_path) == os.path.realpath(drive_config_path):
                    if logger: logger.info(f"🔄 File lokal dan drive identik: {config_path}, menggunakan lokal")
                elif os.path.exists(drive_config_path):
                    # Baca langsung dari file drive untuk mendapatkan versi terbaru
                    with open(drive_config_path, 'r') as f:
                        drive_config = yaml.safe_load(f)
                        
                    if drive_config:
                        # Salin juga ke lokal untuk digunakan sebagai cache
                        os.makedirs(Path(config_path).parent, exist_ok=True)
                        with open(config_path, 'w') as f:
                            yaml.dump(drive_config, f, default_flow_style=False)
                            
                        if logger: logger.info(f"📥 Konfigurasi dimuat dari drive: {drive_config_path}")
                        
                        # PERBAIKAN: Simpan ke ui_components jika tersedia
                        if ui_components: ui_components['config'] = drive_config
                        return drive_config
        except (ImportError, AttributeError) as e:
            if logger: logger.debug(f"ℹ️ Tidak dapat menyalin dari drive: {str(e)}")
                
        # PERBAIKAN: Coba load dari ConfigManager untuk konsistensi
        try:
            from smartcash.common.config import get_config_manager
            config_manager = get_config_manager()
            
            # Paksa reload config untuk mendapatkan data terbaru
            loaded_config = config_manager.load_config(config_path)
            
            # Log hasil untuk debugging
            has_aug = 'augmentation' in loaded_config
            has_preproc = 'preprocessing' in loaded_config
            
            if loaded_config and (has_aug or has_preproc):
                if logger: logger.info(f"ℹ️ Loaded config dari {config_path} via ConfigManager")
                
                # PERBAIKAN: Simpan ke ui_components jika tersedia
                if ui_components: ui_components['config'] = loaded_config
                return loaded_config
        except (ImportError, FileNotFoundError, AttributeError) as e:
            if logger: logger.warning(f"⚠️ Load config fallback: {str(e)}")
        
        # Fallback: load langsung dengan yaml
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    if config:
                        # Log hasil untuk debugging
                        has_aug = 'augmentation' in config
                        has_preproc = 'preprocessing' in config
                        if logger:
                            logger.info(f"📄 Loaded YAML dari {config_path}: augmentation={has_aug}, preprocessing={has_preproc}")
                        
                        # PERBAIKAN: Verifikasi struktur dasar config ada
                        if 'augmentation' not in config: config['augmentation'] = {}
                        if 'preprocessing' not in config: config['preprocessing'] = {}
                        
                        # PERBAIKAN: Simpan ke ui_components jika tersedia
                        if ui_components: ui_components['config'] = config
                        return config
            except Exception as e:
                if logger:
                    logger.warning(f"⚠️ Error saat load YAML: {str(e)}")
    except Exception as e:
        if logger: logger.warning(f"⚠️ Error saat memuat konfigurasi: {str(e)}")
    
    # Jika config tidak bisa dimuat, gunakan default config
    if logger:
        logger.info("📋 Menggunakan konfigurasi default")
    
    default_config = load_default_augmentation_config()
    
    # PERBAIKAN: Simpan default config ke ui_components jika tersedia
    if ui_components: ui_components['config'] = default_config
    
    return default_config

def load_default_augmentation_config() -> Dict[str, Any]:
    """
    Load konfigurasi default untuk augmentasi dataset dengan support balancing dan sumber preprocessed.
    
    Returns:
        Dictionary konfigurasi default
    """
    # Default config dengan nilai standar
    return {
        "augmentation": {
            "enabled": True,
            "types": ["combined", "position", "lighting"],
            "num_variations": 2,
            "output_prefix": "aug",
            "process_bboxes": True,
            "validate_results": True,
            "resume": False,
            "num_workers": 4,
            "target_balance": True,
            "position": {
                "fliplr": 0.5,
                "flipud": 0.0,
                "degrees": 15,
                "translate": 0.1,
                "scale": 0.05,  # Dikurangi dari 0.1 menjadi 0.05 untuk batasi scaling
                "shear": 0.0,
                "rotation_prob": 0.5,
                "max_angle": 15,
                "flip_prob": 0.5,
                "scale_ratio": 0.05  # Dikurangi dari 0.1 menjadi 0.05
            },
            "lighting": {
                "hsv_h": 0.015,
                "hsv_s": 0.7,
                "hsv_v": 0.4,
                "contrast": 0.3,
                "brightness": 0.3,
                "compress": 0.2,
                "brightness_prob": 0.5,
                "brightness_limit": 0.3,
                "contrast_prob": 0.5,
                "contrast_limit": 0.3
            },
            "extreme": {
                "rotation_min": 30,
                "rotation_max": 90,
                "probability": 0.3
            }
        },
        "preprocessing": {
            "file_prefix": "rp",
            "preprocessed_dir": "data/preprocessed"
        }
    }

def update_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Update UI components dari konfigurasi."""
    logger = ui_components.get('logger')
    
    # Gunakan konfigurasi default sebagai fallback
    if not config or 'augmentation' not in config:
        if logger:
            logger.info("ℹ️ Konfigurasi augmentasi tidak ditemukan, menggunakan default")
        config = load_default_augmentation_config()
    
    aug_config = config['augmentation']
    
    try:
        # Update augmentation types dengan conversion map untuk UI
        if 'types' in aug_config:
            type_map = {'combined': 'Combined (Recommended)', 'position': 'Position Variations', 
                       'lighting': 'Lighting Variations', 'extreme_rotation': 'Extreme Rotation'}
            ui_types = [type_map.get(t, 'Combined (Recommended)') 
                        for t in aug_config['types'] 
                        if t in type_map.keys()]
            
            # Pastikan minimal satu tipe augmentasi dipilih
            if not ui_types:
                ui_types = ['Combined (Recommended)']
                
            ui_components['aug_options'].children[0].value = ui_types
        
        # Update inputs dengan values dari config dengan mapping field-index
        options_map = {1: 'num_variations', 2: 'output_prefix', 3: 'process_bboxes', 
                      4: 'validate_results', 5: 'num_workers', 6: 'target_balance'}
        
        # Update semua fields berdasarkan mapping
        for idx, field in options_map.items():
            if idx < len(ui_components['aug_options'].children) and field in aug_config:
                ui_components['aug_options'].children[idx].value = aug_config[field]
                    
        if logger:
            logger.info(f"✅ UI berhasil diupdate dari konfigurasi")
    except Exception as e:
        # Log error jika tersedia
        if logger:
            logger.warning(f"⚠️ Error updating UI from config: {str(e)}")
    
    # PERBAIKAN: Simpan referensi config di ui_components untuk persistensi
    ui_components['config'] = config
    
    return ui_components