"""
File: smartcash/ui/evaluation/handlers/scenario_handler.py
Deskripsi: Handler untuk pengelolaan skenario pengujian dan penyimpanan hasil evaluasi ke drive
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import ipywidgets as widgets

from smartcash.ui.utils.logger_bridge import log_to_service
from smartcash.common.environment import get_environment_manager

def setup_scenario_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup handler untuk skenario pengujian dan penyimpanan hasil ke drive"""
    logger = ui_components.get('logger')
    
    # Inisialisasi konfigurasi skenario jika belum ada
    if 'scenario' not in config:
        config['scenario'] = {
            'selected_scenario': 'scenario_1',
            'save_to_drive': True,
            'drive_path': '/content/drive/MyDrive/SmartCash/evaluation_results',
            'scenarios': {
                'scenario_1': {
                    'name': 'Skenario 1: YOLOv5 Default (CSPDarknet) backbone dengan positional variation',
                    'description': 'Skenario ini mengevaluasi model YOLOv5 dengan backbone default (CSPDarknet) pada variasi posisi mata uang.',
                    'folder_name': 'scenario_1_cspdarknet_position',
                    'backbone': 'cspdarknet_s',
                    'augmentation_type': 'position'
                },
                'scenario_2': {
                    'name': 'Skenario 2: YOLOv5 Default (CSPDarknet) backbone dengan lighting variation',
                    'description': 'Skenario ini mengevaluasi model YOLOv5 dengan backbone default (CSPDarknet) pada variasi pencahayaan mata uang.',
                    'folder_name': 'scenario_2_cspdarknet_lighting',
                    'backbone': 'cspdarknet_s',
                    'augmentation_type': 'lighting'
                },
                'scenario_3': {
                    'name': 'Skenario 3: YOLOv5 dengan EfficientNet-B4 backbone dengan positional variation',
                    'description': 'Skenario ini mengevaluasi model YOLOv5 dengan backbone EfficientNet-B4 pada variasi posisi mata uang.',
                    'folder_name': 'scenario_3_efficientnet_position',
                    'backbone': 'efficientnet_b4',
                    'augmentation_type': 'position'
                },
                'scenario_4': {
                    'name': 'Skenario 4: YOLOv5 dengan EfficientNet-B4 backbone dengan lighting variation',
                    'description': 'Skenario ini mengevaluasi model YOLOv5 dengan backbone EfficientNet-B4 pada variasi pencahayaan mata uang.',
                    'folder_name': 'scenario_4_efficientnet_lighting',
                    'backbone': 'efficientnet_b4',
                    'augmentation_type': 'lighting'
                }
            }
        }
    
    # Handler untuk perubahan skenario
    ui_components['scenario_dropdown'].observe(
        lambda change: on_scenario_change(change, ui_components, config, logger),
        names='value'
    )
    
    # Handler untuk checkbox save to drive
    ui_components['save_to_drive_checkbox'].observe(
        lambda change: on_save_to_drive_change(change, ui_components, config, logger),
        names='value'
    )
    
    # Handler untuk drive path text
    ui_components['drive_path_text'].observe(
        lambda change: on_drive_path_change(change, ui_components, config, logger),
        names='value'
    )
    
    # Inisialisasi deskripsi skenario
    update_scenario_description(ui_components, config, logger)
    
    return ui_components

def on_scenario_change(change, ui_components: Dict[str, Any], config: Dict[str, Any], logger) -> None:
    """Handler untuk perubahan skenario pengujian"""
    selected_scenario = change.new
    
    # Update config
    if 'scenario' in config:
        config['scenario']['selected_scenario'] = selected_scenario
        
        # Dapatkan informasi skenario
        scenario_info = get_scenario_info(selected_scenario, config)
        if scenario_info:
            # Update config dengan informasi skenario
            config['scenario']['backbone'] = scenario_info.get('backbone', 'cspdarknet_s')
            config['scenario']['augmentation_type'] = scenario_info.get('augmentation_type', 'position')
            config['scenario']['name'] = scenario_info.get('name', 'Tidak diketahui')
    
    # Update deskripsi skenario
    update_scenario_description(ui_components, config, logger)
    
    # Log perubahan
    scenario_name = get_scenario_name(selected_scenario, config)
    backbone = config['scenario'].get('backbone', 'cspdarknet_s')
    augmentation = config['scenario'].get('augmentation_type', 'position')
    log_to_service(logger, f"🧪 Skenario pengujian diubah ke: {scenario_name} (backbone: {backbone}, augmentasi: {augmentation})", "info")

def on_save_to_drive_change(change, ui_components: Dict[str, Any], config: Dict[str, Any], logger) -> None:
    """Handler untuk perubahan status save to drive"""
    save_to_drive = change.new
    
    # Update config
    if 'scenario' in config:
        config['scenario']['save_to_drive'] = save_to_drive
    
    # Enable/disable drive path text
    ui_components['drive_path_text'].disabled = not save_to_drive
    
    # Log perubahan
    status = "diaktifkan" if save_to_drive else "dinonaktifkan"
    log_to_service(logger, f"💾 Penyimpanan hasil ke drive {status}", "info")

def on_drive_path_change(change, ui_components: Dict[str, Any], config: Dict[str, Any], logger) -> None:
    """Handler untuk perubahan path drive"""
    drive_path = change.new
    
    # Update config
    if 'scenario' in config:
        config['scenario']['drive_path'] = drive_path
    
    # Log perubahan
    log_to_service(logger, f"💾 Path drive diubah ke: {drive_path}", "info")

def update_scenario_description(ui_components: Dict[str, Any], config: Dict[str, Any], logger) -> None:
    """Update deskripsi skenario berdasarkan skenario yang dipilih"""
    if 'scenario' not in config or 'scenario_description' not in ui_components:
        return
    
    selected_scenario = config['scenario'].get('selected_scenario', 'scenario_1')
    scenario_info = get_scenario_info(selected_scenario, config)
    
    if not scenario_info:
        return
    
    # Dapatkan informasi backbone dan augmentation
    backbone = scenario_info.get('backbone', 'cspdarknet_s')
    augmentation = scenario_info.get('augmentation_type', 'position')
    backbone_name = "EfficientNet-B4" if backbone == "efficientnet_b4" else "CSPDarknet (Default)"
    augmentation_name = "Posisi" if augmentation == "position" else "Pencahayaan"
    
    description_html = f"""
    <div style='padding: 10px; background-color: #f8f9fa; border-left: 3px solid #4CAF50; margin: 10px 0;'>
        <p><b>Skenario Pengujian:</b> {scenario_info.get('name', 'Tidak diketahui')}</p>
        <p>{scenario_info.get('description', '')}</p>
        <p><b>Backbone:</b> {backbone_name}</p>
        <p><b>Tipe Augmentasi:</b> {augmentation_name}</p>
        <p><small>Hasil prediksi akan disimpan ke drive dengan nama folder: <code>{scenario_info.get('folder_name', selected_scenario)}</code></small></p>
    </div>
    """
    
    ui_components['scenario_description'].value = description_html
    
    # Update config dengan informasi skenario
    config['scenario']['backbone'] = backbone
    config['scenario']['augmentation_type'] = augmentation
    config['scenario']['name'] = scenario_info.get('name', 'Tidak diketahui')

def get_scenario_info(scenario_id: str, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Dapatkan informasi skenario berdasarkan ID (UI wrapper)"""
    from smartcash.model.utils.scenario_utils import get_scenario_info as get_scenario_info_core
    return get_scenario_info_core(scenario_id, config)

def get_scenario_name(scenario_id: str, config: Dict[str, Any]) -> str:
    """Dapatkan nama skenario berdasarkan ID (UI wrapper)"""
    scenario_info = get_scenario_info(scenario_id, config)
    if scenario_info:
        return scenario_info.get('name', scenario_id)
    return scenario_id

def get_drive_path_for_scenario(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Optional[str]:
    """Dapatkan path drive untuk menyimpan hasil skenario (UI wrapper)"""
    from smartcash.model.utils.scenario_utils import get_drive_path_for_scenario as get_drive_path_core
    
    # Pastikan konfigurasi UI tercermin dalam config yang dikirim ke fungsi inti
    if 'scenario' in config and ui_components:
        # Update konfigurasi dari UI components jika tersedia
        if 'save_to_drive_checkbox' in ui_components and hasattr(ui_components['save_to_drive_checkbox'], 'value'):
            config['scenario']['save_to_drive'] = ui_components['save_to_drive_checkbox'].value
        
        if 'drive_path_text' in ui_components and hasattr(ui_components['drive_path_text'], 'value'):
            config['scenario']['drive_path'] = ui_components['drive_path_text'].value
    
    return get_drive_path_core(config)

def ensure_drive_directory(drive_path: str, logger=None) -> bool:
    """Pastikan direktori drive untuk menyimpan hasil ada"""
    try:
        # Cek apakah drive sudah di-mount
        if not os.path.exists('/content/drive'):
            if logger:
                log_to_service(logger, "❌ Google Drive tidak di-mount. Gunakan drive.mount() terlebih dahulu.", "error")
            return False
        
        # Buat direktori jika belum ada
        path = Path(drive_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            if logger:
                log_to_service(logger, f"📁 Membuat direktori baru: {drive_path}", "info")
        
        return True
    except Exception as e:
        if logger:
            log_to_service(logger, f"❌ Error saat membuat direktori drive: {str(e)}", "error")
        return False

def save_results_to_drive(results: Dict[str, Any], ui_components: Dict[str, Any], config: Dict[str, Any], logger=None) -> Dict[str, Any]:
    """Simpan hasil evaluasi ke drive sesuai dengan skenario yang dipilih (UI wrapper)"""
    from smartcash.model.utils.scenario_utils import save_results_to_drive as save_results_core
    
    # Wrapper untuk log_to_service jika logger diberikan
    def log_wrapper(message, level="info"):
        if logger:
            log_to_service(logger, message, level)
    
    # Pastikan konfigurasi UI tercermin dalam config yang dikirim ke fungsi inti
    if 'scenario' in config and ui_components:
        # Update konfigurasi dari UI components jika tersedia
        if 'save_to_drive_checkbox' in ui_components and hasattr(ui_components['save_to_drive_checkbox'], 'value'):
            config['scenario']['save_to_drive'] = ui_components['save_to_drive_checkbox'].value
        
        if 'drive_path_text' in ui_components and hasattr(ui_components['drive_path_text'], 'value'):
            config['scenario']['drive_path'] = ui_components['drive_path_text'].value
    
    try:
        # Pastikan direktori ada
        drive_path = get_drive_path_for_scenario(ui_components, config)
        if not drive_path:
            log_wrapper("⚠️ Penyimpanan ke drive tidak diaktifkan atau path tidak valid", "warning")
            return {
                'success': False,
                'message': "Penyimpanan ke drive tidak diaktifkan atau path tidak valid"
            }
            
        if not ensure_drive_directory(drive_path, logger):
            return {
                'success': False,
                'message': f"Gagal membuat direktori di drive: {drive_path}"
            }
        
        # Delegasikan ke implementasi di model module
        result = save_results_core(results, config, logger)
        
        if result['success']:
            log_wrapper(f"✅ Hasil evaluasi berhasil disimpan ke: {result.get('drive_path', drive_path)}", "success")
        else:
            log_wrapper(f"❌ Gagal menyimpan hasil ke drive: {result.get('error', 'Unknown error')}", "error")
        
        # Tambahkan drive_path ke hasil jika tidak ada
        if 'drive_path' not in result:
            result['drive_path'] = drive_path
            
        # Tambahkan message jika tidak ada
        if 'message' not in result:
            if result['success']:
                result['message'] = f"Hasil evaluasi berhasil disimpan ke: {result.get('drive_path', drive_path)}"
            else:
                result['message'] = f"Gagal menyimpan hasil ke drive: {result.get('error', 'Unknown error')}"
        
        return result
        
    except Exception as e:
        log_wrapper(f"❌ Error saat menyimpan hasil ke drive: {str(e)}", "error")
        
        return {
            'success': False,
            'message': f"Gagal menyimpan hasil ke drive: {str(e)}"
        }
