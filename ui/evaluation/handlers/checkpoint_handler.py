"""
File: smartcash/ui/evaluation/handlers/checkpoint_handler.py
Deskripsi: Handler untuk checkpoint selection dan validasi dengan auto-detection checkpoint terbaik
"""

import os
import glob
import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from smartcash.ui.utils.logger_bridge import log_to_service

def setup_checkpoint_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None):
    """Setup handlers untuk checkpoint selection dengan one-liner pattern"""
    logger = ui_components.get('logger')
    
    # One-liner handler registration
    handlers = {
        'check_button': lambda b: check_available_checkpoints(ui_components, config, logger),
        'auto_select_checkbox': lambda change: toggle_checkpoint_selection(ui_components, change, logger),
        'checkpoint_path_text': lambda change: validate_custom_checkpoint(ui_components, change, logger),
        'scenario_dropdown': lambda change: scenario_changed(ui_components, change, config, logger)
    }
    
    # Register handlers dengan safe checking
    [getattr(ui_components.get(k), 'on_click' if 'button' in k else 'observe', lambda x: None)(v) 
     for k, v in handlers.items() if k in ui_components]
    
    # Auto-load checkpoint info saat initialization
    auto_load_checkpoint_info(ui_components, config, logger)
    
    return ui_components

def scenario_changed(ui_components: Dict[str, Any], change: Dict[str, Any], config: Dict[str, Any], logger) -> None:
    """Handler untuk perubahan skenario, otomatis update checkpoint terbaik"""
    if 'new' not in change or change['new'] == change.get('old'):
        return  # Skip jika tidak ada perubahan
    
    new_scenario = change['new']
    log_to_service(logger, f"🔄 Skenario berubah ke: {new_scenario}", "info")
    log_to_service(logger, "🔍 Mencari checkpoint terbaik untuk skenario ini...", "info")
    
    # Pastikan auto-select diaktifkan
    if 'auto_select_checkbox' in ui_components and not ui_components['auto_select_checkbox'].value:
        ui_components['auto_select_checkbox'].value = True
    
    # Cari checkpoint terbaik untuk skenario ini
    check_available_checkpoints(ui_components, config, logger)

def check_available_checkpoints(ui_components: Dict[str, Any], config: Dict[str, Any], logger) -> None:
    """Check dan display available checkpoints dengan metrics sesuai model dan backbone"""
    log_to_service(logger, "🔍 Mencari checkpoint yang tersedia...", "info")
    
    try:
        # Dapatkan model dan backbone dari scenario dropdown
        selected_scenario = ui_components.get('scenario_dropdown', {}).value if 'scenario_dropdown' in ui_components else None
        model_type, backbone = get_model_info_from_scenario(selected_scenario)
        
        log_to_service(logger, f"🧠 Mencari checkpoint untuk model: {model_type}, backbone: {backbone}", "info")
        
        # Checkpoint paths untuk search
        checkpoint_paths = [
            "runs/train/*/weights/best.pt",
            "runs/train/*/weights/last.pt", 
            "models/checkpoints/*.pt",
            "output/training/*/weights/*.pt"
        ]
        
        # Find all checkpoints dengan one-liner
        checkpoints = [cp for pattern in checkpoint_paths 
                      for cp in glob.glob(pattern) if os.path.exists(cp)]
        
        if not checkpoints:
            log_to_service(logger, "❌ Tidak ditemukan checkpoint. Pastikan training telah selesai.", "warning")
            return
        
        # Get checkpoint info dengan metrics
        checkpoint_info = [get_checkpoint_info(cp) for cp in checkpoints]
        
        # Filter checkpoint berdasarkan model dan backbone jika tersedia
        if model_type and backbone:
            filtered_checkpoints = filter_checkpoints_by_model(checkpoint_info, model_type, backbone)
            if filtered_checkpoints:
                log_to_service(logger, f"✅ Ditemukan {len(filtered_checkpoints)} checkpoint yang sesuai dengan {model_type} ({backbone})", "success")
                checkpoint_info = filtered_checkpoints
            else:
                log_to_service(logger, f"⚠️ Tidak ditemukan checkpoint untuk {model_type} ({backbone}). Menampilkan semua checkpoint.", "warning")
        
        # Pilih checkpoint terbaik
        best_checkpoint = get_best_checkpoint(checkpoint_info)
        
        # Display results dalam log
        display_checkpoint_results(ui_components, checkpoint_info, best_checkpoint, logger)
        
        # Update UI berdasarkan hasil
        update_checkpoint_ui(ui_components, best_checkpoint, logger)
        
    except Exception as e:
        log_to_service(logger, f"❌ Error saat mencari checkpoint: {str(e)}", "error")

def get_checkpoint_info(checkpoint_path: str) -> Dict[str, Any]:
    """Extract info dari checkpoint dengan one-liner pattern"""
    try:
        import torch
        
        # Load checkpoint dengan safe loading
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract info dengan one-liner dict comprehension
        info = {
            'path': checkpoint_path,
            'name': Path(checkpoint_path).stem,
            'folder': Path(checkpoint_path).parent.name,
            'size_mb': round(os.path.getsize(checkpoint_path) / (1024*1024), 2),
            'epoch': checkpoint.get('epoch', 0),
            'best_fitness': checkpoint.get('best_fitness', 0.0),
            'metrics': checkpoint.get('results', {}),
            'model_info': {
                'yaml': checkpoint.get('yaml', {}),
                'nc': checkpoint.get('nc', 10),  # Number of classes
                'names': checkpoint.get('names', [])
            }
        }
        
        # Parse additional metrics dari results
        if hasattr(checkpoint.get('results'), '__len__') and len(checkpoint.get('results', [])) > 0:
            results = checkpoint['results'][-1] if isinstance(checkpoint['results'], list) else checkpoint['results']
            info['metrics'].update({
                'precision': float(results[0]) if len(results) > 0 else 0.0,
                'recall': float(results[1]) if len(results) > 1 else 0.0,
                'mAP@0.5': float(results[2]) if len(results) > 2 else 0.0,
                'mAP@0.5:0.95': float(results[3]) if len(results) > 3 else 0.0,
                'loss': float(results[4]) if len(results) > 4 else 0.0
            })
        
        return info
        
    except Exception as e:
        return {
            'path': checkpoint_path, 'name': Path(checkpoint_path).stem,
            'error': str(e), 'valid': False, 'size_mb': 0, 'metrics': {}
        }

def get_best_checkpoint(checkpoint_infos: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Pilih checkpoint terbaik berdasarkan metrics dengan one-liner logic"""
    
    # Filter valid checkpoints
    valid_checkpoints = [cp for cp in checkpoint_infos if not cp.get('error')]
    
    if not valid_checkpoints:
        return None
    
    # Ranking berdasarkan priority metrics dengan one-liner
    def score_checkpoint(cp: Dict[str, Any]) -> float:
        metrics = cp.get('metrics', {})
        return (
            metrics.get('mAP@0.5:0.95', 0) * 0.4 +  # Primary metric
            metrics.get('mAP@0.5', 0) * 0.3 +       # Secondary metric
            cp.get('best_fitness', 0) * 0.2 +        # YOLOv5 fitness score
            (1 - metrics.get('loss', 1)) * 0.1       # Lower loss is better
        )
    
    # Return checkpoint dengan score tertinggi
    return max(valid_checkpoints, key=score_checkpoint)

def display_checkpoint_results(ui_components: Dict[str, Any], checkpoint_infos: List[Dict[str, Any]], 
                             best_checkpoint: Optional[Dict[str, Any]], logger) -> None:
    """Display hasil checkpoint search dalam log dengan formatted table"""
    
    log_to_service(logger, f"📊 Ditemukan {len(checkpoint_infos)} checkpoint:", "success")
    
    # Header table
    header = "| Checkpoint | Epoch | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Size (MB) |"
    separator = "|" + "-" * 78 + "|"
    
    log_to_service(logger, header, "info")
    log_to_service(logger, separator, "info")
    
    # One-liner untuk format setiap checkpoint
    [log_to_service(logger, format_checkpoint_row(cp, cp == best_checkpoint), "info") 
     for cp in checkpoint_infos if not cp.get('error')]
    
    if best_checkpoint:
        log_to_service(logger, f"🏆 Checkpoint terbaik: {best_checkpoint['name']} (Score: {calculate_score(best_checkpoint):.4f})", "success")

def format_checkpoint_row(cp: Dict[str, Any], is_best: bool = False) -> str:
    """Format checkpoint info sebagai table row dengan one-liner"""
    metrics = cp.get('metrics', {})
    marker = "🏆" if is_best else "  "
    
    return f"| {marker} {cp['name'][:12]:<12} | {cp.get('epoch', 0):>5} | {metrics.get('mAP@0.5', 0):>7.3f} | {metrics.get('mAP@0.5:0.95', 0):>12.3f} | {metrics.get('precision', 0):>9.3f} | {metrics.get('recall', 0):>6.3f} | {cp.get('size_mb', 0):>8.1f} |"

def calculate_score(cp: Dict[str, Any]) -> float:
    """Calculate score untuk checkpoint dengan same logic sebagai get_best_checkpoint"""
    metrics = cp.get('metrics', {})
    return (
        metrics.get('mAP@0.5:0.95', 0) * 0.4 +
        metrics.get('mAP@0.5', 0) * 0.3 +
        cp.get('best_fitness', 0) * 0.2 +
        (1 - metrics.get('loss', 1)) * 0.1
    )

def update_checkpoint_ui(ui_components: Dict[str, Any], best_checkpoint: Optional[Dict[str, Any]], logger) -> None:
    """Update UI dengan info checkpoint terbaik"""
    if not best_checkpoint:
        return
    
    # Update checkbox untuk auto-select
    if 'auto_select_checkbox' in ui_components:
        ui_components['auto_select_checkbox'].value = True
    
    # Update path text dengan checkpoint terbaik
    if 'checkpoint_path_text' in ui_components:
        ui_components['checkpoint_path_text'].value = best_checkpoint['path']
        ui_components['checkpoint_path_text'].disabled = True  # Disable saat auto-select
    
    log_to_service(logger, f"✅ UI diupdate dengan checkpoint terbaik: {best_checkpoint['name']}", "success")

def toggle_checkpoint_selection(ui_components: Dict[str, Any], change: Dict[str, Any], logger) -> None:
    """Toggle antara auto-select dan manual selection"""
    is_auto = change['new']
    
    if 'checkpoint_path_text' in ui_components:
        ui_components['checkpoint_path_text'].disabled = is_auto
        
        if is_auto:
            # Auto-load best checkpoint
            log_to_service(logger, "🔄 Switching ke auto-select checkpoint terbaik...", "info")
            check_available_checkpoints(ui_components, {}, logger)
        else:
            # Enable manual input
            ui_components['checkpoint_path_text'].value = ""
            log_to_service(logger, "✏️ Manual checkpoint selection enabled", "info")

def validate_custom_checkpoint(ui_components: Dict[str, Any], change: Dict[str, Any], logger) -> None:
    """Validasi custom checkpoint path"""
    checkpoint_path = change['new']
    
    if not checkpoint_path:
        return
    
    try:
        if not os.path.exists(checkpoint_path):
            log_to_service(logger, f"⚠️ Checkpoint tidak ditemukan: {checkpoint_path}", "warning")
            return
        
        # Validate checkpoint dengan get_checkpoint_info
        info = get_checkpoint_info(checkpoint_path)
        
        if info.get('error'):
            log_to_service(logger, f"❌ Checkpoint tidak valid: {info['error']}", "error")
        else:
            log_to_service(logger, f"✅ Checkpoint valid: {info['name']} (Epoch: {info.get('epoch', 0)})", "success")
            
    except Exception as e:
        log_to_service(logger, f"❌ Error validasi checkpoint: {str(e)}", "error")

def get_model_info_from_scenario(scenario_name: str) -> Tuple[Optional[str], Optional[str]]:
    """Ekstrak informasi model dan backbone dari nama skenario"""
    if not scenario_name:
        return None, None
    
    # Mapping skenario ke model dan backbone
    scenario_mapping = {
        'efficientnet_b4': ('yolov5', 'efficientnet_b4'),
        'efficientnet_optimized': ('yolov5', 'efficientnet_b4'),
        'csp_darknet': ('yolov5', 'csp_darknet'),
        'yolov5s': ('yolov5', 'csp_darknet'),
        'yolov5m': ('yolov5', 'csp_darknet'),
        'yolov5l': ('yolov5', 'csp_darknet'),
        'yolov5x': ('yolov5', 'csp_darknet')
    }
    
    # Cari match berdasarkan substring dalam nama skenario
    for key, (model, backbone) in scenario_mapping.items():
        if key.lower() in scenario_name.lower():
            return model, backbone
    
    # Default jika tidak ada yang cocok
    return 'yolov5', 'efficientnet_b4'

def filter_checkpoints_by_model(checkpoints: List[Dict[str, Any]], model_type: str, backbone: str) -> List[Dict[str, Any]]:
    """Filter checkpoint berdasarkan model dan backbone"""
    filtered = []
    
    for cp in checkpoints:
        # Cek dari nama checkpoint atau folder
        checkpoint_name = cp.get('name', '').lower()
        folder_name = cp.get('folder', '').lower()
        path = cp.get('path', '').lower()
        
        # Cek dari model_info jika tersedia
        model_info = cp.get('model_info', {})
        yaml_config = model_info.get('yaml', {})
        
        # Cek backbone dari yaml config atau nama
        has_backbone = False
        if isinstance(yaml_config, dict) and 'backbone' in yaml_config:
            has_backbone = backbone.lower() in str(yaml_config['backbone']).lower()
        else:
            has_backbone = backbone.lower() in checkpoint_name or backbone.lower() in folder_name or backbone.lower() in path
        
        # Cek model type dari yaml config atau nama
        has_model = False
        if isinstance(yaml_config, dict) and 'model_type' in yaml_config:
            has_model = model_type.lower() in str(yaml_config['model_type']).lower()
        else:
            has_model = model_type.lower() in checkpoint_name or model_type.lower() in folder_name or model_type.lower() in path
        
        # Tambahkan ke hasil jika cocok
        if has_model or has_backbone:  # Cukup salah satu yang cocok
            filtered.append(cp)
    
    return filtered

def auto_load_checkpoint_info(ui_components: Dict[str, Any], config: Dict[str, Any], logger) -> None:
    """Auto-load checkpoint info saat initialization"""
    if config.get('checkpoint', {}).get('auto_select_best', True):
        # Delayed loading untuk prevent blocking UI initialization
        import threading
        threading.Thread(
            target=lambda: check_available_checkpoints(ui_components, config, logger),
            daemon=True
        ).start()