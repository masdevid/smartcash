"""
File: smartcash/model/utils/scenario_utils.py
Deskripsi: Utilitas untuk pengelolaan skenario evaluasi model dengan berbagai backbone dan augmentasi
"""

import os
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union

def get_scenario_info(scenario_id: str, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Dapatkan informasi skenario berdasarkan ID
    
    Args:
        scenario_id: ID skenario yang dipilih
        config: Konfigurasi yang berisi definisi skenario
        
    Returns:
        Dict berisi informasi skenario atau None jika tidak ditemukan
    """
    if 'scenario' not in config or 'scenarios' not in config['scenario']:
        return None
    
    scenarios = config['scenario']['scenarios']
    return scenarios.get(scenario_id, None)

def get_drive_path_for_scenario(scenario_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dapatkan path Google Drive dan informasi skenario untuk menyimpan hasil evaluasi
    
    Args:
        scenario_id: ID skenario evaluasi
        config: Konfigurasi yang berisi definisi skenario
        
    Returns:
        Dict berisi informasi skenario dan path Google Drive
    """
    if 'scenario' not in config:
        return {'success': False, 'error': 'Konfigurasi skenario tidak ditemukan'}
    
    # Dapatkan info skenario
    scenario_info = get_scenario_info(scenario_id, config)
    
    if not scenario_info:
        scenario_info = {'id': scenario_id, 'name': scenario_id}
    
    # Tambahkan ID ke info skenario
    scenario_info['id'] = scenario_id
    
    # Cek apakah save to drive diaktifkan
    save_to_drive = config['scenario'].get('save_to_drive', False)
    if not save_to_drive:
        return {
            'success': True, 
            'drive_enabled': False, 
            'name': scenario_info.get('name', scenario_id),
            'id': scenario_id,
            'info': scenario_info
        }
    
    # Dapatkan base path
    base_path = config['scenario'].get('drive_path', '/content/drive/MyDrive/SmartCash/evaluation_results')
    
    # Dapatkan folder name untuk skenario
    folder_name = scenario_info.get('folder_name', scenario_id)
    
    # Gabungkan path
    drive_path = os.path.join(base_path, folder_name)
    
    return {
        'success': True,
        'drive_enabled': True,
        'drive_path': drive_path,
        'name': scenario_info.get('name', scenario_id),
        'id': scenario_id,
        'info': scenario_info
    }

def save_evaluation_results(results: Dict[str, Any], save_path: str, 
                           scenario_info: Dict[str, Any], 
                           create_dirs: bool = True) -> Dict[str, Any]:
    """
    Simpan hasil evaluasi ke disk
    
    Args:
        results: Hasil evaluasi yang akan disimpan
        save_path: Path untuk menyimpan hasil
        scenario_info: Informasi skenario yang digunakan
        create_dirs: Buat direktori jika belum ada
        
    Returns:
        Dict berisi informasi file yang disimpan
    """
    import json
    import os
    import numpy as np
    from datetime import datetime
    
    # Buat direktori jika belum ada
    if create_dirs and not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    # Timestamp untuk nama file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Simpan metrics
    metrics_path = os.path.join(save_path, f"metrics_{timestamp}.json")
    
    # Tambahkan metadata skenario dan timestamp ke metrics
    metrics_data = results.get('metrics', {}).copy()
    metrics_data['metadata'] = metrics_data.get('metadata', {})
    metrics_data['metadata']['scenario'] = scenario_info
    metrics_data['metadata']['timestamp'] = datetime.now().isoformat()
    
    # Tambahkan inference metrics
    if 'inference_metrics' in results:
        metrics_data['inference_time'] = results['inference_metrics']
    
    # Konversi numpy arrays ke list untuk JSON serialization
    metrics_data = _convert_numpy_to_python(metrics_data)
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    # Simpan confusion matrix jika ada
    cm_path = None
    if 'metrics' in results and 'confusion_matrix' in results['metrics'] and results['metrics']['confusion_matrix']:
        cm_data = results['metrics']['confusion_matrix']
        cm_path = os.path.join(save_path, f"confusion_matrix_{timestamp}.npy")
        
        # Simpan matrix dan labels
        np.save(cm_path, {
            'matrix': cm_data.get('matrix', np.zeros((1, 1))),
            'labels': cm_data.get('labels', []),
            'normalized': cm_data.get('normalized', np.zeros((1, 1)))
        })
    
    # Simpan prediksi
    predictions_path = None
    if 'predictions' in results:
        predictions = results['predictions']
        predictions_path = os.path.join(save_path, f"predictions_{timestamp}.json")
        
        # Konversi numpy arrays ke list
        predictions_data = _convert_numpy_to_python(predictions)
        
        with open(predictions_path, 'w') as f:
            json.dump(predictions_data, f, indent=2)
    
    # Simpan visualisasi jika ada
    visualization_paths = []
    if 'visualizations' in results:
        visualizations = results['visualizations']
        for i, vis in enumerate(visualizations):
            if 'image' in vis and vis['image'] is not None:
                vis_path = os.path.join(save_path, f"visualization_{i}_{timestamp}.png")
                
                # Simpan gambar
                try:
                    import cv2
                    cv2.imwrite(vis_path, vis['image'])
                    visualization_paths.append(vis_path)
                except Exception:
                    pass
    
    # Kumpulkan semua path file yang disimpan
    saved_files = {
        'metrics': metrics_path,
        'confusion_matrix': cm_path,
        'predictions': predictions_path,
        'visualizations': visualization_paths,
        'base_path': save_path
    }
    
    return saved_files

def save_results_to_drive(results_data: Dict[str, Any], scenario_id: str, config: Dict[str, Any], 
                         logger=None) -> Dict[str, Any]:
    """
    Simpan hasil evaluasi ke Google Drive
    
    Args:
        results_data: Dictionary berisi hasil evaluasi yang akan disimpan
        scenario_id: ID skenario evaluasi
        config: Konfigurasi yang berisi definisi skenario
        logger: Logger untuk mencatat proses (opsional)
        
    Returns:
        Dict berisi informasi file yang disimpan di Google Drive
    """
    # Dapatkan path drive dan info skenario
    scenario_result = get_drive_path_for_scenario(scenario_id, config)
    
    if not scenario_result['success']:
        if logger:
            logger(f"⚠️ Gagal mendapatkan info skenario: {scenario_result.get('error', 'Unknown error')}", "warning")
        return {'success': False, 'error': scenario_result.get('error', 'Failed to get scenario info')}
    
    # Jika drive tidak diaktifkan
    if not scenario_result.get('drive_enabled', False):
        if logger:
            logger(f"⚠️ Penyimpanan ke Google Drive tidak diaktifkan untuk skenario {scenario_result['name']}", "warning")
        return {'success': False, 'error': 'Drive storage not enabled', 'scenario_info': scenario_result}
    
    # Dapatkan path dan info skenario
    drive_path = scenario_result['drive_path']
    scenario_info = scenario_result['info']
    
    try:
        # Simpan hasil ke drive
        if logger:
            logger(f"💾 Menyimpan hasil ke Google Drive: {drive_path}", "info")
        
        saved_files = save_evaluation_results(results_data, drive_path, scenario_info)
        
        if logger:
            logger(f"✅ Hasil berhasil disimpan ke Google Drive: {drive_path}", "success")
        
        return {
            'success': True, 
            'saved_files': saved_files, 
            'drive_path': drive_path,
            'scenario_info': scenario_result
        }
        
    except Exception as e:
        if logger:
            logger(f"❌ Gagal menyimpan hasil ke Google Drive: {str(e)}", "error")
        
        return {'success': False, 'error': str(e), 'scenario_info': scenario_result}

def _convert_numpy_to_python(obj):
    """
    Konversi numpy types ke Python native types untuk JSON serialization
    
    Args:
        obj: Object yang akan dikonversi
        
    Returns:
        Object yang sudah dikonversi
    """
    import numpy as np
    
    if isinstance(obj, dict):
        return {k: _convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_numpy_to_python(item) for item in obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                         np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj
