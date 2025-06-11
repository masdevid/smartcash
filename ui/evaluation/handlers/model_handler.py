"""
File: smartcash/ui/evaluation/handlers/model_handler.py
Deskripsi: Handler UI untuk memuat model berdasarkan skenario
"""

from typing import Dict, Any
from smartcash.ui.utils.logger_bridge import log_to_service
from smartcash.model.utils.model_loader import get_model_for_scenario, load_model_for_scenario as load_model_core

def load_model_for_scenario(scenario_info: Dict[str, Any], config: Dict[str, Any], device: str = 'cuda', logger=None) -> Dict[str, Any]:
    """UI wrapper untuk memuat model berdasarkan skenario dengan one-liner style.
    
    PENTING: Fungsi ini harus mengembalikan hasil dari load_model_core tanpa modifikasi apapun
    untuk kompatibilitas dengan test integrasi.
    """
    # Wrapper untuk log_to_service jika logger diberikan dengan one-liner
    def log_wrapper(message, level="info"): logger and log_to_service(logger, message, level)
    
    try:
        # Dapatkan informasi model dengan one-liner
        model_info = get_model_for_scenario(scenario_info, config)
        
        # Log informasi model dengan one-liner
        log_wrapper(f"🔍 Memuat model dengan backbone: {model_info.get('backbone', 'unknown')}")
        source_message = {
            'custom': f"Menggunakan checkpoint kustom: {model_info.get('checkpoint_path', 'unknown')}",
            'drive': f"Menggunakan model dari Google Drive: {model_info.get('checkpoint_path', 'unknown')}"
        }.get(model_info.get('source', ''), f"Menggunakan model pretrained default untuk {model_info.get('backbone', 'unknown')}")
        log_wrapper(f"📦 {source_message}")
        
        # PENTING: Delegasikan ke implementasi di model module dan kembalikan hasil tanpa modifikasi
        # untuk kompatibilitas dengan test integrasi
        result = load_model_core(scenario_info, config, device)
        
        # Log hasil dengan one-liner
        if result.get('success', False):
            log_wrapper(f"✅ Model berhasil dimuat: {model_info.get('backbone', 'unknown')}", "success")
        else:
            log_wrapper(f"❌ Gagal memuat model: {result.get('error', 'Unknown error')}", "error")
            
        # PENTING: Kembalikan result tanpa modifikasi apapun untuk kompatibilitas dengan test
        return result
    except Exception as e:
        # Log error dengan one-liner
        log_wrapper(f"❌ Gagal memuat model: {str(e)}", "error")
        return {'success': False, 'error': str(e)}
