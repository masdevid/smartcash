"""
File: smartcash/model/utils/model_loader.py
Deskripsi: Utilitas untuk memuat model YOLOv5 dengan berbagai backbone untuk evaluasi
"""

import os
import torch
from typing import Dict, Any, Optional, Union

def check_pretrained_model_in_drive(backbone: str) -> Optional[str]:
    """
    Periksa ketersediaan model pretrained di Google Drive
    
    Args:
        backbone: Tipe backbone ('cspdarknet_s' atau 'efficientnet_b4')
        
    Returns:
        Path ke model pretrained jika tersedia, None jika tidak
    """
    # Mapping backbone ke nama file
    backbone_to_file = {
        'cspdarknet_s': 'yolov5s.pt',
        'efficientnet_b4': 'efficientnet_b4_huggingface.bin'
    }
    
    if backbone not in backbone_to_file:
        return None
    
    # Path ke direktori model di Google Drive
    drive_model_dir = '/content/drive/MyDrive/SmartCash/models'
    
    # Path lengkap ke file model
    model_path = os.path.join(drive_model_dir, backbone_to_file[backbone])
    
    # Periksa keberadaan file
    if os.path.exists(model_path):
        return model_path
    
    return None

def get_model_for_scenario(scenario_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dapatkan model checkpoint berdasarkan skenario
    
    Args:
        scenario_id: ID skenario evaluasi
        config: Konfigurasi yang berisi definisi model
        
    Returns:
        Dict berisi informasi model
    """
    # Dapatkan informasi skenario
    from smartcash.model.utils.scenario_utils import get_scenario_info
    
    scenario_info = get_scenario_info(scenario_id, config)
    
    if not scenario_info:
        # Gunakan default jika skenario tidak ditemukan
        return {
            'success': False,
            'error': f"Skenario dengan ID {scenario_id} tidak ditemukan",
            'backbone': 'cspdarknet_s',  # Default backbone
            'source': 'default'
        }
    
    # Dapatkan backbone dari skenario
    backbone = scenario_info.get('backbone', 'cspdarknet_s')
    
    # Periksa apakah ada checkpoint kustom yang dipilih
    use_custom_checkpoint = config.get('model', {}).get('use_custom_checkpoint', False)
    custom_checkpoint_path = config.get('model', {}).get('custom_checkpoint_path', '')
    
    if use_custom_checkpoint and custom_checkpoint_path and os.path.exists(custom_checkpoint_path):
        # Gunakan checkpoint kustom
        return {
            'checkpoint_path': custom_checkpoint_path,
            'backbone': backbone,
            'source': 'custom',
            'exists': True
        }
    
    # Periksa apakah model pretrained tersedia di Google Drive
    drive_model_path = check_pretrained_model_in_drive(backbone)
    
    if drive_model_path:
        # Gunakan model dari Google Drive
        return {
            'checkpoint_path': drive_model_path,
            'backbone': backbone,
            'source': 'drive',
            'exists': True
        }
    
    # Gunakan model pretrained default
    return {
        'success': True,
        'checkpoint_path': None,  # Akan menggunakan pretrained default
        'backbone': backbone,
        'source': 'default',
        'exists': False,
        'scenario_id': scenario_id
    }

def load_model_for_scenario(scenario_id: str, config: Dict[str, Any], 
                           device: str = 'cuda', logger=None) -> Dict[str, Any]:
    """
    Muat model berdasarkan skenario
    
    Args:
        scenario_id: ID skenario evaluasi
        config: Konfigurasi yang berisi definisi model
        device: Device untuk inferensi ('cuda' atau 'cpu')
        logger: Logger untuk mencatat proses (opsional)
        
    Returns:
        Dict berisi model dan metadata
    """
    # Dapatkan informasi model
    model_info = get_model_for_scenario(scenario_id, config)
    
    # Cek apakah berhasil mendapatkan info model
    if not model_info.get('success', True):
        if logger:
            logger(f"❌ Gagal mendapatkan info model: {model_info.get('error', 'Unknown error')}", "error")
        return {'success': False, 'error': model_info.get('error', 'Failed to get model info')}
    
    # Log informasi model
    if logger:
        logger(f"🔍 Memuat model dengan backbone: {model_info['backbone']}", "info")
        if model_info['source'] == 'custom':
            logger(f"📦 Menggunakan checkpoint kustom: {model_info['checkpoint_path']}", "info")
        elif model_info['source'] == 'drive':
            logger(f"📦 Menggunakan model dari Google Drive: {model_info['checkpoint_path']}", "info")
        else:
            logger(f"📦 Menggunakan model pretrained default untuk {model_info['backbone']}", "info")
    
    try:
        # Muat model
        model = load_model(
            model_info['checkpoint_path'], 
            backbone=model_info['backbone'],
            img_size=config.get('model', {}).get('img_size', 416),
            device=device
        )
        
        # Log sukses
        if logger:
            logger(f"✅ Model berhasil dimuat", "success")
        
        return {
            'success': True,
            'model': model,
            'backbone': model_info['backbone'],
            'source': model_info['source'],
            'img_size': config.get('model', {}).get('img_size', 416),
            'scenario_id': scenario_id
        }
        
    except Exception as e:
        # Log error
        if logger:
            logger(f"❌ Gagal memuat model: {str(e)}", "error")
        
        return {'success': False, 'error': str(e), 'scenario_id': scenario_id}

def load_model(checkpoint_path: Optional[str] = None, backbone: str = 'cspdarknet_s', 
              img_size: int = 416, device: str = 'cuda') -> torch.nn.Module:
    """
    Muat model YOLOv5 dengan backbone tertentu
    
    Args:
        checkpoint_path: Path ke checkpoint model, None untuk pretrained default
        backbone: Tipe backbone ('cspdarknet_s' atau 'efficientnet_b4')
        img_size: Target image size
        device: Device untuk inferensi ('cuda' atau 'cpu')
        
    Returns:
        Model YOLOv5
    """
    # Validasi device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    if backbone == 'efficientnet_b4':
        # Muat model YOLOv5 dengan EfficientNet-B4 backbone
        model = load_yolov5_with_efficientnet(checkpoint_path, img_size, device)
    else:
        # Muat model YOLOv5 dengan CSPDarknet backbone (default)
        model = load_yolov5_with_cspdarknet(checkpoint_path, img_size, device)
    
    # Pindahkan model ke device yang sesuai
    model = model.to(device)
    
    # Set model ke mode evaluasi
    model.eval()
    
    return model

def load_yolov5_with_cspdarknet(checkpoint_path: Optional[str] = None, 
                              img_size: int = 416, 
                              device: str = 'cuda') -> torch.nn.Module:
    """
    Muat model YOLOv5 dengan CSPDarknet backbone
    
    Args:
        checkpoint_path: Path ke checkpoint model, None untuk pretrained default
        img_size: Target image size
        device: Device untuk inferensi ('cuda' atau 'cpu')
        
    Returns:
        Model YOLOv5 dengan CSPDarknet backbone
    """
    try:
        # Coba import YOLOv5 dari repo
        import sys
        if '/content/yolov5' not in sys.path:
            sys.path.append('/content/yolov5')
        
        from models.experimental import attempt_load
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            # Muat model dari checkpoint
            model = attempt_load(checkpoint_path, device=device)
        else:
            # Muat model pretrained default
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        
        # Resize model ke target size
        model.eval()
        
        return model
        
    except Exception as e:
        # Fallback ke torch hub jika import dari repo gagal
        try:
            if checkpoint_path and os.path.exists(checkpoint_path):
                # Muat model dari checkpoint dengan torch hub
                model = torch.hub.load('ultralytics/yolov5', 'custom', path=checkpoint_path)
            else:
                # Muat model pretrained default
                model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            
            model.eval()
            
            return model
            
        except Exception as e2:
            raise RuntimeError(f"Gagal memuat model YOLOv5 dengan CSPDarknet: {str(e)} -> {str(e2)}")

def load_yolov5_with_efficientnet(checkpoint_path: Optional[str] = None, 
                                img_size: int = 416, 
                                device: str = 'cuda') -> torch.nn.Module:
    """
    Muat model YOLOv5 dengan EfficientNet-B4 backbone
    
    Args:
        checkpoint_path: Path ke checkpoint model, None untuk pretrained default
        img_size: Target image size
        device: Device untuk inferensi ('cuda' atau 'cpu')
        
    Returns:
        Model YOLOv5 dengan EfficientNet-B4 backbone
    """
    try:
        # Coba import YOLOv5 dari repo
        import sys
        if '/content/yolov5' not in sys.path:
            sys.path.append('/content/yolov5')
        
        # Import model kustom dengan EfficientNet backbone
        from smartcash.model.architectures.backbones.efficientnet_backbone import YOLOv5WithEfficientNet
        
        # Buat model
        model = YOLOv5WithEfficientNet(
            num_classes=len(['Rp1000', 'Rp2000', 'Rp5000', 'Rp10000', 'Rp20000', 'Rp50000', 'Rp75000', 'Rp100000']),
            img_size=img_size,
            efficientnet_variant='b4'
        )
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            # Muat weights dari checkpoint
            state_dict = torch.load(checkpoint_path, map_location=device)
            
            # Cek apakah state dict langsung atau dalam format tertentu
            if isinstance(state_dict, dict) and 'model' in state_dict:
                state_dict = state_dict['model']
            
            # Load state dict
            model.load_state_dict(state_dict, strict=False)
        
        model.eval()
        
        return model
        
    except Exception as e:
        raise RuntimeError(f"Gagal memuat model YOLOv5 dengan EfficientNet-B4: {str(e)}")
