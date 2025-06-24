"""
File: smartcash/model/utils/validation_utils.py
Deskripsi: Utilitas untuk validasi input dan konfigurasi evaluasi model
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

def validate_evaluation_config(validation_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validasi konfigurasi dan input untuk evaluasi model
    
    Args:
        validation_input: Dictionary berisi input yang akan divalidasi
            - scenario_id: ID skenario evaluasi
            - test_folder: Path folder test data
            - config: Konfigurasi evaluasi
            
    Returns:
        Dict berisi hasil validasi
    """
    try:
        # Ekstrak input
        scenario_id = validation_input.get('scenario_id')
        test_folder = validation_input.get('test_folder')
        config = validation_input.get('config', {})
        
        # Validasi skenario
        if not scenario_id:
            return {'valid': False, 'message': "❌ Silakan pilih skenario evaluasi terlebih dahulu"}
        
        # Validasi test folder
        if not test_folder or not os.path.exists(test_folder):
            return {'valid': False, 'message': "❌ Test data folder tidak valid atau tidak ditemukan"}
        
        # Validasi images folder
        images_folder = os.path.join(test_folder, 'images')
        if not os.path.exists(images_folder):
            # Coba gunakan folder test langsung jika tidak ada subfolder images
            images_folder = test_folder
            if not any(ext in str(f) for f in Path(images_folder).iterdir() for ext in ['.jpg', '.jpeg', '.png']):
                return {'valid': False, 'message': f"❌ Tidak ada gambar di folder test data: {test_folder}"}
        
        # Hitung jumlah gambar
        image_count = 0
        for ext in ['.jpg', '.jpeg', '.png']:
            image_count += len(list(Path(images_folder).glob(f"*{ext}")))
        
        if image_count == 0:
            return {'valid': False, 'message': "❌ Tidak ada gambar di folder test data"}
        
        # Validasi konfigurasi model
        model_config = config.get('model', {})
        if not model_config:
            # Tidak fatal, akan menggunakan default
            pass
        
        # Validasi konfigurasi test data
        test_config = config.get('test_data', {})
        if not test_config:
            # Tidak fatal, akan menggunakan default
            pass
        
        return {
            'valid': True, 
            'message': "✅ Input valid", 
            'image_count': image_count,
            'images_folder': images_folder
        }
        
    except Exception as e:
        return {'valid': False, 'message': f"❌ Error validasi: {str(e)}"}
