"""
File: smartcash/ui/dataset/download/utils/download_executor.py  
Deskripsi: Updated download executor dengan observer progress
"""

from typing import Dict, Any
from smartcash.ui.dataset.download.services.ui_download_service_final import UIDownloadServiceFinal

def execute_roboflow_download(ui_components: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Eksekusi download dengan observer progress tracking (no tqdm)."""
    
    try:
        # Create final UI download service dengan progress callback
        download_service = UIDownloadServiceFinal(ui_components)
        
        # Execute download (progress via observer + callback)
        result = download_service.download_dataset(params)
        
        return result
        
    except Exception as e:
        logger = ui_components.get('logger')
        if logger:
            logger.error(f"❌ Error download: {str(e)}")
        return {'status': 'error', 'message': str(e)}