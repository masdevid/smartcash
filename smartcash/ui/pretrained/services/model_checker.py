"""
File: smartcash/ui/pretrained/services/model_checker.py
Deskripsi: Service untuk checking pretrained models
"""

import os
from typing import Optional
from smartcash.common.logger import get_logger

class PretrainedModelChecker:
    """Service untuk checking pretrained model files"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def check_model_exists(self, model_path: str, min_size_mb: Optional[int] = None) -> bool:
        """Check jika model file exists dan valid"""
        try:
            if not os.path.exists(model_path):
                return False
            
            # Check file size jika min_size specified
            if min_size_mb:
                file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
                if file_size_mb < min_size_mb:
                    self.logger.warning(f"⚠️ Model file too small: {file_size_mb:.1f}MB < {min_size_mb}MB")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error checking model: {str(e)}")
            return False
