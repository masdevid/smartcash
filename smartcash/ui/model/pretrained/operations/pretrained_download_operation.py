"""
Download operation for pretrained models.
"""

import os
import urllib.request
from typing import Dict, Any

from .pretrained_base_operation import PretrainedBaseOperation
from smartcash.ui.model.pretrained.constants import DEFAULT_MODEL_URLS, PretrainedModelType


class PretrainedDownloadOperation(PretrainedBaseOperation):
    """Download operation for pretrained models."""
    
    def execute_operation(self) -> Dict[str, Any]:
        """Execute download operation with actual model downloads."""
        try:
            self.log(f"📁 Using models directory: {self.models_dir}", 'info')
            
            # Step 1: Check existing models
            self.log("🔍 Checking existing models", 'info')
            existing_models = self.check_existing_models(self.models_dir)
            
            # Step 2: Prepare download directory
            self.log("📁 Preparing download directory", 'info')
            os.makedirs(self.models_dir, exist_ok=True)
            
            downloaded_models = []
            
            # Step 3: Download YOLOv5s model
            yolo_path = os.path.join(self.models_dir, 'yolov5s.pt')
            if not os.path.exists(yolo_path) or 'yolov5s' not in existing_models:
                self.log("📥 Downloading YOLOv5s model", 'info')
                yolo_url = self.config.get('yolo_url') or DEFAULT_MODEL_URLS[PretrainedModelType.YOLOV5S.value]
                
                if yolo_url:
                    urllib.request.urlretrieve(yolo_url, yolo_path)
                    downloaded_models.append('yolov5s')
                    self.log(f"✅ YOLOv5s downloaded to {yolo_path}", 'success')
                else:
                    self.log("⚠️ YOLOv5s URL not provided, skipping", 'warning')
            else:
                self.log("✅ YOLOv5s already exists", 'info')
            
            # Step 4: Download EfficientNet-B4 model
            efficientnet_path = os.path.join(self.models_dir, 'efficientnet_b4.pth')
            if not os.path.exists(efficientnet_path) or 'efficientnet_b4' not in existing_models:
                self.log("📥 Downloading EfficientNet-B4 model", 'info')
                efficientnet_url = self.config.get('efficientnet_url')
                
                if efficientnet_url:
                    # Download from custom URL
                    urllib.request.urlretrieve(efficientnet_url, efficientnet_path)
                    downloaded_models.append('efficientnet_b4')
                    self.log(f"✅ EfficientNet-B4 downloaded to {efficientnet_path}", 'success')
                else:
                    # Use timm to download
                    try:
                        import timm
                        import torch
                        
                        model = timm.create_model('efficientnet_b4', pretrained=True)
                        torch.save(model.state_dict(), efficientnet_path)
                        downloaded_models.append('efficientnet_b4')
                        self.log(f"✅ EfficientNet-B4 downloaded via timm to {efficientnet_path}", 'success')
                    except Exception as e:
                        self.log(f"⚠️ Failed to download EfficientNet-B4 via timm: {e}", 'warning')
            else:
                self.log("✅ EfficientNet-B4 already exists", 'info')
            
            # Step 5: Validate downloaded models
            self.log("🔍 Validating downloaded models", 'info')
            validation_results = self.validate_downloaded_models(self.models_dir)
            
            # Step 6: Complete
            self.log("✅ Download complete", 'success')
            
            return {
                'success': True,
                'message': f'Model download completed. Downloaded: {", ".join(downloaded_models) if downloaded_models else "No new models"}',
                'models_downloaded': downloaded_models,
                'validation_results': validation_results
            }
            
        except Exception as e:
            self.log(f"❌ Error in download operation: {e}", 'error')
            return {'success': False, 'error': str(e)}