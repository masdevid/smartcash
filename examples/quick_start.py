"""
SmartCash YOLOv5 Quick Start Example
Minimal example to get started quickly
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from smartcash.model.architectures.smartcash_yolov5 import create_smartcash_yolov5


def quick_start():
    """Minimal working example"""
    print("🚀 SmartCash YOLOv5 Quick Start")
    
    # 1. Create model (choose backbone)
    print("\n📦 Creating model...")
    model = create_smartcash_yolov5(
        backbone="yolov5s",      # or "efficientnet_b4" for higher accuracy
        pretrained=False,        # Set True for better starting point
        device="cpu"            # or "cuda" if available
    )
    print("✅ Model created!")
    
    # 2. Model information
    info = model.get_model_info()
    print(f"   Backbone: {info['backbone']}")
    print(f"   Parameters: {info['total_params']:,}")
    print(f"   Training Phase: {info['phase_info']['phase']}")
    
    # 3. Create test image
    print("\n🖼️ Creating test image...")
    test_image = torch.randn(1, 3, 640, 640)  # Batch of 1, RGB, 640x640
    
    # 4. Make prediction
    print("🔍 Making prediction...")
    model.model.eval()
    with torch.no_grad():
        results = model.predict(test_image)
    
    # 5. Show results
    if len(results) > 0:
        result = results[0]
        print("🎯 Results:")
        print(f"   Total detections: {result.get('total_detections', 0)}")
        
        # Show denomination scores
        denom_scores = result.get('denomination_scores', torch.zeros(7))
        denominations = ['001', '002', '005', '010', '020', '050', '100']
        print("   Denomination scores:")
        for name, score in zip(denominations, denom_scores):
            if score > 0.1:
                print(f"     {name}: {score:.3f}")
    else:
        print("🎯 No detections found")
    
    print("\n✅ Quick start completed!")
    return model


if __name__ == "__main__":
    model = quick_start()