"""
SmartCash YOLOv5 Prediction Example
Complete prediction workflow with both backbones and real data
"""

import torch
import numpy as np
from PIL import Image
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from smartcash.model.architectures.smartcash_yolov5 import create_smartcash_yolov5
from smartcash.model.inference.post_prediction_mapper import PostPredictionMapper
from smartcash.common.logger import SmartCashLogger

logger = SmartCashLogger(__name__)


def load_test_image(image_path: str = None):
    """Load and preprocess test image"""
    if image_path and Path(image_path).exists():
        # Load real image
        image = Image.open(image_path).convert('RGB')
        logger.info(f"📷 Loaded image: {image_path} ({image.size})")
    else:
        # Create synthetic test image
        image = Image.new('RGB', (640, 640), color=(100, 150, 200))
        logger.info("📷 Created synthetic test image (640x640)")
    
    # Resize to model input size
    image = image.resize((640, 640))
    
    # Convert to tensor
    image_tensor = torch.from_numpy(np.array(image)).float()
    image_tensor = image_tensor.permute(2, 0, 1) / 255.0  # HWC -> CHW, normalize
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    
    return image, image_tensor


def predict_with_yolov5s():
    """Example: Prediction with YOLOv5s backbone"""
    logger.info("🔍 YOLOv5s Prediction Example")
    
    # Create model
    model = create_smartcash_yolov5(
        backbone="yolov5s",
        pretrained=False,
        device="cpu"
    )
    
    logger.info("📊 Model Info:")
    model_info = model.get_model_info()
    logger.info(f"   Backbone: {model_info['backbone']}")
    logger.info(f"   Parameters: {model_info['total_params']:,}")
    logger.info(f"   Phase: {model_info['phase_info']['phase']}")
    
    # Load test image
    image, image_tensor = load_test_image()
    
    # Make prediction
    logger.info("🚀 Running inference...")
    model.model.eval()
    
    with torch.no_grad():
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        import time
        start = time.time()
        
        # Get predictions
        results = model.predict(image_tensor)
        
        end = time.time()
        inference_time = end - start
    
    logger.info(f"⚡ Inference time: {inference_time*1000:.2f}ms")
    
    # Process results
    if len(results) > 0:
        result = results[0]
        logger.info("🎯 Prediction Results:")
        logger.info(f"   Total detections: {result.get('total_detections', 0)}")
        
        if 'detailed_results' in result:
            for detection in result['detailed_results'][:3]:  # Show first 3
                logger.info(f"   - {detection['denomination_name']}: "
                           f"confidence={detection['confidence']:.3f}, "
                           f"L2_support={detection['supporting_layer2_count']}, "
                           f"L3_support={detection['supporting_layer3_count']}")
        
        # Show denomination scores
        denom_scores = result.get('denomination_scores', torch.zeros(7))
        logger.info("💰 Denomination Scores:")
        denomination_names = ['001', '002', '005', '010', '020', '050', '100']
        for i, (name, score) in enumerate(zip(denomination_names, denom_scores)):
            if score > 0.1:  # Only show significant scores
                logger.info(f"   {name}: {score:.3f}")
    else:
        logger.info("🎯 No detections found")
    
    return model, results


def predict_with_efficientnet_b4():
    """Example: Prediction with EfficientNet-B4 backbone"""
    logger.info("🔍 EfficientNet-B4 Prediction Example")
    
    # Create model
    model = create_smartcash_yolov5(
        backbone="efficientnet_b4",
        pretrained=True,
        device="cpu"
    )
    
    logger.info("📊 Model Info:")
    model_info = model.get_model_info()
    logger.info(f"   Backbone: {model_info['backbone']}")
    logger.info(f"   Parameters: {model_info['total_params']:,}")
    logger.info(f"   Phase: {model_info['phase_info']['phase']}")
    
    # Load test image
    image, image_tensor = load_test_image()
    
    # Make prediction
    logger.info("🚀 Running inference...")
    model.model.eval()
    
    with torch.no_grad():
        import time
        start = time.time()
        
        # Get predictions
        results = model.predict(image_tensor)
        
        end = time.time()
        inference_time = end - start
    
    logger.info(f"⚡ Inference time: {inference_time*1000:.2f}ms")
    
    # Process results
    if len(results) > 0:
        result = results[0]
        logger.info("🎯 Prediction Results:")
        logger.info(f"   Total detections: {result.get('total_detections', 0)}")
        
        if 'detailed_results' in result:
            for detection in result['detailed_results'][:3]:  # Show first 3
                logger.info(f"   - {detection['denomination_name']}: "
                           f"confidence={detection['confidence']:.3f}, "
                           f"L2_support={detection['supporting_layer2_count']}, "
                           f"L3_support={detection['supporting_layer3_count']}")
        
        # Show denomination scores
        denom_scores = result.get('denomination_scores', torch.zeros(7))
        logger.info("💰 Denomination Scores:")
        denomination_names = ['001', '002', '005', '010', '020', '050', '100']
        for i, (name, score) in enumerate(zip(denomination_names, denom_scores)):
            if score > 0.1:  # Only show significant scores
                logger.info(f"   {name}: {score:.3f}")
    else:
        logger.info("🎯 No detections found")
    
    return model, results


def batch_prediction_example():
    """Example: Batch prediction with multiple images"""
    logger.info("📦 Batch Prediction Example")
    
    # Create model
    model = create_smartcash_yolov5(
        backbone="yolov5s",  # Use faster backbone for batch processing
        pretrained=False,
        device="cpu"
    )
    
    # Create batch of test images
    batch_size = 4
    images = []
    image_tensors = []
    
    for i in range(batch_size):
        # Create different colored test images
        color = (50 + i*50, 100 + i*30, 150 + i*20)
        image = Image.new('RGB', (640, 640), color=color)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(np.array(image)).float()
        image_tensor = image_tensor.permute(2, 0, 1) / 255.0
        
        images.append(image)
        image_tensors.append(image_tensor)
    
    # Stack into batch
    batch_tensor = torch.stack(image_tensors)
    
    logger.info(f"📦 Processing batch of {batch_size} images...")
    
    # Make batch prediction
    model.model.eval()
    with torch.no_grad():
        import time
        start = time.time()
        
        # Get batch predictions
        batch_results = model.predict(batch_tensor)
        
        end = time.time()
        inference_time = end - start
    
    logger.info(f"⚡ Batch inference time: {inference_time*1000:.2f}ms")
    logger.info(f"⚡ Average per image: {inference_time*1000/batch_size:.2f}ms")
    logger.info(f"🚀 Batch FPS: {batch_size/inference_time:.1f}")
    
    # Process batch results
    logger.info("🎯 Batch Results:")
    for i, result in enumerate(batch_results):
        total_detections = result.get('total_detections', 0)
        logger.info(f"   Image {i+1}: {total_detections} detections")
    
    return model, batch_results


def real_data_prediction_example():
    """Example: Prediction with real preprocessed data"""
    logger.info("💳 Real Data Prediction Example")
    
    # Check for real data
    test_data_dir = Path("data/preprocessed/test/images")
    if not test_data_dir.exists():
        logger.warning("⚠️ No real data found, creating synthetic example")
        return predict_with_yolov5s()
    
    # Find test images
    image_files = list(test_data_dir.glob("*.jpg"))[:3]  # Take first 3
    if not image_files:
        logger.warning("⚠️ No JPG images found in test data")
        return predict_with_yolov5s()
    
    logger.info(f"📁 Found {len(image_files)} test images")
    
    # Create model (use EfficientNet-B4 for higher accuracy on real data)
    model = create_smartcash_yolov5(
        backbone="efficientnet_b4",
        pretrained=True,
        device="cpu"
    )
    
    results_summary = []
    
    for image_file in image_files:
        logger.info(f"🔍 Processing: {image_file.name}")
        
        # Load and preprocess image
        image, image_tensor = load_test_image(str(image_file))
        
        # Make prediction
        model.model.eval()
        with torch.no_grad():
            import time
            start = time.time()
            results = model.predict(image_tensor)
            end = time.time()
        
        inference_time = end - start
        
        if len(results) > 0:
            result = results[0]
            total_detections = result.get('total_detections', 0)
            
            logger.info(f"   ⚡ Time: {inference_time*1000:.2f}ms")
            logger.info(f"   🎯 Detections: {total_detections}")
            
            # Show top detection
            if 'detailed_results' in result and result['detailed_results']:
                top_detection = result['detailed_results'][0]
                logger.info(f"   🏆 Top: {top_detection['denomination_name']} "
                           f"(conf={top_detection['confidence']:.3f})")
            
            results_summary.append({
                'image': image_file.name,
                'detections': total_detections,
                'time_ms': inference_time * 1000,
                'top_detection': top_detection['denomination_name'] if 'detailed_results' in result and result['detailed_results'] else 'None'
            })
    
    # Summary
    logger.info("📊 Real Data Summary:")
    total_time = sum(r['time_ms'] for r in results_summary)
    total_detections = sum(r['detections'] for r in results_summary)
    
    logger.info(f"   📷 Images processed: {len(results_summary)}")
    logger.info(f"   ⚡ Total time: {total_time:.2f}ms")
    logger.info(f"   📈 Average time: {total_time/len(results_summary):.2f}ms/image")
    logger.info(f"   🎯 Total detections: {total_detections}")
    
    return model, results_summary


def compare_backbone_prediction():
    """Compare prediction performance between backbones"""
    logger.info("⚖️ Backbone Prediction Comparison")
    
    backbones = ["yolov5s", "efficientnet_b4"]
    results = {}
    
    # Create test image
    image, image_tensor = load_test_image()
    
    for backbone in backbones:
        logger.info(f"🧪 Testing {backbone}")
        
        # Create model
        model = create_smartcash_yolov5(
            backbone=backbone,
            pretrained=(backbone == "efficientnet_b4"),
            device="cpu"
        )
        
        # Warm up
        model.model.eval()
        with torch.no_grad():
            _ = model.predict(image_tensor)
        
        # Timed prediction
        import time
        times = []
        for _ in range(5):  # Average over 5 runs
            start = time.time()
            with torch.no_grad():
                prediction_result = model.predict(image_tensor)
            end = time.time()
            times.append(end - start)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        # Process results
        if len(prediction_result) > 0:
            result = prediction_result[0]
            total_detections = result.get('total_detections', 0)
            max_confidence = 0
            if 'detailed_results' in result and result['detailed_results']:
                max_confidence = result['detailed_results'][0]['confidence']
        else:
            total_detections = 0
            max_confidence = 0
        
        model_info = model.get_model_info()
        results[backbone] = {
            'avg_time_ms': avg_time * 1000,
            'std_time_ms': std_time * 1000,
            'fps': 1 / avg_time,
            'total_params': model_info['total_params'],
            'detections': total_detections,
            'max_confidence': max_confidence
        }
        
        logger.info(f"   ⚡ Time: {avg_time*1000:.2f}±{std_time*1000:.2f}ms")
        logger.info(f"   🚀 FPS: {1/avg_time:.1f}")
        logger.info(f"   🎯 Detections: {total_detections}")
    
    # Comparison summary
    logger.info("📊 Comparison Results:")
    for backbone, result in results.items():
        logger.info(f"   {backbone}:")
        logger.info(f"     Speed: {result['fps']:.1f} FPS ({result['avg_time_ms']:.2f}ms)")
        logger.info(f"     Parameters: {result['total_params']:,}")
        logger.info(f"     Detections: {result['detections']}")
        logger.info(f"     Max Confidence: {result['max_confidence']:.3f}")
    
    return results


if __name__ == "__main__":
    try:
        # Example 1: YOLOv5s prediction
        logger.info("=" * 60)
        logger.info("EXAMPLE 1: YOLOv5s Prediction")
        logger.info("=" * 60)
        
        yolov5s_model, yolov5s_results = predict_with_yolov5s()
        
        # Example 2: EfficientNet-B4 prediction
        logger.info("\n" + "=" * 60)
        logger.info("EXAMPLE 2: EfficientNet-B4 Prediction")
        logger.info("=" * 60)
        
        efficientnet_model, efficientnet_results = predict_with_efficientnet_b4()
        
        # Example 3: Batch prediction
        logger.info("\n" + "=" * 60)
        logger.info("EXAMPLE 3: Batch Prediction")
        logger.info("=" * 60)
        
        batch_model, batch_results = batch_prediction_example()
        
        # Example 4: Real data prediction
        logger.info("\n" + "=" * 60)
        logger.info("EXAMPLE 4: Real Data Prediction")
        logger.info("=" * 60)
        
        real_model, real_results = real_data_prediction_example()
        
        # Example 5: Backbone comparison
        logger.info("\n" + "=" * 60)
        logger.info("EXAMPLE 5: Backbone Comparison")
        logger.info("=" * 60)
        
        comparison_results = compare_backbone_prediction()
        
        logger.info("\n🎉 All prediction examples completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Prediction example failed: {e}")
        import traceback
        traceback.print_exc()