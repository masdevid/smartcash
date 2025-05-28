"""
File: smartcash/ui/evaluation/handlers/evaluation_handler.py
Deskripsi: Handler untuk proses evaluasi model dengan augmentasi test data dan inference
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor
from smartcash.ui.utils.logger_bridge import log_to_service
from smartcash.ui.utils.button_state_manager import get_button_state_manager
from smartcash.dataset.utils.augmentation_utils import create_inference_augmentation_pipeline

def setup_evaluation_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None):
    """Setup handlers untuk evaluation process dengan one-liner pattern"""
    logger = ui_components.get('logger')
    button_manager = get_button_state_manager(ui_components)
    
    # One-liner handler untuk evaluate button
    ui_components['evaluate_button'].on_click(
        lambda b: run_evaluation_process(ui_components, config, logger, button_manager)
    )
    
    # Progress tracking handlers
    setup_progress_handlers(ui_components)
    
    return ui_components

def run_evaluation_process(ui_components: Dict[str, Any], config: Dict[str, Any], logger, button_manager) -> None:
    """Jalankan proses evaluasi dengan progress tracking dan error handling"""
    
    with button_manager.operation_context('evaluation'):
        try:
            # Show progress container
            ui_components.get('show_for_operation', lambda x: None)('evaluation')
            ui_components.get('update_progress', lambda *args: None)('overall', 0, "🚀 Memulai evaluasi model...")
            
            # Step 1: Validate inputs
            validation_result = validate_evaluation_inputs(ui_components, config, logger)
            if not validation_result['valid']:
                ui_components.get('error_operation', lambda x: None)(validation_result['message'])
                return
            
            ui_components.get('update_progress', lambda *args: None)('overall', 10, "✅ Validasi input berhasil")
            
            # Step 2: Load model dan checkpoint
            model_info = load_model_and_checkpoint(ui_components, config, logger)
            if not model_info['success']:
                ui_components.get('error_operation', lambda x: None)(f"❌ Gagal load model: {model_info['error']}")
                return
            
            ui_components.get('update_progress', lambda *args: None)('overall', 25, "🧠 Model berhasil dimuat")
            
            # Step 3: Prepare test data dengan augmentasi
            test_data = prepare_test_data_with_augmentation(ui_components, config, logger)
            if not test_data['success']:
                ui_components.get('error_operation', lambda x: None)(f"❌ Gagal prepare test data: {test_data['error']}")
                return
            
            ui_components.get('update_progress', lambda *args: None)('overall', 50, f"📊 Test data siap: {test_data['count']} images")
            
            # Step 4: Run inference
            predictions = run_model_inference(model_info['model'], test_data['dataloader'], ui_components, config, logger)
            if not predictions['success']:
                ui_components.get('error_operation', lambda x: None)(f"❌ Inference gagal: {predictions['error']}")
                return
            
            ui_components.get('update_progress', lambda *args: None)('overall', 75, f"🎯 Inference selesai: {len(predictions['results'])} prediksi")
            
            # Step 5: Calculate metrics dan save results
            metrics_result = calculate_and_save_metrics(predictions['results'], test_data['labels'], ui_components, config, logger)
            if not metrics_result['success']:
                ui_components.get('error_operation', lambda x: None)(f"❌ Gagal hitung metrics: {metrics_result['error']}")
                return
            
            ui_components.get('update_progress', lambda *args: None)('overall', 100, "🎉 Evaluasi selesai!")
            
            # Update UI dengan results
            update_results_ui(ui_components, metrics_result['metrics'], predictions['results'], logger)
            ui_components.get('complete_operation', lambda x: None)("✅ Evaluasi model berhasil diselesaikan!")
            
        except Exception as e:
            log_to_service(logger, f"❌ Error dalam proses evaluasi: {str(e)}", "error")
            ui_components.get('error_operation', lambda x: None)(f"❌ Error: {str(e)}")

def validate_evaluation_inputs(ui_components: Dict[str, Any], config: Dict[str, Any], logger) -> Dict[str, Any]:
    """Validasi input untuk evaluasi dengan one-liner checks"""
    
    # Extract UI values dengan one-liner
    checkpoint_path = ui_components.get('checkpoint_path_text', {}).get('value', '')
    test_folder = ui_components.get('test_folder_text', {}).get('value', 'data/test')
    
    # One-liner validation checks
    checks = [
        (checkpoint_path and os.path.exists(checkpoint_path), f"Checkpoint tidak ditemukan: {checkpoint_path}"),
        (os.path.exists(test_folder), f"Folder test tidak ditemukan: {test_folder}"),
        (len(os.listdir(test_folder)) > 0 if os.path.exists(test_folder) else False, f"Folder test kosong: {test_folder}")
    ]
    
    # Check dan return hasil
    failed_checks = [msg for valid, msg in checks if not valid]
    
    if failed_checks:
        log_to_service(logger, f"❌ Validasi gagal: {failed_checks[0]}", "error")
        return {'valid': False, 'message': failed_checks[0]}
    
    log_to_service(logger, "✅ Validasi input berhasil", "success")
    return {'valid': True, 'checkpoint_path': checkpoint_path, 'test_folder': test_folder}

def load_model_and_checkpoint(ui_components: Dict[str, Any], config: Dict[str, Any], logger) -> Dict[str, Any]:
    """Load YOLOv5 model dengan checkpoint dan prepare untuk inference"""
    
    try:
        ui_components.get('update_progress', lambda *args: None)('step', 0, "🔄 Loading checkpoint...")
        
        checkpoint_path = ui_components.get('checkpoint_path_text', {}).get('value', '')
        
        # Load checkpoint dengan torch
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        ui_components.get('update_progress', lambda *args: None)('step', 33, "🧠 Initializing model...")
        
        # Extract model dari checkpoint
        model = checkpoint.get('model')
        if model is None:
            # Fallback: load from ema jika ada
            model = checkpoint.get('ema', {}).get('model') if 'ema' in checkpoint else None
        
        if model is None:
            return {'success': False, 'error': 'Model tidak ditemukan dalam checkpoint'}
        
        # Set model ke evaluation mode
        model.eval()
        
        ui_components.get('update_progress', lambda *args: None)('step', 66, "⚙️ Configuring inference...")
        
        # Configure inference settings dari UI
        conf_thresh = ui_components.get('confidence_slider', {}).get('value', 0.25)
        iou_thresh = ui_components.get('iou_slider', {}).get('value', 0.45)
        img_size = ui_components.get('image_size_dropdown', {}).get('value', 416)
        
        # Set inference parameters
        if hasattr(model, 'model') and hasattr(model.model[-1], 'conf'):
            model.model[-1].conf = conf_thresh
            model.model[-1].iou = iou_thresh
        
        ui_components.get('update_progress', lambda *args: None)('step', 100, "✅ Model ready")
        
        model_info = {
            'model': model,
            'checkpoint': checkpoint,
            'config': {
                'conf_thresh': conf_thresh,
                'iou_thresh': iou_thresh,
                'img_size': img_size,
                'nc': checkpoint.get('nc', 10),
                'names': checkpoint.get('names', [])
            }
        }
        
        log_to_service(logger, f"✅ Model loaded: {checkpoint.get('nc', 10)} classes, img_size={img_size}", "success")
        return {'success': True, 'model': model_info}
        
    except Exception as e:
        log_to_service(logger, f"❌ Error loading model: {str(e)}", "error")
        return {'success': False, 'error': str(e)}

def prepare_test_data_with_augmentation(ui_components: Dict[str, Any], config: Dict[str, Any], logger) -> Dict[str, Any]:
    """Prepare test data dengan augmentasi untuk inference"""
    
    try:
        ui_components.get('update_progress', lambda *args: None)('step', 0, "📁 Scanning test images...")
        
        test_folder = ui_components.get('test_folder_text', {}).get('value', 'data/test')
        apply_aug = ui_components.get('apply_augmentation_checkbox', {}).get('value', True)
        batch_size = ui_components.get('batch_size_slider', {}).get('value', 16)
        img_size = ui_components.get('image_size_dropdown', {}).get('value', 416)
        
        # Scan images dengan one-liner
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_files = [f for ext in image_extensions 
                      for f in Path(test_folder).rglob(f"*{ext}")]
        
        if not image_files:
            return {'success': False, 'error': f'Tidak ada image ditemukan di {test_folder}'}
        
        ui_components.get('update_progress', lambda *args: None)('step', 25, f"📊 Found {len(image_files)} images")
        
        # Create augmentation pipeline untuk inference
        aug_pipeline = None
        if apply_aug:
            aug_pipeline = create_inference_augmentation_pipeline(img_size)
            log_to_service(logger, "🔄 Augmentation pipeline created untuk inference", "info")
        
        ui_components.get('update_progress', lambda *args: None)('step', 50, "🔄 Creating dataloader...")
        
        # Create custom dataloader untuk inference
        dataloader = create_inference_dataloader(image_files, aug_pipeline, batch_size, img_size)
        
        ui_components.get('update_progress', lambda *args: None)('step', 75, "📋 Loading ground truth labels...")
        
        # Load ground truth labels jika ada
        labels = load_ground_truth_labels(test_folder, image_files, logger)
        
        ui_components.get('update_progress', lambda *args: None)('step', 100, "✅ Test data ready")
        
        log_to_service(logger, f"✅ Test data prepared: {len(image_files)} images, batch_size={batch_size}", "success")
        
        return {
            'success': True,
            'dataloader': dataloader,
            'image_files': image_files,
            'labels': labels,
            'count': len(image_files),
            'augmentation': apply_aug
        }
        
    except Exception as e:
        log_to_service(logger, f"❌ Error preparing test data: {str(e)}", "error")
        return {'success': False, 'error': str(e)}

def create_inference_dataloader(image_files: List[Path], aug_pipeline, batch_size: int, img_size: int):
    """Create custom dataloader untuk inference dengan augmentation"""
    import torch
    from torch.utils.data import Dataset, DataLoader
    import cv2
    from PIL import Image
    
    class InferenceDataset(Dataset):
        def __init__(self, image_files, aug_pipeline, img_size):
            self.image_files = image_files
            self.aug_pipeline = aug_pipeline
            self.img_size = img_size
        
        def __len__(self):
            return len(self.image_files)
        
        def __getitem__(self, idx):
            img_path = self.image_files[idx]
            
            # Load image
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply augmentation jika ada
            if self.aug_pipeline:
                augmented = self.aug_pipeline(image=image)
                image = augmented['image']
            
            # Resize dan normalize untuk YOLO
            image = cv2.resize(image, (self.img_size, self.img_size))
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)  # HWC to CHW
            
            return image, str(img_path)
    
    dataset = InferenceDataset(image_files, aug_pipeline, img_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

def run_model_inference(model_info: Dict[str, Any], dataloader, ui_components: Dict[str, Any], config: Dict[str, Any], logger) -> Dict[str, Any]:
    """Run inference pada test data dengan progress tracking"""
    
    try:
        model = model_info['model']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        predictions = []
        total_batches = len(dataloader)
        
        log_to_service(logger, f"🎯 Running inference pada {device}: {total_batches} batches", "info")
        
        with torch.no_grad():
            for batch_idx, (images, paths) in enumerate(dataloader):
                # Update progress
                progress = int((batch_idx / total_batches) * 100)
                ui_components.get('update_progress', lambda *args: None)('current', progress, f"🔄 Batch {batch_idx+1}/{total_batches}")
                
                # Move to device
                images = images.to(device)
                
                # Forward pass
                outputs = model(images)
                
                # Process outputs dengan NMS
                processed_outputs = apply_nms(outputs, model_info['config'])
                
                # Store predictions dengan metadata
                for i, (output, path) in enumerate(zip(processed_outputs, paths)):
                    predictions.append({
                        'image_path': path,
                        'predictions': output,
                        'batch_idx': batch_idx,
                        'image_idx': i
                    })
        
        ui_components.get('update_progress', lambda *args: None)('current', 100, "✅ Inference completed")
        
        log_to_service(logger, f"✅ Inference selesai: {len(predictions)} prediksi generated", "success")
        
        return {'success': True, 'results': predictions}
        
    except Exception as e:
        log_to_service(logger, f"❌ Error dalam inference: {str(e)}", "error")
        return {'success': False, 'error': str(e)}

def apply_nms(outputs, config: Dict[str, Any]):
    """Apply Non-Maximum Suppression pada model outputs"""
    try:
        from yolov5.utils.general import non_max_suppression
        
        # Apply NMS dengan config dari UI
        return non_max_suppression(
            outputs,
            conf_thres=config['conf_thresh'],
            iou_thres=config['iou_thresh'],
            classes=None,
            agnostic=False,
            max_det=1000
        )
    except ImportError:
        # Fallback NMS implementation
        return simple_nms(outputs, config)

def simple_nms(outputs, config: Dict[str, Any]):
    """Simple NMS implementation sebagai fallback"""
    import torchvision.ops as ops
    
    processed = []
    conf_thresh = config['conf_thresh']
    iou_thresh = config['iou_thresh']
    
    for output in outputs:
        if output is None or len(output) == 0:
            processed.append(torch.empty((0, 6)))
            continue
        
        # Filter by confidence
        mask = output[:, 4] > conf_thresh
        output = output[mask]
        
        if len(output) == 0:
            processed.append(torch.empty((0, 6)))
            continue
        
        # Apply NMS
        boxes = output[:, :4]
        scores = output[:, 4]
        keep = ops.nms(boxes, scores, iou_thresh)
        
        processed.append(output[keep])
    
    return processed

def load_ground_truth_labels(test_folder: str, image_files: List[Path], logger) -> Dict[str, Any]:
    """Load ground truth labels untuk evaluation metrics"""
    
    try:
        labels = {}
        labels_folder = Path(test_folder) / 'labels'
        
        if not labels_folder.exists():
            log_to_service(logger, "⚠️ Folder labels tidak ditemukan, akan skip perhitungan mAP", "warning")
            return {'available': False, 'labels': {}}
        
        # Load label files dengan one-liner
        for img_file in image_files:
            label_file = labels_folder / f"{img_file.stem}.txt"
            if label_file.exists():
                with open(label_file, 'r') as f:
                    # Parse YOLO format: class x_center y_center width height
                    lines = [list(map(float, line.strip().split())) for line in f.readlines() if line.strip()]
                    labels[str(img_file)] = lines
        
        log_to_service(logger, f"📋 Loaded {len(labels)} label files dari {len(image_files)} images", "info")
        
        return {'available': True, 'labels': labels, 'count': len(labels)}
        
    except Exception as e:
        log_to_service(logger, f"⚠️ Error loading labels: {str(e)}", "warning")
        return {'available': False, 'labels': {}, 'error': str(e)}

def setup_progress_handlers(ui_components: Dict[str, Any]) -> None:
    """Setup handlers untuk progress tracking selama evaluation"""
    
    # Progress update handlers sudah di-handle di create_progress_tracking_container
    # Ini untuk additional setup jika diperlukan
    
    if 'progress_container' in ui_components:
        # Hide progress container initially
        ui_components['progress_container'].layout.display = 'none'