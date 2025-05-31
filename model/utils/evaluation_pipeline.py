"""
File: smartcash/model/utils/evaluation_pipeline.py
Deskripsi: Pipeline lengkap untuk evaluasi model dengan berbagai skenario
"""

import os
import time
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from concurrent.futures import ThreadPoolExecutor

from smartcash.model.utils.validation_utils import validate_evaluation_config
from smartcash.model.utils.model_loader import get_model_for_scenario, load_model_for_scenario
from smartcash.model.utils.evaluation_utils import run_model_inference, load_ground_truth_labels
from smartcash.model.utils.scenario_utils import get_drive_path_for_scenario, save_results_to_drive
from smartcash.dataset.utils.test_data_utils import prepare_test_data_for_scenario
from smartcash.dataset.augmentor.strategies.evaluation_augmentation import get_augmentation_pipeline

def run_evaluation_pipeline(
    evaluation_input: Dict[str, Any],
    progress_callback: Callable = None,
    status_callback: Callable = None,
    logger = None
) -> Dict[str, Any]:
    """
    Menjalankan pipeline lengkap evaluasi model dengan progress tracking dan error handling
    
    Args:
        evaluation_input: Dictionary berisi input untuk evaluasi
            - scenario_id: ID skenario evaluasi
            - test_folder: Path folder test data
            - ui_components: UI components untuk interaksi (opsional)
            - config: Konfigurasi evaluasi
        progress_callback: Callback untuk melaporkan progress (stage, progress, message)
        status_callback: Callback untuk melaporkan status (status, message)
        logger: Logger untuk logging
        
    Returns:
        Dict berisi hasil evaluasi
    """
    try:
        # Ekstrak input
        scenario_id = evaluation_input.get('scenario_id')
        test_folder = evaluation_input.get('test_folder')
        config = evaluation_input.get('config', {})
        
        # Fungsi helper untuk progress tracking
        def update_progress(stage, progress, message):
            if progress_callback:
                progress_callback(stage, progress, message)
            if logger:
                log_message = f"{message} ({progress}%)"
                logger(log_message, "info")
        
        # Fungsi helper untuk status reporting
        def report_status(status, message):
            if status_callback:
                status_callback(status, message)
            if logger:
                logger(message, status)
        
        # Step 1: Validasi input
        update_progress('overall', 0, "🚀 Memulai evaluasi model...")
        
        validation_result = validate_evaluation_config({
            'scenario_id': scenario_id,
            'test_folder': test_folder,
            'config': config
        })
        
        if not validation_result['valid']:
            report_status('error', validation_result['message'])
            return {'success': False, 'error': validation_result['message']}
        
        update_progress('overall', 10, "✅ Validasi input berhasil")
        
        # Step 2: Dapatkan info skenario
        scenario_info = get_drive_path_for_scenario(scenario_id, config)
        
        if not scenario_info['success']:
            report_status('error', f"❌ Gagal mendapatkan info skenario: {scenario_info['error']}")
            return {'success': False, 'error': scenario_info['error']}
        
        # Step 3: Load model dan checkpoint
        update_progress('overall', 15, f"🧠 Loading model untuk skenario {scenario_info['name']}...")
        
        model_info = get_model_for_scenario(scenario_id, config)
        
        if not model_info['success']:
            report_status('error', f"❌ Gagal mendapatkan info model: {model_info['error']}")
            return {'success': False, 'error': model_info['error']}
        
        model_result = load_model_for_scenario(scenario_id, config)
        
        if not model_result['success']:
            report_status('error', f"❌ Gagal load model: {model_result['error']}")
            return {'success': False, 'error': model_result['error']}
        
        model = model_result['model']
        
        update_progress('overall', 25, "✅ Model berhasil dimuat")
        
        # Step 4: Dapatkan augmentation pipeline
        update_progress('overall', 30, f"🔄 Mempersiapkan pipeline augmentasi untuk {scenario_info['name']}...")
        
        augmentation_info = get_augmentation_pipeline(scenario_id, config)
        
        if not augmentation_info['success']:
            report_status('error', f"❌ Gagal membuat pipeline augmentasi: {augmentation_info['error']}")
            return {'success': False, 'error': augmentation_info['error']}
        
        aug_pipeline = augmentation_info['pipeline']
        
        # Step 5: Prepare test data dengan augmentasi
        update_progress('overall', 35, "📊 Mempersiapkan data test...")
        
        batch_size = config.get('test_data', {}).get('batch_size', 8)
        img_size = config.get('test_data', {}).get('img_size', 416)
        
        test_data = prepare_test_data_for_scenario(
            test_folder=test_folder,
            scenario_info=scenario_info,
            augmentation_pipeline=aug_pipeline,
            batch_size=batch_size,
            img_size=img_size
        )
        
        if not test_data['success']:
            report_status('error', f"❌ Gagal prepare test data: {test_data['error']}")
            return {'success': False, 'error': test_data['error']}
        
        update_progress('overall', 45, f"✅ Test data siap: {test_data['count']} images")
        
        # Step 6: Load ground truth labels
        update_progress('overall', 50, "📋 Memuat ground truth labels...")
        
        class_names = config.get('evaluation', {}).get('class_names', [
            'Rp1000', 'Rp2000', 'Rp5000', 'Rp10000', 'Rp20000', 'Rp50000', 'Rp75000', 'Rp100000'
        ])
        
        image_paths = [str(img_file) for img_file in test_data['image_files']]
        labels_info = load_ground_truth_labels(image_paths, class_names)
        
        # Step 7: Run inference
        update_progress('overall', 55, "🔍 Menjalankan inferensi model...")
        
        # Progress callback untuk inferensi
        def inference_progress_callback(progress, message):
            update_progress('current', progress, message)
        
        # Dapatkan config untuk inferensi
        conf_thresh = config.get('test_data', {}).get('confidence_threshold', 0.25)
        iou_thresh = config.get('test_data', {}).get('iou_threshold', 0.45)
        
        # Tentukan device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Run inference
        inference_result = run_model_inference(
            model=model,
            dataloader=test_data['dataloader'],
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            device=device,
            progress_callback=inference_progress_callback
        )
        
        predictions = inference_result.get('predictions', [])
        avg_inference_time = inference_result.get('avg_inference_time', 0)
        fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        
        update_progress('overall', 75, f"✅ Inference selesai: {len(predictions)} prediksi, FPS: {fps:.2f}")
        
        # Step 8: Calculate metrics
        update_progress('overall', 80, "📊 Menghitung metrik evaluasi...")
        
        # Hitung metrics (mAP, precision, recall, dll)
        from smartcash.model.utils.metrics_utils import calculate_detection_metrics
        
        metrics_result = calculate_detection_metrics(
            predictions=predictions,
            ground_truth=labels_info,
            class_names=class_names,
            iou_threshold=iou_thresh
        )
        
        # Tambahkan inference metrics
        metrics_result['inference_metrics'] = {
            'avg_inference_time': avg_inference_time,
            'fps': fps,
            'device': device
        }
        
        # Step 9: Save results
        update_progress('overall', 90, "💾 Menyimpan hasil evaluasi...")
        
        # Prepare results data
        results_data = {
            'metrics': metrics_result,
            'predictions': predictions,
            'scenario_info': scenario_info,
            'model_info': model_info,
            'timestamp': time.time()
        }
        
        # Save results to drive
        save_result = save_results_to_drive(
            results_data=results_data,
            scenario_id=scenario_id,
            config=config
        )
        
        if not save_result['success']:
            report_status('warning', f"⚠️ Gagal menyimpan hasil ke drive: {save_result['error']}")
        
        # Step 10: Finalize
        update_progress('overall', 100, "✅ Evaluasi selesai")
        report_status('success', "✅ Evaluasi model berhasil diselesaikan")
        
        return {
            'success': True,
            'metrics': metrics_result,
            'predictions': predictions,
            'scenario_info': scenario_info,
            'model_info': model_info,
            'save_result': save_result
        }
        
    except Exception as e:
        if status_callback:
            status_callback('error', f"❌ Error dalam evaluasi: {str(e)}")
        if logger:
            logger(f"❌ Error dalam evaluasi: {str(e)}", "error")
        
        return {'success': False, 'error': str(e)}
