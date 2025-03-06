"""
File: smartcash/handlers/ui_handlers/evaluation_handlers.py
Author: Alfrida Sabar
Deskripsi: Handler untuk UI komponen evaluasi model, menangani proses evaluasi dan visualisasi results.
"""

import gc
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from IPython.display import display, clear_output, HTML
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, Any, Optional, List, Callable, Union

@contextmanager
def memory_manager():
    """Context manager untuk mengoptimalkan penggunaan memori."""
    try:
        yield
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def init_ui(ui_components, checkpoint_handler):
    """
    Initialize UI with available models.
    
    Args:
        ui_components: Dictionary komponen UI dari create_evaluation_ui()
        checkpoint_handler: Instance dari CheckpointHandler
    """
    # Load available checkpoints
    if checkpoint_handler:
        checkpoints = checkpoint_handler.list_checkpoints()
        
        options = []
        
        # Add best checkpoints if available
        if checkpoints.get('best'):
            for ckpt in checkpoints.get('best', []):
                options.append((f"Best: {ckpt.name}", str(ckpt)))
        
        # Add latest checkpoints if available
        if checkpoints.get('latest'):
            for ckpt in checkpoints.get('latest', []):
                options.append((f"Latest: {ckpt.name}", str(ckpt)))
        
        # Add epoch checkpoints if available (limit to 5)
        if checkpoints.get('epoch'):
            for ckpt in checkpoints.get('epoch', [])[:5]:
                options.append((f"Epoch: {ckpt.name}", str(ckpt)))
        
        if options:
            ui_components['model_dropdown'].options = options
            # Select first option as default
            ui_components['model_dropdown'].value = options[0][1] if options else None
        else:
            print("⚠️ Tidak ada checkpoint tersedia untuk evaluasi")

def evaluate_model(model_path=None, test_dataset=None, conf_threshold=0.25, 
                  evaluation_handler=None, config=None, logger=None):
    """
    Evaluasi model dengan checkpoint yang dipilih.
    
    Args:
        model_path: Path ke checkpoint model (jika None, gunakan checkpoint terbaik)
        test_dataset: Path ke dataset testing (jika None, gunakan dataset default)
        conf_threshold: Threshold confidence untuk deteksi
        evaluation_handler: Instance dari EvaluationHandler
        config: Dictionary konfigurasi
        logger: Logger untuk mencatat aktivitas
    
    Returns:
        Dict hasil evaluasi
    """
    try:
        with memory_manager():
            # Jika tidak ada evaluation handler, coba buat baru
            if evaluation_handler is None:
                from smartcash.handlers.evaluation_handler import EvaluationHandler
                evaluation_handler = EvaluationHandler(
                    config=config,
                    logger=logger
                )
            
            # Jalankan evaluasi
            logger.info(f"🔍 Mengevaluasi model dari {model_path}")
            results = evaluation_handler.evaluate(
                eval_type='regular',
                model_path=model_path,
                test_dataset=test_dataset,
                conf_threshold=conf_threshold
            )
            
            return results
    except Exception as e:
        logger.error(f"❌ Error saat evaluasi model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def evaluate_multiple_runs(model_path=None, test_dataset=None, num_runs=3, conf_threshold=0.25,
                          evaluation_handler=None, config=None, logger=None):
    """
    Evaluasi model dengan multiple runs untuk konsistensi.
    
    Args:
        model_path: Path ke checkpoint model
        test_dataset: Path ke dataset testing
        num_runs: Jumlah run evaluasi
        conf_threshold: Threshold confidence
        evaluation_handler: Instance dari EvaluationHandler
        config: Dictionary konfigurasi
        logger: Logger untuk mencatat aktivitas
        
    Returns:
        Dict hasil evaluasi rata-rata
    """
    try:
        logger.info(f"🔄 Menjalankan {num_runs} kali evaluasi untuk konsistensi")
        
        all_results = []
        for i in range(num_runs):
            logger.info(f"🔍 Run {i+1}/{num_runs}")
            result = evaluate_model(
                model_path=model_path, 
                test_dataset=test_dataset, 
                conf_threshold=conf_threshold,
                evaluation_handler=evaluation_handler,
                config=config,
                logger=logger
            )
            if result:
                all_results.append(result)
        
        if not all_results:
            logger.error("❌ Semua run evaluasi gagal")
            return None
            
        # Agregasi hasil
        avg_metrics = {}
        
        # Ambil metrik dasar dari hasil pertama
        base_metrics = all_results[0].get('metrics', {})
        for key, value in base_metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                # Hitung rata-rata untuk nilai numerik
                values = [r.get('metrics', {}).get(key, 0) for r in all_results]
                avg_metrics[key] = sum(values) / len(values)
                # Tambahkan standar deviasi
                if len(values) > 1:
                    std = np.std(values)
                    avg_metrics[f"{key}_std"] = std
        
        # Buat hasil agregasi
        aggregated_result = {
            'metrics': avg_metrics,
            'num_runs': num_runs,
            'model_path': model_path,
            'test_dataset': test_dataset,
            'conf_threshold': conf_threshold,
            'original_results': all_results
        }
        
        # Tambahkan confusion matrix dari hasil terakhir jika ada
        if 'confusion_matrix' in all_results[-1]:
            aggregated_result['confusion_matrix'] = all_results[-1]['confusion_matrix']
            
        # Tambahkan sample detections jika ada
        if 'sample_detections' in all_results[-1]:
            aggregated_result['sample_detections'] = all_results[-1]['sample_detections']
        
        logger.success(f"✅ Berhasil mengagregasi hasil dari {len(all_results)} run")
        return aggregated_result
        
    except Exception as e:
        logger.error(f"❌ Error saat menjalankan multiple evaluasi: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def visualize_evaluation_results(results, visualizer=None, show_confusion_matrix=True, show_class_metrics=True, logger=None):
    """
    Visualisasikan hasil evaluasi model menggunakan visualizer jika tersedia atau fallback.
    
    Args:
        results: Dict hasil evaluasi
        visualizer: Instance visualizer (optional)
        show_confusion_matrix: Boolean untuk menampilkan confusion matrix
        show_class_metrics: Boolean untuk menampilkan metrik per kelas
        logger: Logger untuk mencatat aktivitas
    """
    if not results:
        if logger:
            logger.warning("⚠️ Tidak ada hasil evaluasi untuk divisualisasikan")
        else:
            print("⚠️ Tidak ada hasil evaluasi untuk divisualisasikan")
        return
    
    try:
        # Coba gunakan visualizer jika tersedia
        if visualizer and hasattr(visualizer, 'visualize_evaluation_results'):
            visualizer.visualize_evaluation_results(
                results=results,
                title="Hasil Evaluasi Model",
                show_confusion_matrix=show_confusion_matrix,
                show_class_metrics=show_class_metrics
            )
            return
        elif visualizer and hasattr(visualizer, 'display_evaluation_results'):
            visualizer.display_evaluation_results(
                results=results,
                title="Hasil Evaluasi Model",
                show_confusion_matrix=show_confusion_matrix,
                show_class_metrics=show_class_metrics
            )
            return
            
        # Fallback visualization if no visualizer available
        # Dapatkan metrics
        metrics = results.get('metrics', {})
        
        # Tampilkan ringkasan metrics
        display(HTML("<h3>📊 Ringkasan Hasil Evaluasi</h3>"))
        
        # Format metrics untuk tampilan
        metrics_html = "<table style='width:60%; border-collapse:collapse; margin-bottom:20px;'>"
        metrics_html += "<tr style='background-color:#f0f0f0;'><th style='padding:8px; text-align:left; border:1px solid #ddd;'>Metrik</th><th style='padding:8px; text-align:right; border:1px solid #ddd;'>Nilai</th></tr>"
        
        # Key metrics to display
        key_metrics = {
            'accuracy': 'Akurasi',
            'precision': 'Precision',
            'recall': 'Recall',
            'f1': 'F1-Score',
            'mAP': 'mAP',
            'inference_time': 'Inference Time (ms)'
        }
        
        for key, label in key_metrics.items():
            if key in metrics:
                value = metrics[key]
                # Format based on metric type
                if key == 'inference_time':
                    formatted_value = f"{value:.2f} ms"
                elif key == 'mAP' or key in ['accuracy', 'precision', 'recall', 'f1']:
                    # Convert to percentage
                    formatted_value = f"{value * 100:.2f}%"
                else:
                    formatted_value = f"{value:.4f}"
                    
                # Add row with styling based on value quality
                if key in ['accuracy', 'precision', 'recall', 'f1', 'mAP']:
                    if value * 100 >= 90:
                        bg_color = "#d4edda"  # Good (green)
                    elif value * 100 >= 70:
                        bg_color = "#fff3cd"  # Average (yellow)
                    else:
                        bg_color = "#f8d7da"  # Poor (red)
                else:
                    bg_color = "#ffffff"  # Default
                    
                metrics_html += f"<tr style='background-color:{bg_color};'><td style='padding:8px; border:1px solid #ddd;'>{label}</td><td style='padding:8px; text-align:right; border:1px solid #ddd;'>{formatted_value}</td></tr>"
        
        metrics_html += "</table>"
        display(HTML(metrics_html))
        
        # Tampilkan std deviation jika ada
        std_metrics = {k: v for k, v in metrics.items() if k.endswith('_std')}
        if std_metrics:
            display(HTML("<h4>📉 Standar Deviasi (Multiple Runs)</h4>"))
            
            std_html = "<table style='width:60%; border-collapse:collapse; margin-bottom:20px;'>"
            std_html += "<tr style='background-color:#f0f0f0;'><th style='padding:8px; text-align:left; border:1px solid #ddd;'>Metrik</th><th style='padding:8px; text-align:right; border:1px solid #ddd;'>Std Dev</th></tr>"
            
            for key, value in std_metrics.items():
                base_metric = key.replace('_std', '')
                label = key_metrics.get(base_metric, base_metric)
                
                if base_metric == 'inference_time':
                    formatted_value = f"{value:.2f} ms"
                elif base_metric in ['accuracy', 'precision', 'recall', 'f1', 'mAP']:
                    formatted_value = f"{value * 100:.2f}%"
                else:
                    formatted_value = f"{value:.4f}"
                
                std_html += f"<tr><td style='padding:8px; border:1px solid #ddd;'>{label}</td><td style='padding:8px; text-align:right; border:1px solid #ddd;'>{formatted_value}</td></tr>"
            
            std_html += "</table>"
            display(HTML(std_html))
        
        # Tampilkan confusion matrix jika diminta dan tersedia
        if show_confusion_matrix and 'confusion_matrix' in results:
            display(HTML("<h3>🔄 Confusion Matrix</h3>"))
            
            cm = results['confusion_matrix']
            plt.figure(figsize=(10, 8))
            
            # Get class names
            class_names = results.get('class_names', [str(i) for i in range(cm.shape[0])])
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.show()
        
        # Tampilkan metrik per kelas jika diminta
        if show_class_metrics:
            # Extract class metrics
            class_metrics = {}
            for key, value in metrics.items():
                if key.startswith(('precision_cls_', 'recall_cls_', 'f1_cls_')):
                    parts = key.split('_')
                    metric_type = parts[0]
                    class_id = int(parts[-1])
                    
                    if class_id not in class_metrics:
                        class_metrics[class_id] = {}
                    
                    class_metrics[class_id][metric_type] = value
            
            if class_metrics:
                display(HTML("<h3>📊 Metrik per Kelas</h3>"))
                
                # Get class names
                class_names = results.get('class_names', {})
                
                # Prepare data for table
                rows = []
                for class_id, metrics in class_metrics.items():
                    class_name = class_names.get(class_id, f"Class {class_id}")
                    rows.append({
                        'Class': class_name,
                        'Precision': metrics.get('precision', 0) * 100,
                        'Recall': metrics.get('recall', 0) * 100,
                        'F1-Score': metrics.get('f1', 0) * 100
                    })
                
                # Create and display DataFrame
                class_df = pd.DataFrame(rows)
                display(class_df.style.format({
                    'Precision': '{:.2f}%',
                    'Recall': '{:.2f}%',
                    'F1-Score': '{:.2f}%'
                }))
        
        # Tampilkan informasi tambahan
        display(HTML("<h3>📌 Informasi Model</h3>"))
        
        model_info_html = "<table style='width:60%; border-collapse:collapse;'>"
        model_info_html += "<tr style='background-color:#f0f0f0;'><th style='padding:8px; text-align:left; border:1px solid #ddd;'>Parameter</th><th style='padding:8px; text-align:left; border:1px solid #ddd;'>Nilai</th></tr>"
        
        # Add model info rows
        if 'config' in results:
            config = results['config']
            model_info = [
                ('Backbone', config.get('model', {}).get('backbone', 'unknown')),
                ('Mode Deteksi', 'Multi-layer' if len(config.get('layers', [])) > 1 else 'Single-layer'),
                ('Ukuran Input', f"{config.get('model', {}).get('img_size', [0, 0])}")
            ]
        else:
            model_info = [
                ('Backbone', 'unknown'),
                ('Mode Deteksi', 'unknown'),
                ('Ukuran Input', 'unknown')
            ]
            
        # Add dataset info
        test_dataset = results.get('test_dataset', 'unknown')
        model_info.append(('Dataset', test_dataset))
        
        # Add model path
        model_path = results.get('model_path', 'unknown')
        if isinstance(model_path, str):
            model_filename = os.path.basename(model_path)
            model_info.append(('Checkpoint', model_filename))
        
        # Add confidence threshold
        conf_threshold = results.get('conf_threshold', 0.25)
        model_info.append(('Confidence Threshold', f"{conf_threshold:.2f}"))
        
        # Add number of runs if multiple
        if 'num_runs' in results:
            model_info.append(('Jumlah Run', results['num_runs']))
        
        # Format model info table
        for label, value in model_info:
            model_info_html += f"<tr><td style='padding:8px; border:1px solid #ddd;'>{label}</td><td style='padding:8px; border:1px solid #ddd;'>{value}</td></tr>"
        
        model_info_html += "</table>"
        display(HTML(model_info_html))
        
        # Tampilkan sample detections jika tersedia
        if 'sample_detections' in results:
            display(HTML("<h3>🖼️ Contoh Hasil Deteksi</h3>"))
            
            samples = results['sample_detections']
            for i, sample in enumerate(samples[:3]):  # Limit to 3 samples
                plt.figure(figsize=(10, 8))
                plt.imshow(sample)
                plt.axis('off')
                plt.title(f"Sample Detection {i+1}")
                plt.tight_layout()
                plt.show()
    
    except Exception as e:
        if logger:
            logger.error(f"❌ Error pada visualisasi hasil evaluasi: {str(e)}")
        else:
            print(f"❌ Error pada visualisasi hasil evaluasi: {str(e)}")
        import traceback
        traceback.print_exc()

def on_run_evaluation_button_clicked(ui_components, evaluation_handler, checkpoint_handler, 
                                   visualizer, config, logger):
    """
    Handler untuk tombol evaluasi model.
    
    Args:
        ui_components: Dictionary komponen UI dari create_evaluation_ui()
        evaluation_handler: Instance dari EvaluationHandler
        checkpoint_handler: Instance dari CheckpointHandler
        visualizer: Instance dari visualizer (optional)
        config: Dictionary konfigurasi
        logger: Logger untuk mencatat aktivitas
    """
    with ui_components['evaluation_output']:
        clear_output()
        
        model_path = ui_components['model_dropdown'].value
        test_dataset = ui_components['dataset_dropdown'].value
        
        if not model_path:
            print("❌ Pilih model checkpoint terlebih dahulu")
            return
        
        # Get evaluation settings
        conf_threshold = ui_components['conf_threshold_slider'].value
        num_runs = ui_components['num_runs_slider'].value
        show_confusion_matrix = ui_components['confusion_matrix_checkbox'].value
        show_class_metrics = ui_components['class_metrics_checkbox'].value
        
        # Set loading state
        ui_components['run_evaluation_button'].disabled = True
        ui_components['run_evaluation_button'].description = "Mengevaluasi..."
        
        try:
            print(f"🔍 Mengevaluasi model: {os.path.basename(model_path)}")
            print(f"📊 Dataset: {test_dataset or 'Default'}")
            print(f"⚙️ Confidence threshold: {conf_threshold}, Jumlah run: {num_runs}")
            
            # Jalankan evaluasi
            if num_runs > 1:
                results = evaluate_multiple_runs(
                    model_path=model_path, 
                    test_dataset=test_dataset,
                    num_runs=num_runs, 
                    conf_threshold=conf_threshold,
                    evaluation_handler=evaluation_handler,
                    config=config,
                    logger=logger
                )
            else:
                results = evaluate_model(
                    model_path=model_path, 
                    test_dataset=test_dataset,
                    conf_threshold=conf_threshold,
                    evaluation_handler=evaluation_handler,
                    config=config,
                    logger=logger
                )
            
            if results:
                print("✅ Evaluasi berhasil")
                visualize_evaluation_results(
                    results=results, 
                    visualizer=visualizer,
                    show_confusion_matrix=show_confusion_matrix,
                    show_class_metrics=show_class_metrics,
                    logger=logger
                )
            else:
                print("❌ Evaluasi gagal atau tidak menghasilkan data")
        except Exception as e:
            print(f"❌ Error saat evaluasi: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            # Reset button state
            ui_components['run_evaluation_button'].disabled = False
            ui_components['run_evaluation_button'].description = "Evaluasi Model"

def setup_evaluation_handlers(ui_components, evaluation_handler, checkpoint_handler,
                            visualizer, config, logger):
    """
    Setup semua event handlers untuk UI evaluasi model.
    
    Args:
        ui_components: Dictionary komponen UI
        evaluation_handler: Instance dari EvaluationHandler 
        checkpoint_handler: Instance dari CheckpointHandler
        visualizer: Instance dari visualizer (optional)
        config: Dictionary konfigurasi
        logger: Logger untuk mencatat aktivitas
        
    Returns:
        Dictionary updated UI components dengan handlers yang sudah di-attach
    """
    # Initialize UI with model list
    init_ui(ui_components, checkpoint_handler)
    
    # Setup handler untuk tombol evaluasi
    ui_components['run_evaluation_button'].on_click(
        lambda b: on_run_evaluation_button_clicked(
            ui_components, evaluation_handler, checkpoint_handler, 
            visualizer, config, logger
        )
    )
    
    return ui_components