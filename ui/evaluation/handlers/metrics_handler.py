"""
File: smartcash/ui/evaluation/handlers/metrics_handler.py
Deskripsi: Handler untuk perhitungan dan display metrics evaluasi dengan tabel dan visualisasi
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple
from sklearn.metrics import confusion_matrix, classification_report
from smartcash.ui.utils.logger_bridge import log_to_service

def setup_metrics_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None):
    """Setup handlers untuk metrics calculation dan display"""
    
    # Metrics handlers akan di-trigger dari evaluation_handler
    # Setup display update handlers
    setup_metrics_display_handlers(ui_components)
    
    return ui_components

def calculate_and_save_metrics(predictions: List[Dict[str, Any]], labels_info: Dict[str, Any], 
                              ui_components: Dict[str, Any], config: Dict[str, Any], logger) -> Dict[str, Any]:
    """Calculate comprehensive metrics dan save results dengan one-liner calculations"""
    
    try:
        ui_components.get('update_progress', lambda *args: None)('step', 0, "📊 Calculating metrics...")
        
        # Extract config dari UI
        class_names = config.get('evaluation', {}).get('class_names', [])
        save_metrics = ui_components.get('save_metrics_checkbox', {}).get('value', True)
        generate_cm = ui_components.get('confusion_matrix_checkbox', {}).get('value', True)
        
        # Calculate detection metrics
        detection_metrics = calculate_detection_metrics(predictions, labels_info, class_names, logger)
        ui_components.get('update_progress', lambda *args: None)('step', 30, "📈 Detection metrics calculated")
        
        # Calculate classification metrics jika labels tersedia
        classification_metrics = {}
        if labels_info.get('available', False):
            classification_metrics = calculate_classification_metrics(predictions, labels_info, class_names, logger)
            ui_components.get('update_progress', lambda *args: None)('step', 60, "🎯 Classification metrics calculated")
        
        # Generate confusion matrix
        confusion_data = {}
        if generate_cm and labels_info.get('available', False):
            confusion_data = generate_confusion_matrix(predictions, labels_info, class_names, logger)
            ui_components.get('update_progress', lambda *args: None)('step', 80, "🔥 Confusion matrix generated")
        
        # Combine all metrics
        all_metrics = {
            'detection': detection_metrics,
            'classification': classification_metrics,
            'confusion_matrix': confusion_data,
            'summary': create_metrics_summary(detection_metrics, classification_metrics),
            'metadata': {
                'total_images': len(predictions),
                'total_predictions': sum(len(p['predictions']) for p in predictions),
                'has_ground_truth': labels_info.get('available', False),
                'class_names': class_names
            }
        }
        
        # Save metrics jika diminta
        if save_metrics:
            save_results = save_metrics_to_files(all_metrics, ui_components, config, logger)
            all_metrics['saved_files'] = save_results
        
        ui_components.get('update_progress', lambda *args: None)('step', 100, "✅ Metrics calculation completed")
        
        log_to_service(logger, f"✅ Metrics calculated: {len(all_metrics['summary'])} summary metrics", "success")
        
        return {'success': True, 'metrics': all_metrics}
        
    except Exception as e:
        log_to_service(logger, f"❌ Error calculating metrics: {str(e)}", "error")
        return {'success': False, 'error': str(e)}

def calculate_detection_metrics(predictions: List[Dict[str, Any]], labels_info: Dict[str, Any], 
                               class_names: List[str], logger) -> Dict[str, Any]:
    """Calculate detection-specific metrics dengan one-liner aggregations"""
    
    try:
        # Aggregate prediction statistics dengan one-liner
        total_predictions = sum(len(pred['predictions']) for pred in predictions)
        predictions_per_image = [len(pred['predictions']) for pred in predictions]
        
        # Confidence statistics dengan one-liner numpy operations
        all_confidences = [conf.item() if hasattr(conf, 'item') else float(conf)
                          for pred in predictions 
                          for detection in pred['predictions']
                          for conf in [detection[4]] if len(detection) > 4]
        
        # Class distribution dengan one-liner
        class_counts = {}
        for pred in predictions:
            for detection in pred['predictions']:
                if len(detection) > 5:
                    class_id = int(detection[5])
                    class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Calculate basic statistics
        detection_metrics = {
            'total_predictions': total_predictions,
            'total_images': len(predictions),
            'predictions_per_image': {
                'mean': np.mean(predictions_per_image) if predictions_per_image else 0,
                'std': np.std(predictions_per_image) if predictions_per_image else 0,
                'min': min(predictions_per_image) if predictions_per_image else 0,
                'max': max(predictions_per_image) if predictions_per_image else 0
            },
            'confidence_stats': {
                'mean': np.mean(all_confidences) if all_confidences else 0,
                'std': np.std(all_confidences) if all_confidences else 0,
                'min': min(all_confidences) if all_confidences else 0,
                'max': max(all_confidences) if all_confidences else 0,
                'percentiles': {
                    '25': np.percentile(all_confidences, 25) if all_confidences else 0,
                    '50': np.percentile(all_confidences, 50) if all_confidences else 0,
                    '75': np.percentile(all_confidences, 75) if all_confidences else 0,
                    '95': np.percentile(all_confidences, 95) if all_confidences else 0
                }
            },
            'class_distribution': class_counts,
            'detection_rate': total_predictions / len(predictions) if predictions else 0
        }
        
        return detection_metrics
        
    except Exception as e:
        log_to_service(logger, f"⚠️ Error calculating detection metrics: {str(e)}", "warning")
        return {}

def calculate_classification_metrics(predictions: List[Dict[str, Any]], labels_info: Dict[str, Any], 
                                   class_names: List[str], logger) -> Dict[str, Any]:
    """Calculate classification metrics dengan ground truth comparison"""
    
    try:
        labels = labels_info.get('labels', {})
        
        # Match predictions dengan ground truth
        matched_data = match_predictions_with_labels(predictions, labels, class_names)
        
        if not matched_data['y_true'] or not matched_data['y_pred']:
            log_to_service(logger, "⚠️ Tidak ada matches untuk classification metrics", "warning")
            return {}
        
        # Calculate precision, recall, f1-score per class
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score
        
        precision, recall, f1, support = precision_recall_fscore_support(
            matched_data['y_true'], matched_data['y_pred'], 
            labels=list(range(len(class_names))), average=None, zero_division=0
        )
        
        # Per-class metrics dengan one-liner dict comprehension
        per_class_metrics = {
            class_names[i]: {
                'precision': float(precision[i]) if i < len(precision) else 0.0,
                'recall': float(recall[i]) if i < len(recall) else 0.0,
                'f1_score': float(f1[i]) if i < len(f1) else 0.0,
                'support': int(support[i]) if i < len(support) else 0
            } for i in range(len(class_names))
        }
        
        # Overall metrics dengan weighted averages
        precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
            matched_data['y_true'], matched_data['y_pred'], average='weighted', zero_division=0
        )
        
        classification_metrics = {
            'accuracy': float(accuracy_score(matched_data['y_true'], matched_data['y_pred'])),
            'precision_weighted': float(precision_avg),
            'recall_weighted': float(recall_avg),
            'f1_score_weighted': float(f1_avg),
            'per_class': per_class_metrics,
            'total_matches': len(matched_data['y_true'])
        }
        
        return classification_metrics
        
    except Exception as e:
        log_to_service(logger, f"⚠️ Error calculating classification metrics: {str(e)}", "warning")
        return {}

def match_predictions_with_labels(predictions: List[Dict[str, Any]], labels: Dict[str, Any], 
                                 class_names: List[str]) -> Dict[str, List]:
    """Match predictions dengan ground truth labels berdasarkan IoU threshold"""
    
    y_true, y_pred = [], []
    iou_threshold = 0.5  # Standard IoU threshold untuk matching
    
    try:
        for pred_data in predictions:
            img_path = pred_data['image_path']
            pred_boxes = pred_data['predictions']
            
            if img_path not in labels:
                continue
            
            gt_boxes = labels[img_path]  # Ground truth boxes
            
            # Convert formats dan match
            matches = match_boxes_iou(pred_boxes, gt_boxes, iou_threshold)
            
            # Extract matched classes
            for match in matches:
                y_true.append(match['gt_class'])
                y_pred.append(match['pred_class'])
        
        return {'y_true': y_true, 'y_pred': y_pred}
        
    except Exception:
        return {'y_true': [], 'y_pred': []}

def match_boxes_iou(pred_boxes, gt_boxes, iou_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """Match predicted boxes dengan ground truth berdasarkan IoU"""
    
    matches = []
    
    try:
        import torch
        
        for pred_box in pred_boxes:
            if len(pred_box) < 6:
                continue
            
            # Convert pred_box format: [x1, y1, x2, y2, conf, class]
            pred_bbox = pred_box[:4]
            pred_class = int(pred_box[5])
            
            best_iou = 0
            best_gt_class = None
            
            for gt_box in gt_boxes:
                if len(gt_box) < 5:
                    continue
                
                # Convert YOLO format ke bbox format
                gt_class, x_center, y_center, width, height = gt_box
                gt_bbox = yolo_to_bbox(x_center, y_center, width, height)
                
                # Calculate IoU
                iou = calculate_iou(pred_bbox, gt_bbox)
                
                if iou > best_iou and iou > iou_threshold:
                    best_iou = iou
                    best_gt_class = int(gt_class)
            
            if best_gt_class is not None:
                matches.append({
                    'pred_class': pred_class,
                    'gt_class': best_gt_class,
                    'iou': best_iou
                })
        
        return matches
        
    except Exception:
        return []

def yolo_to_bbox(x_center: float, y_center: float, width: float, height: float) -> List[float]:
    """Convert YOLO format ke bbox format [x1, y1, x2, y2]"""
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    return [x1, y1, x2, y2]

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate Intersection over Union (IoU) antara dua bounding boxes"""
    
    try:
        # Extract coordinates
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
        
    except Exception:
        return 0.0

def generate_confusion_matrix(predictions: List[Dict[str, Any]], labels_info: Dict[str, Any], 
                            class_names: List[str], logger) -> Dict[str, Any]:
    """Generate confusion matrix dan visualization"""
    
    try:
        # Match predictions dengan labels
        matched_data = match_predictions_with_labels(predictions, labels_info.get('labels', {}), class_names)
        
        if not matched_data['y_true'] or not matched_data['y_pred']:
            return {'available': False, 'message': 'Tidak ada data untuk confusion matrix'}
        
        # Generate confusion matrix
        cm = confusion_matrix(matched_data['y_true'], matched_data['y_pred'], 
                            labels=list(range(len(class_names))))
        
        # Convert ke format yang bisa di-serialize
        cm_data = {
            'matrix': cm.tolist(),
            'class_names': class_names,
            'total_predictions': len(matched_data['y_pred']),
            'accuracy': np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0
        }
        
        # Generate normalized confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)  # Replace NaN dengan 0
        cm_data['matrix_normalized'] = cm_normalized.tolist()
        
        log_to_service(logger, f"🔥 Confusion matrix generated: {cm.shape[0]}x{cm.shape[1]}", "success")
        
        return {'available': True, 'data': cm_data}
        
    except Exception as e:
        log_to_service(logger, f"⚠️ Error generating confusion matrix: {str(e)}", "warning")
        return {'available': False, 'error': str(e)}

def create_metrics_summary(detection_metrics: Dict[str, Any], classification_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Create summary metrics untuk display dengan one-liner aggregations"""
    
    summary = {}
    
    # Detection summary
    if detection_metrics:
        summary.update({
            'Total Images': detection_metrics.get('total_images', 0),
            'Total Predictions': detection_metrics.get('total_predictions', 0),
            'Avg Predictions/Image': round(detection_metrics.get('detection_rate', 0), 2),
            'Avg Confidence': round(detection_metrics.get('confidence_stats', {}).get('mean', 0), 3),
            'Min Confidence': round(detection_metrics.get('confidence_stats', {}).get('min', 0), 3),
            'Max Confidence': round(detection_metrics.get('confidence_stats', {}).get('max', 0), 3)
        })
    
    # Classification summary
    if classification_metrics:
        summary.update({
            'Overall Accuracy': round(classification_metrics.get('accuracy', 0), 3),
            'Weighted Precision': round(classification_metrics.get('precision_weighted', 0), 3),
            'Weighted Recall': round(classification_metrics.get('recall_weighted', 0), 3),
            'Weighted F1-Score': round(classification_metrics.get('f1_score_weighted', 0), 3),
            'Total Matches': classification_metrics.get('total_matches', 0)
        })
    
    return summary

def save_metrics_to_files(metrics: Dict[str, Any], ui_components: Dict[str, Any], 
                         config: Dict[str, Any], logger) -> Dict[str, str]:
    """Save metrics ke berbagai format files dengan one-liner file operations"""
    
    try:
        # Create output directory
        output_dir = Path(config.get('output', {}).get('results_folder', 'output/evaluation'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        export_formats = config.get('output', {}).get('export_format', ['csv', 'json'])
        
        # Save JSON format (comprehensive)
        if 'json' in export_formats:
            json_file = output_dir / 'evaluation_metrics.json'
            with open(json_file, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            saved_files['json'] = str(json_file)
        
        # Save CSV format (summary only)
        if 'csv' in export_formats:
            csv_file = output_dir / 'evaluation_summary.csv'
            summary_df = pd.DataFrame([metrics['summary']])
            summary_df.to_csv(csv_file, index=False)
            saved_files['csv'] = str(csv_file)
        
        # Save per-class metrics sebagai CSV
        if metrics.get('classification', {}).get('per_class'):
            per_class_file = output_dir / 'per_class_metrics.csv'
            per_class_df = pd.DataFrame(metrics['classification']['per_class']).T
            per_class_df.to_csv(per_class_file)
            saved_files['per_class'] = str(per_class_file)
        
        # Save confusion matrix sebagai CSV
        if metrics.get('confusion_matrix', {}).get('available'):
            cm_file = output_dir / 'confusion_matrix.csv'
            cm_data = metrics['confusion_matrix']['data']
            cm_df = pd.DataFrame(cm_data['matrix'], 
                               index=cm_data['class_names'], 
                               columns=cm_data['class_names'])
            cm_df.to_csv(cm_file)
            saved_files['confusion_matrix'] = str(cm_file)
        
        log_to_service(logger, f"💾 Metrics saved ke {len(saved_files)} files: {list(saved_files.keys())}", "success")
        
        return saved_files
        
    except Exception as e:
        log_to_service(logger, f"⚠️ Error saving metrics: {str(e)}", "warning")
        return {}

def update_results_ui(ui_components: Dict[str, Any], metrics: Dict[str, Any], 
                     predictions: List[Dict[str, Any]], logger) -> None:
    """Update UI dengan hasil metrics dan predictions"""
    
    try:
        # Update metrics table
        update_metrics_table(ui_components, metrics, logger)
        
        # Update confusion matrix display
        update_confusion_matrix_display(ui_components, metrics, logger)
        
        # Update sample predictions display
        update_predictions_display(ui_components, predictions, metrics, logger)
        
        # Show results tabs
        if 'results_tabs' in ui_components:
            ui_components['results_tabs'].selected_index = 0  # Show metrics tab
        
        log_to_service(logger, "✅ UI results updated successfully", "success")
        
    except Exception as e:
        log_to_service(logger, f"⚠️ Error updating results UI: {str(e)}", "warning")

def update_metrics_table(ui_components: Dict[str, Any], metrics: Dict[str, Any], logger) -> None:
    """Update metrics table dengan formatted HTML"""
    
    try:
        summary = metrics.get('summary', {})
        detection = metrics.get('detection', {})
        classification = metrics.get('classification', {})
        
        # Create HTML table dengan one-liner table generation
        html_content = f"""
        <div style="padding: 15px; font-family: 'Segoe UI', sans-serif;">
            <h4 style="color: #2c3e50; margin-bottom: 20px;">📊 Hasil Evaluasi Model</h4>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;">
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #28a745;">
                    <h5 style="color: #28a745; margin-top: 0;">🎯 Detection Metrics</h5>
                    {create_metrics_table_section(summary, ['Total Images', 'Total Predictions', 'Avg Predictions/Image', 'Avg Confidence'])}
                </div>
                
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #007bff;">
                    <h5 style="color: #007bff; margin-top: 0;">📈 Classification Metrics</h5>
                    {create_metrics_table_section(summary, ['Overall Accuracy', 'Weighted Precision', 'Weighted Recall', 'Weighted F1-Score']) if classification else '<p style="color: #666;">Tidak tersedia (tidak ada ground truth)</p>'}
                </div>
            </div>
            
            <div style="background: #fff; padding: 15px; border: 1px solid #dee2e6; border-radius: 8px;">
                <h5 style="color: #6c757d; margin-top: 0;">📋 Summary Statistics</h5>
                {create_full_metrics_table(summary)}
            </div>
            
            {create_class_distribution_section(detection.get('class_distribution', {})) if detection.get('class_distribution') else ''}
        </div>
        """
        
        if 'metrics_table' in ui_components:
            ui_components['metrics_table'].value = html_content
        
    except Exception as e:
        log_to_service(logger, f"⚠️ Error updating metrics table: {str(e)}", "warning")

def create_metrics_table_section(summary: Dict[str, Any], keys: List[str]) -> str:
    """Create HTML table section untuk specific metrics dengan one-liner"""
    
    rows = [f"<tr><td style='font-weight: 500; color: #495057;'>{key}:</td><td style='text-align: right; color: #28a745; font-weight: 600;'>{summary.get(key, 'N/A')}</td></tr>" 
            for key in keys if key in summary]
    
    return f"""
    <table style="width: 100%; border-collapse: collapse;">
        {''.join(rows)}
    </table>
    """ if rows else "<p style='color: #666;'>No data available</p>"

def create_full_metrics_table(summary: Dict[str, Any]) -> str:
    """Create comprehensive metrics table dengan all available metrics"""
    
    if not summary:
        return "<p style='color: #666;'>No summary data available</p>"
    
    rows = [f"""
    <tr>
        <td style='padding: 8px; border-bottom: 1px solid #dee2e6; font-weight: 500;'>{key}</td>
        <td style='padding: 8px; border-bottom: 1px solid #dee2e6; text-align: right; font-family: monospace;'>{value}</td>
    </tr>
    """ for key, value in summary.items()]
    
    return f"""
    <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
        <thead>
            <tr style="background-color: #f8f9fa;">
                <th style="padding: 10px; text-align: left; border-bottom: 2px solid #dee2e6;">Metric</th>
                <th style="padding: 10px; text-align: right; border-bottom: 2px solid #dee2e6;">Value</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    """

def create_class_distribution_section(class_distribution: Dict[str, int]) -> str:
    """Create class distribution visualization section"""
    
    if not class_distribution:
        return ""
    
    total = sum(class_distribution.values())
    
    # Sort by count untuk better visualization
    sorted_classes = sorted(class_distribution.items(), key=lambda x: x[1], reverse=True)
    
    bars = []
    for class_name, count in sorted_classes:
        percentage = (count / total) * 100 if total > 0 else 0
        bar_width = min(percentage * 2, 100)  # Scale untuk visualization
        
        bars.append(f"""
        <div style="margin: 8px 0; display: flex; align-items: center;">
            <div style="width: 100px; font-size: 12px; color: #495057;">{class_name}:</div>
            <div style="flex: 1; background: #e9ecef; height: 20px; border-radius: 10px; margin: 0 10px; position: relative;">
                <div style="background: linear-gradient(90deg, #28a745, #20c997); height: 100%; width: {bar_width}%; border-radius: 10px;"></div>
            </div>
            <div style="width: 80px; text-align: right; font-size: 12px; font-family: monospace;">{count} ({percentage:.1f}%)</div>
        </div>
        """)
    
    return f"""
    <div style="background: #fff; padding: 15px; border: 1px solid #dee2e6; border-radius: 8px; margin-top: 20px;">
        <h5 style="color: #6c757d; margin-top: 0;">📊 Class Distribution</h5>
        {''.join(bars)}
    </div>
    """

def update_confusion_matrix_display(ui_components: Dict[str, Any], metrics: Dict[str, Any], logger) -> None:
    """Update confusion matrix display dengan matplotlib visualization"""
    
    try:
        cm_data = metrics.get('confusion_matrix', {})
        
        if not cm_data.get('available'):
            # Show message jika confusion matrix tidak tersedia
            if 'confusion_matrix_output' in ui_components:
                with ui_components['confusion_matrix_output']:
                    from IPython.display import display, HTML
                    display(HTML("""
                    <div style="text-align: center; padding: 50px; color: #666;">
                        <h4>🔥 Confusion Matrix</h4>
                        <p>Confusion matrix tidak tersedia karena tidak ada ground truth labels.</p>
                        <p>Pastikan folder test memiliki subfolder 'labels' dengan file .txt yang sesuai.</p>
                    </div>
                    """))
            return
        
        # Generate matplotlib plot
        if 'confusion_matrix_output' in ui_components:
            with ui_components['confusion_matrix_output']:
                plot_confusion_matrix(cm_data['data'])
        
    except Exception as e:
        log_to_service(logger, f"⚠️ Error updating confusion matrix display: {str(e)}", "warning")

def plot_confusion_matrix(cm_data: Dict[str, Any]) -> None:
    """Plot confusion matrix menggunakan matplotlib"""
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from IPython.display import display
        
        # Setup plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot raw confusion matrix
        cm_raw = np.array(cm_data['matrix'])
        sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=cm_data['class_names'], 
                   yticklabels=cm_data['class_names'], ax=ax1)
        ax1.set_title('🔥 Confusion Matrix (Raw Counts)')
        ax1.set_xlabel('Predicted Class')
        ax1.set_ylabel('True Class')
        
        # Plot normalized confusion matrix
        cm_norm = np.array(cm_data['matrix_normalized'])
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=cm_data['class_names'],
                   yticklabels=cm_data['class_names'], ax=ax2)
        ax2.set_title('🔥 Confusion Matrix (Normalized)')
        ax2.set_xlabel('Predicted Class')
        ax2.set_ylabel('True Class')
        
        plt.tight_layout()
        display(fig)
        plt.close()
        
    except ImportError:
        # Fallback HTML display jika matplotlib tidak tersedia
        from IPython.display import display, HTML
        html_cm = create_html_confusion_matrix(cm_data)
        display(HTML(html_cm))
    except Exception:
        from IPython.display import display, HTML
        display(HTML("<div style='color: red; text-align: center;'>Error creating confusion matrix visualization</div>"))

def create_html_confusion_matrix(cm_data: Dict[str, Any]) -> str:
    """Create HTML confusion matrix sebagai fallback"""
    
    matrix = cm_data['matrix']
    class_names = cm_data['class_names']
    
    # Create HTML table
    header = "<tr><th>True\\Pred</th>" + "".join([f"<th>{name}</th>" for name in class_names]) + "</tr>"
    
    rows = []
    for i, true_class in enumerate(class_names):
        row = f"<tr><td><strong>{true_class}</strong></td>"
        for j in range(len(class_names)):
            value = matrix[i][j] if i < len(matrix) and j < len(matrix[i]) else 0
            intensity = min(value * 10, 100)  # Scale untuk color intensity
            row += f"<td style='background-color: rgba(54, 162, 235, {intensity/100}); text-align: center;'>{value}</td>"
        row += "</tr>"
        rows.append(row)
    
    return f"""
    <div style="text-align: center; padding: 20px;">
        <h4>🔥 Confusion Matrix</h4>
        <table style="margin: 0 auto; border-collapse: collapse; border: 1px solid #ddd;">
            {header}
            {''.join(rows)}
        </table>
        <p style="margin-top: 10px; color: #666; font-size: 12px;">
            Accuracy: {cm_data.get('accuracy', 0):.3f} | Total Predictions: {cm_data.get('total_predictions', 0)}
        </p>
    </div>
    """

def update_predictions_display(ui_components: Dict[str, Any], predictions: List[Dict[str, Any]], 
                              metrics: Dict[str, Any], logger) -> None:
    """Update sample predictions display dengan image previews"""
    
    try:
        if 'predictions_output' in ui_components:
            with ui_components['predictions_output']:
                from IPython.display import display, HTML
                
                # Show sample predictions summary
                sample_size = min(10, len(predictions))
                sample_predictions = predictions[:sample_size]
                
                html_content = f"""
                <div style="padding: 15px;">
                    <h4>🎯 Sample Predictions ({sample_size}/{len(predictions)})</h4>
                    {create_predictions_summary_table(sample_predictions)}
                </div>
                """
                
                display(HTML(html_content))
        
    except Exception as e:
        log_to_service(logger, f"⚠️ Error updating predictions display: {str(e)}", "warning")

def create_predictions_summary_table(predictions: List[Dict[str, Any]]) -> str:
    """Create summary table untuk sample predictions"""
    
    if not predictions:
        return "<p>No predictions to display</p>"
    
    rows = []
    for i, pred in enumerate(predictions[:10]):  # Limit ke 10 samples
        img_path = Path(pred['image_path']).name
        pred_count = len(pred['predictions'])
        
        # Get top confidence jika ada predictions
        top_conf = 0
        if pred['predictions'] and len(pred['predictions']) > 0:
            confidences = [float(p[4]) for p in pred['predictions'] if len(p) > 4]
            top_conf = max(confidences) if confidences else 0
        
        rows.append(f"""
        <tr>
            <td style="padding: 8px; border-bottom: 1px solid #dee2e6;">{i+1}</td>
            <td style="padding: 8px; border-bottom: 1px solid #dee2e6; font-family: monospace; font-size: 12px;">{img_path}</td>
            <td style="padding: 8px; border-bottom: 1px solid #dee2e6; text-align: center;">{pred_count}</td>
            <td style="padding: 8px; border-bottom: 1px solid #dee2e6; text-align: center; font-family: monospace;">{top_conf:.3f}</td>
        </tr>
        """)
    
    return f"""
    <table style="width: 100%; border-collapse: collapse; font-size: 14px; margin-top: 10px;">
        <thead>
            <tr style="background-color: #f8f9fa;">
                <th style="padding: 12px; text-align: left; border-bottom: 2px solid #dee2e6;">#</th>
                <th style="padding: 12px; text-align: left; border-bottom: 2px solid #dee2e6;">Image</th>
                <th style="padding: 12px; text-align: center; border-bottom: 2px solid #dee2e6;">Detections</th>
                <th style="padding: 12px; text-align: center; border-bottom: 2px solid #dee2e6;">Max Confidence</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    """

def setup_metrics_display_handlers(ui_components: Dict[str, Any]) -> None:
    """Setup handlers untuk metrics display interactions"""
    
    # Setup tab change handlers jika diperlukan
    if 'results_tabs' in ui_components:
        def on_tab_change(change):
            tab_index = change['new']
            # Handle tab-specific updates jika diperlukan
            pass
        
        # Observe tab changes
        ui_components['results_tabs'].observe(on_tab_change, names='selected_index')