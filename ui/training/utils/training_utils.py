"""smartcash/ui/training/utils/training_utils.py

Fungsi utilitas untuk training module.
"""

from smartcash.model.metrics_tracker import MetricsTracker
import logging
import numpy as np

def get_metrics_from_model():
    """Mengambil metrik dari model"""
    tracker = MetricsTracker.get_instance()
    return {
        'mAP': tracker.get_metric('mAP'),
        'loss': tracker.get_metric('loss'),
        'accuracy': tracker.get_metric('accuracy'),
        'precision': tracker.get_metric('precision'),
        'f1': tracker.get_metric('f1'),
        'inference_time': tracker.get_metric('inference_time')
    }

def update_charts_data(ui_components, metrics_history):
    """Memperbarui data grafik"""
    try:
        # Update loss curve
        loss_plot = ui_components['charts_accordion'].children[0].children[0]
        loss_plot.data[0].x = list(range(len(metrics_history['loss'])))
        loss_plot.data[0].y = metrics_history['loss']
        
        # Update mAP curve
        map_plot = ui_components['charts_accordion'].children[1].children[0]
        map_plot.data[0].x = list(range(len(metrics_history['mAP'])))
        map_plot.data[0].y = metrics_history['mAP']
        
    except Exception as e:
        print(f"Error updating charts: {str(e)}")

def update_confusion_matrix(ui_components, cm_data, classes):
    """Update confusion matrix heatmap"""
    try:
        heatmap = ui_components['confusion_matrix_accordion'].children[0].children[0]
        
        # Normalisasi untuk persentase
        cm_normalized = cm_data.astype('float') / cm_data.sum(axis=1)[:, np.newaxis]
        
        # Update data heatmap
        with heatmap.batch_update():
            heatmap.data[0].z = cm_normalized
            heatmap.data[0].x = classes
            heatmap.data[0].y = classes
            
        # Update layout jika perlu
        heatmap.update_layout(title='Confusion Matrix (Normalized)')
        
    except Exception as e:
        logging.error(f"❌ Error update confusion matrix: {str(e)}")

def update_confusion_matrix(component, cm, class_labels):
    """Update confusion matrix component"""
    if component:
        component.update_matrix(cm, class_labels)
