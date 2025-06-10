"""smartcash/ui/training/handlers/training_handlers.py

Handler untuk UI Training.
"""

from IPython.lib import backgroundjobs as bg
from smartcash.model.training_service import TrainingService
from smartcash.ui.utils import logging_utils
from smartcash.ui.training.utils import training_utils

# Global reference untuk training service
training_job = None
training_service = None

def setup_handlers(ui_components, config_manager, env_manager):
    """Menyiapkan handler untuk training module"""
    # Handler untuk tombol Mulai Training
    ui_components['buttons']['primary_button'].on_click(
        lambda _: start_training_handler(ui_components, config_manager, env_manager)
    )
    
    # Handler untuk tombol Jeda
    ui_components['buttons']['secondary_buttons'][0].on_click(
        lambda _: pause_training_handler(ui_components)
    )
    
    # Handler untuk tombol Hentikan
    ui_components['buttons']['secondary_buttons'][1].on_click(
        lambda _: stop_training_handler(ui_components)
    )

def start_training_handler(ui_components, config_manager, env_manager):
    """Handler untuk memulai training"""
    global training_job, training_service
    
    try:
        # Dapatkan konfigurasi
        config = config_manager.get_config()
        
        # Update status
        logging_utils.update_status(ui_components['status_panel'], '⏳ Memulai training...', 'info')
        
        # Inisialisasi service
        training_service = TrainingService(config, env_manager)
        
        # Setup callback untuk update UI
        training_service.set_progress_callback(
            lambda progress: update_progress_ui(ui_components, progress)
        )
        training_service.set_metrics_callback(
            lambda metrics: update_metrics_ui(ui_components, metrics)
        )
        training_service.set_evaluation_callback(
            lambda cm, classes: update_confusion_matrix_ui(ui_components, cm, classes)
        )
        
        # Daftarkan callback untuk update confusion matrix
        def on_confusion_matrix(cm):
            """Callback untuk menerima data confusion matrix"""
            # Pastikan komponen sudah terinisialisasi
            if 'confusion_matrix' in ui_components:
                # Dapatkan class labels dari config
                class_labels = config.get('class_labels', 
                    ['Rp1000', 'Rp2000', 'Rp5000', 'Rp10000', 'Rp20000', 'Rp50000', 'Rp100000'])
                
                # Update komponen UI
                ui_components['confusion_matrix'].update_matrix(cm, class_labels)
        
        training_service.register_callback('confusion_matrix', on_confusion_matrix)
        
        # Jalankan training sebagai background job
        training_job = bg.BackgroundJob(
            training_service.start,
            "Training Job"
        )
        
        # Update tombol
        ui_components['buttons']['primary_button'].disabled = True
        ui_components['buttons']['secondary_buttons'][0].disabled = False
        ui_components['buttons']['secondary_buttons'][1].disabled = False
        
    except Exception as e:
        logging_utils.log_error(ui_components['log_accordion'], f"❌ Gagal memulai training: {str(e)}")

def pause_training_handler(ui_components):
    """Handler untuk menjeda/melanjutkan training"""
    global training_service
    
    if training_service:
        if training_service.is_paused():
            training_service.resume()
            logging_utils.update_status(ui_components['status_panel'], '▶️ Training dilanjutkan', 'success')
            ui_components['buttons']['secondary_buttons'][0].description = 'Jeda'
            ui_components['buttons']['secondary_buttons'][0].button_style = 'warning'
        else:
            training_service.pause()
            logging_utils.update_status(ui_components['status_panel'], '⏸️ Training dijeda', 'warning')
            ui_components['buttons']['secondary_buttons'][0].description = 'Lanjutkan'
            ui_components['buttons']['secondary_buttons'][0].button_style = 'success'

def stop_training_handler(ui_components):
    """Handler untuk menghentikan training"""
    global training_service, training_job
    
    if training_service:
        training_service.stop()
        logging_utils.update_status(ui_components['status_panel'], '⏹️ Training dihentikan', 'error')
        
        # Update tombol
        ui_components['buttons']['primary_button'].disabled = False
        ui_components['buttons']['secondary_buttons'][0].disabled = True
        ui_components['buttons']['secondary_buttons'][1].disabled = True

def update_progress_ui(ui_components, progress_data):
    """Update progress UI berdasarkan data dari service"""
    try:
        # Update progress tracker
        ui_components['progress']['trackers']['Epoch'].value = progress_data['epoch_progress']
        ui_components['progress']['trackers']['Batch'].value = progress_data['batch_progress']
        ui_components['progress']['trackers']['Overall'].value = progress_data['overall_progress']
        
        # Update labels
        ui_components['progress']['labels']['Epoch'].value = progress_data['epoch_label']
        ui_components['progress']['labels']['Batch'].value = progress_data['batch_label']
        ui_components['progress']['labels']['Overall'].value = progress_data['overall_label']
        
    except Exception as e:
        logging_utils.log_error(ui_components['log_accordion'], f"❌ Error update progress: {str(e)}")

def update_metrics_ui(ui_components, metrics):
    """Update metrics UI berdasarkan data dari service"""
    try:
        # Update metrik cards
        for metric_name, value in metrics.items():
            if metric_name in ui_components['metrics_cards'].children:
                card = ui_components['metrics_cards'].children[metric_name]
                card.value = str(round(value, 4))
        
        # Update charts
        training_utils.update_charts_data(ui_components, metrics)
        
    except Exception as e:
        logging_utils.log_error(ui_components['log_accordion'], f"❌ Error update metrik: {str(e)}")

def update_confusion_matrix_ui(ui_components, cm_data, classes):
    """Update UI confusion matrix"""
    from smartcash.ui.training.utils import training_utils
    training_utils.update_confusion_matrix(ui_components, cm_data, classes)
