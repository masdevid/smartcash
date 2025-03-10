"""
File: smartcash/ui_handlers/dataset_handlers.py
Author: Alfrida Sabar (refactored)
Deskripsi: Handler untuk UI pengelolaan dataset mata uang Rupiah
"""

import os
import sys
import yaml
from pathlib import Path
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets

# Fallback import untuk development tanpa smartcash
try:
    from smartcash.handlers.dataset import DatasetManager
    from smartcash.utils.logger import get_logger
    from smartcash.utils.observer import EventDispatcher, EventTopics
    from smartcash.configs import get_config
except ImportError:
    # Placeholder saat modul belum ada
    def get_logger(name):
        class DummyLogger:
            def info(self, msg): print(f"[INFO] {msg}")
            def warning(self, msg): print(f"[WARNING] {msg}")
            def error(self, msg): print(f"[ERROR] {msg}")
            def success(self, msg): print(f"[SUCCESS] {msg}")
        return DummyLogger()
    
    class DummyDatasetManager:
        def __init__(self, config, logger=None):
            self.config = config
            self.logger = logger
            
        def download_dataset(self, **kwargs):
            print("📥 Download dataset simulation")
            return "data/train"
            
        def validate_dataset(self, **kwargs):
            print("🔍 Validate dataset simulation")
            return {"valid_files": 100, "invalid_files": 0}
            
        def augment_dataset(self, **kwargs):
            print("🔄 Augment dataset simulation")
            return {"augmented_files": 200}
            
        def analyze_dataset(self, **kwargs):
            print("📊 Analyze dataset simulation")
            return {"classes": {"100": 50, "50": 50, "20": 50}}
            
        def visualize_class_distribution(self, **kwargs):
            print("📈 Visualize dataset simulation")
            return "visualizations/class_distribution.png"
    
    DatasetManager = DummyDatasetManager
    
    class DummyEventDispatcher:
        @staticmethod
        def register(event_type, callback): pass
        @staticmethod
        def notify(event_type, sender, **kwargs): pass
    
    EventDispatcher = DummyEventDispatcher
    EventTopics = type('obj', (object,), {
        'DOWNLOAD_PROGRESS': 'download.progress',
        'VALIDATION_PROGRESS': 'validation.progress',
        'AUGMENTATION_PROGRESS': 'augmentation.progress'
    })
    
    def get_config(config_path=None):
        return {
            'data': {
                'source': 'roboflow',
                'roboflow': {
                    'api_key': '',
                    'workspace': 'smartcash-wo2us',
                    'project': 'rupiah-emisi-2022',
                    'version': '3'
                }
            }
        }

def setup_dataset_handlers(ui):
    """Setup semua handler untuk UI dataset."""
    logger = get_logger("dataset_ui")
    config = get_config()
    
    # Handler untuk perubahan sumber dataset
    def on_source_change(change):
        if change['new'] == 'roboflow':
            ui['roboflow_card'].layout.display = ''
        else:
            ui['roboflow_card'].layout.display = 'none'
    
    ui['source_radio'].observe(on_source_change, names='value')
    
    # === Download Dataset Handler ===
    def handle_download_dataset():
        def on_download(b):
            with ui['download_status']:
                clear_output()
                
                # Validasi input
                if ui['source_radio'].value == 'roboflow' and not ui['roboflow_api_key'].value:
                    display(HTML(
                        """<div style="padding: 10px; background: #fff3cd; color: #856404; border-left: 4px solid #ffc107;">
                            <p><b>⚠️ Error:</b> Roboflow API Key tidak boleh kosong.</p>
                        </div>"""
                    ))
                    return
                
                # Update config
                if ui['source_radio'].value == 'roboflow':
                    config['data']['source'] = 'roboflow'
                    config['data']['roboflow']['api_key'] = ui['roboflow_api_key'].value
                    config['data']['roboflow']['workspace'] = ui['roboflow_workspace'].value
                    config['data']['roboflow']['project'] = ui['roboflow_project'].value
                    config['data']['roboflow']['version'] = ui['roboflow_version'].value
                else:
                    config['data']['source'] = 'local'
                
                # Show progress bar
                ui['download_progress'].layout.visibility = 'visible'
                
                # Setup observer untuk progress
                def update_download_progress(sender, progress=0, total=100, message=None, **kwargs):
                    ui['download_progress'].max = total
                    ui['download_progress'].value = progress
                    if message:
                        ui['download_progress'].description = message
                
                # Register observer
                EventDispatcher.register(EventTopics.DOWNLOAD_PROGRESS, update_download_progress)
                
                # Initialize dataset manager
                display(HTML("<p>🔄 Menginisialisasi Dataset Manager...</p>"))
                dataset_manager = DatasetManager(config, logger=logger)
                
                # Download dataset
                display(HTML("<p>📥 Memulai download dataset...</p>"))
                try:
                    dataset_path = dataset_manager.download_dataset(
                        format="yolov5",
                        show_progress=True
                    )
                    
                    # Show success message
                    display(HTML(
                        f"""<div style="padding: 10px; background: #d4edda; color: #155724; border-left: 4px solid #28a745;">
                            <h3 style="margin-top: 0;">✅ Dataset berhasil didownload!</h3>
                            <p>Dataset tersimpan di <code>{dataset_path}</code></p>
                        </div>"""
                    ))
                except Exception as e:
                    display(HTML(
                        f"""<div style="padding: 10px; background: #f8d7da; color: #721c24; border-left: 4px solid #dc3545;">
                            <h3 style="margin-top: 0;">❌ Error saat download dataset</h3>
                            <p>Error: {str(e)}</p>
                        </div>"""
                    ))
                finally:
                    # Hide progress bar
                    ui['download_progress'].layout.visibility = 'hidden'
        
        ui['download_button'].on_click(on_download)
    
    # === Validation Dataset Handler ===