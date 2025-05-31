"""
File: examples/notification_manager_usage.py
Deskripsi: Contoh penggunaan NotificationManager dalam SmartCash
"""

import time
import ipywidgets as widgets
from IPython.display import display
from typing import Dict, Any

# Import NotificationManager
from smartcash.components.notification import get_notification_manager

def setup_ui_components() -> Dict[str, Any]:
    """Setup UI components untuk contoh."""
    # Buat container utama
    main_container = widgets.VBox()
    
    # Progress bar
    progress_bar = widgets.IntProgress(
        value=0,
        min=0,
        max=100,
        description='Progress:',
        bar_style='info',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='100%', margin='10px 0px')
    )
    
    # Status label
    status_label = widgets.HTML(
        value="<div style='color: #6c757d;'>Siap memulai proses</div>",
        layout=widgets.Layout(margin='5px 0px')
    )
    
    # Log output
    log_output = widgets.Output(
        layout=widgets.Layout(
            width='100%',
            max_height='200px',
            overflow='auto',
            border='1px solid #ddd',
            padding='10px'
        )
    )
    
    # Tombol untuk memulai proses
    start_button = widgets.Button(
        description='Mulai Proses',
        button_style='primary',
        icon='play',
        layout=widgets.Layout(width='auto', margin='10px 5px 10px 0px')
    )
    
    # Tombol untuk berhenti
    stop_button = widgets.Button(
        description='Berhenti',
        button_style='danger',
        icon='stop',
        layout=widgets.Layout(width='auto', margin='10px 5px')
    )
    stop_button.layout.display = 'none'  # Sembunyikan di awal
    
    # Tambahkan semua komponen ke container
    main_container.children = [
        widgets.HTML("<h3>Contoh NotificationManager</h3>"),
        progress_bar,
        status_label,
        widgets.HBox([start_button, stop_button]),
        log_output
    ]
    
    # Setup logger sederhana
    class SimpleLogger:
        def __init__(self, output):
            self.output = output
            
        def info(self, message):
            with self.output:
                print(f"[INFO] {message}")
                
        def error(self, message):
            with self.output:
                print(f"[ERROR] {message}")
                
        def warning(self, message):
            with self.output:
                print(f"[WARNING] {message}")
                
        def debug(self, message):
            with self.output:
                print(f"[DEBUG] {message}")
    
    logger = SimpleLogger(log_output)
    
    # Buat dictionary ui_components
    ui_components = {
        'progress_bar': progress_bar,
        'status_label': status_label,
        'log_output': log_output,
        'start_button': start_button,
        'stop_button': stop_button,
        'logger': logger,
        'main_container': main_container
    }
    
    # Tambahkan fungsi update_progress
    def update_progress(progress_type, value, message, status='info'):
        if progress_type == 'overall':
            progress_bar.value = value
            progress_bar.description = f'Progress: {value}%'
            if status == 'error':
                progress_bar.bar_style = 'danger'
            elif status == 'success':
                progress_bar.bar_style = 'success'
            else:
                progress_bar.bar_style = 'info'
        
        status_label.value = f"<div style='color: {get_status_color(status)};'>{message}</div>"
        logger.info(f"Progress ({progress_type}): {value}% - {message}")
    
    def get_status_color(status):
        colors = {
            'info': '#007bff',
            'warning': '#ffc107',
            'error': '#dc3545',
            'success': '#28a745',
            'default': '#6c757d'
        }
        return colors.get(status, colors['default'])
    
    ui_components['update_progress'] = update_progress
    
    return ui_components

def simulate_process(ui_components: Dict[str, Any]):
    """Simulasi proses dengan NotificationManager."""
    # Dapatkan NotificationManager
    notification_manager = get_notification_manager(ui_components)
    
    # Dapatkan komponen UI
    start_button = ui_components['start_button']
    stop_button = ui_components['stop_button']
    logger = ui_components['logger']
    
    # Flag untuk menandakan proses berhenti
    should_stop = [False]
    
    def on_start_button_click(b):
        # Sembunyikan tombol start dan tampilkan tombol stop
        start_button.layout.display = 'none'
        stop_button.layout.display = 'block'
        
        # Reset flag stop
        should_stop[0] = False
        
        # Mulai proses di thread terpisah
        import threading
        thread = threading.Thread(target=run_process, args=(should_stop,))
        thread.daemon = True
        thread.start()
    
    def on_stop_button_click(b):
        # Set flag stop
        should_stop[0] = True
        logger.warning("Menghentikan proses...")
    
    # Assign handler ke tombol
    start_button.on_click(on_start_button_click)
    stop_button.on_click(on_stop_button_click)
    
    def run_process(should_stop):
        try:
            # Notifikasi awal proses
            notification_manager.notify_process_start(
                "simulation", 
                "Memulai simulasi proses",
                total_steps=5
            )
            
            # Simulasi langkah 1: Inisialisasi
            notification_manager.update_status("Menginisialisasi proses...")
            for i in range(10):
                if should_stop[0]:
                    raise Exception("Proses dihentikan oleh pengguna")
                notification_manager.update_progress("simulation", i, 10, f"Inisialisasi langkah {i+1}/10")
                time.sleep(0.2)
            
            # Simulasi langkah 2: Memproses data
            notification_manager.update_status("Memproses data...")
            for i in range(20):
                if should_stop[0]:
                    raise Exception("Proses dihentikan oleh pengguna")
                notification_manager.update_progress("simulation", i, 20, f"Memproses data {i+1}/20")
                time.sleep(0.1)
            
            # Simulasi langkah 3: Menganalisis hasil
            notification_manager.update_status("Menganalisis hasil...")
            for i in range(15):
                if should_stop[0]:
                    raise Exception("Proses dihentikan oleh pengguna")
                notification_manager.update_progress("simulation", i, 15, f"Menganalisis hasil {i+1}/15")
                time.sleep(0.15)
            
            # Simulasi langkah 4: Menyimpan hasil
            notification_manager.update_status("Menyimpan hasil...")
            for i in range(5):
                if should_stop[0]:
                    raise Exception("Proses dihentikan oleh pengguna")
                notification_manager.update_progress("simulation", i, 5, f"Menyimpan hasil {i+1}/5")
                time.sleep(0.3)
            
            # Simulasi selesai
            notification_manager.notify_process_complete(
                "simulation",
                "Simulasi proses selesai dengan sukses",
                stats={
                    "total_steps": 4,
                    "total_items": 50,
                    "duration": "5.5 detik"
                }
            )
            
        except Exception as e:
            # Notifikasi error
            notification_manager.notify_process_error(
                "simulation",
                str(e),
                exception=e
            )
        finally:
            # Kembalikan tampilan tombol
            start_button.layout.display = 'block'
            stop_button.layout.display = 'none'
    
    # Tampilkan UI
    display(ui_components['main_container'])
    logger.info("UI siap digunakan. Klik 'Mulai Proses' untuk memulai simulasi.")

# Contoh penggunaan dalam notebook
if __name__ == "__main__":
    # Setup UI components
    ui_components = setup_ui_components()
    
    # Jalankan simulasi
    simulate_process(ui_components)
