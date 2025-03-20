"""
File: smartcash/ui/dataset/preprocessing_progress_handler.py
Deskripsi: Handler progress tracking preprocessing dataset dengan perbaikan perhitungan progress keseluruhan dan saat ini
"""

from typing import Dict, Any
from IPython.display import display
from smartcash.ui.utils.constants import ICONS

def setup_progress_handler(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """Setup handler progress tracking dengan integrasi observer standar dan pengaturan proporsi progress yang lebih akurat."""
    logger = ui_components.get('logger')
    
    # Status global untuk tracking progress
    progress_state = {
        'total_steps': 3,             # Total langkah preprocessing (default: load, process, save)
        'current_step': 0,            # Langkah saat ini
        'step_weights': [0.1, 0.8, 0.1], # Bobot setiap langkah dalam progress keseluruhan
        'current_progress': 0,        # Progress dalam langkah saat ini (0-100)
        'current_total': 100,         # Total progress dalam langkah saat ini
        'overall_progress': 0,        # Progress keseluruhan (0-100)
        'step_names': {               # Nama setiap langkah untuk tampilan
            0: "Persiapan dataset",
            1: "Preprocessing gambar",
            2: "Penyimpanan hasil"
        }
    }
    
    # Fungsi untuk menghitung progress keseluruhan berdasarkan step dan progress saat ini
    def calculate_overall_progress() -> int:
        """
        Menghitung progress keseluruhan berdasarkan step saat ini dan progressnya.
        
        Returns:
            Persentase progress keseluruhan (0-100)
        """
        # Progress dari step-step sebelumnya
        completed_steps_progress = sum(progress_state['step_weights'][:progress_state['current_step']])
        
        # Progress dari step saat ini
        if progress_state['current_total'] > 0:
            current_step_contribution = (progress_state['current_progress'] / progress_state['current_total']) * progress_state['step_weights'][progress_state['current_step']]
        else:
            current_step_contribution = 0
            
        # Total progress (dalam skala 0-1)
        total_progress = completed_steps_progress + current_step_contribution
        
        # Konversi ke persentase (0-100)
        return int(total_progress * 100)
    
    # Fungsi progress callback yang lebih akurat untuk tracking keseluruhan dan saat ini
    def progress_callback(progress=None, total=None, message=None, status='info', 
                         current_step=None, current_total=None, step=None, **kwargs):
        """
        Progress callback dengan kalkulasi progress yang terintegrasi.
        
        Args:
            progress: Nilai progress saat ini dalam step (absolut)
            total: Total nilai progress dalam step
            message: Pesan progress 
            status: Status progress ('info', 'success', 'warning', 'error')
            current_step: Nilai step saat ini (0-based)
            current_total: Total maksimum step
            step: Nama langkah saat ini (string identifier)
            **kwargs: Parameter lain yang diteruskan dari caller
        """
        # Skip jika preprocessing sudah dihentikan
        if not ui_components.get('preprocessing_running', True): 
            return
        
        # Update step jika diberikan
        if current_step is not None and current_total is not None:
            # Update progress_state dengan info step
            progress_state['current_step'] = min(current_step, progress_state['total_steps']-1)
            
            # Jika step berubah, reset progress saat ini
            progress_state['current_progress'] = 0
            progress_state['current_total'] = current_total
        
        # Update progress saat ini jika diberikan
        if progress is not None and total is not None:
            progress_state['current_progress'] = progress
            progress_state['current_total'] = total
        
        # Hitung progress keseluruhan
        progress_state['overall_progress'] = calculate_overall_progress()
        
        # Format pesan dengan info step
        formatted_message = message
        if not formatted_message and step in progress_state['step_names']:
            formatted_message = progress_state['step_names'][step]
        elif not formatted_message:
            # Default message berdasarkan step
            step_name = progress_state['step_names'].get(progress_state['current_step'], f"Langkah {progress_state['current_step']+1}")
            current_percent = int((progress_state['current_progress'] / progress_state['current_total']) * 100) if progress_state['current_total'] > 0 else 0
            formatted_message = f"{step_name}: {current_percent}%"
            
        # Update progress bar utama (overall)
        if 'progress_bar' in ui_components:
            ui_components['progress_bar'].max = 100
            ui_components['progress_bar'].value = progress_state['overall_progress']
            ui_components['progress_bar'].description = f"Overall: {progress_state['overall_progress']}%"
            ui_components['progress_bar'].layout.visibility = 'visible'
        
        # Update progress bar saat ini (current step)
        if 'current_progress' in ui_components:
            ui_components['current_progress'].max = progress_state['current_total']
            ui_components['current_progress'].value = progress_state['current_progress']
            current_percent = int((progress_state['current_progress'] / progress_state['current_total']) * 100) if progress_state['current_total'] > 0 else 0
            ui_components['current_progress'].description = f"Current: {current_percent}%"
            ui_components['current_progress'].layout.visibility = 'visible'
                
        # Notifikasi observer dengan observer standar
        try:
            from smartcash.components.observer import notify
            from smartcash.components.observer.event_topics_observer import EventTopics
            
            # Notifikasi progress keseluruhan pada perubahan signifikan 
            if progress_state['overall_progress'] % 5 == 0 or progress is None or total is None:
                notify(
                    event_type=EventTopics.PREPROCESSING_PROGRESS, 
                    sender="preprocessing_handler",
                    message=formatted_message,
                    progress=progress_state['overall_progress'],
                    total=100
                )
            
            # Notifikasi progress saat ini (untuk substep detail)
            if progress is not None and total is not None and progress % 10 == 0:
                notify(
                    event_type=EventTopics.PREPROCESSING_CURRENT_PROGRESS, 
                    sender="preprocessing_handler",
                    message=formatted_message,
                    progress=progress,
                    total=total,
                    current_step=progress_state['current_step'],
                    step=step
                )
        except ImportError:
            pass
    
    # Fungsi untuk registrasi callback ke dataset manager dengan validasi
    def register_progress_callback(dataset_manager):
        """Register callback progress ke dataset manager."""
        if not dataset_manager or not hasattr(dataset_manager, 'register_progress_callback'): 
            return False
        
        # Register callback ke dataset manager
        dataset_manager.register_progress_callback(progress_callback)
        
        # Reset progress state setiap kali registrasi
        progress_state['current_step'] = 0
        progress_state['current_progress'] = 0
        progress_state['overall_progress'] = 0
        
        return True
    
    # Setup observer integrasi full jika tersedia
    try:
        from smartcash.components.observer.event_topics_observer import EventTopics
        from smartcash.ui.handlers.observer_handler import create_progress_observer
        
        # Setup progress observer untuk progress utama (overall)
        create_progress_observer(
            ui_components=ui_components,
            event_type=[
                EventTopics.PREPROCESSING_PROGRESS,
                EventTopics.PREPROCESSING_START,
                EventTopics.PREPROCESSING_END,
                EventTopics.PREPROCESSING_ERROR
            ],
            total=100,  # Overall progress selalu 0-100
            progress_widget_key='progress_bar',
            output_widget_key='status',
            observer_group='preprocessing_observers'
        )
        
        # Setup progress observer untuk current progress
        create_progress_observer(
            ui_components=ui_components,
            event_type=EventTopics.PREPROCESSING_CURRENT_PROGRESS,
            total=100,  # Default 100, akan diupdate dinamis
            progress_widget_key='current_progress',
            update_output=False,
            observer_group='preprocessing_observers'
        )
        
        if logger: logger.info(f"{ICONS['success']} Progress tracking terintegrasi berhasil setup")
    except (ImportError, AttributeError) as e:
        if logger: logger.warning(f"{ICONS['warning']} Observer progress tidak tersedia: {str(e)}")
    
    # Fungsi untuk advance ke step berikutnya
    def advance_to_step(step_index: int, step_name: str = None, message: str = None):
        """
        Pindah ke step berikutnya dan reset progress saat ini.
        
        Args:
            step_index: Indeks step (0-based)
            step_name: Nama step (opsional)
            message: Pesan untuk ditampilkan
        """
        # Update progress state
        progress_state['current_step'] = min(step_index, progress_state['total_steps']-1)
        progress_state['current_progress'] = 0
        
        # Update step name jika diberikan
        if step_name and step_index in progress_state['step_names']:
            progress_state['step_names'][step_index] = step_name
            
        # Update progress display
        progress_callback(
            progress=0, 
            total=100, 
            message=message or f"Memulai {progress_state['step_names'].get(step_index, f'Langkah {step_index+1}')}",
            current_step=step_index,
            current_total=100,
            step=step_index
        )
        
        # Notifikasi observer
        try:
            from smartcash.components.observer import notify
            from smartcash.components.observer.event_topics_observer import EventTopics
            notify(
                event_type=EventTopics.PREPROCESSING_STEP_CHANGE, 
                sender="preprocessing_handler",
                message=message or f"Memulai {progress_state['step_names'].get(step_index, f'Langkah {step_index+1}')}",
                current_step=step_index,
                total_steps=progress_state['total_steps']
            )
        except ImportError:
            pass
    
    # Update progress utility dengan penanganan step
    def update_progress_bar(progress, total, message=None, step_index=None):
        """
        Update progress bar dengan parameter step yang lebih lengkap.
        
        Args:
            progress: Nilai progress saat ini
            total: Nilai maksimal progress
            message: Pesan opsional
            step_index: Indeks step saat ini (opsional)
        """
        # Update step jika diberikan
        if step_index is not None:
            progress_state['current_step'] = min(step_index, progress_state['total_steps']-1)
        
        # Update progress callback
        progress_callback(
            progress=progress,
            total=total,
            message=message,
            current_step=progress_state['current_step'],
            current_total=total
        )
    
    # Reset progress ke kondisi awal
    def reset_progress_bar():
        """Reset semua komponen progress ke nilai awal."""
        # Reset progress state
        progress_state['current_step'] = 0
        progress_state['current_progress'] = 0
        progress_state['overall_progress'] = 0
        
        # Reset widgets
        if 'progress_bar' in ui_components:
            ui_components['progress_bar'].value = 0
            ui_components['progress_bar'].description = 'Overall:'
            ui_components['progress_bar'].layout.visibility = 'hidden'
            
        if 'current_progress' in ui_components:
            ui_components['current_progress'].value = 0
            ui_components['current_progress'].description = 'Current:'
            ui_components['current_progress'].layout.visibility = 'hidden'
    
    # Fungsi untuk mengatur jumlah step dan bobot masing-masing
    def configure_progress_steps(total_steps: int, step_weights: List[float] = None, step_names: Dict[int, str] = None):
        """
        Konfigurasi jumlah step dalam progress dan bobotnya masing-masing.
        
        Args:
            total_steps: Jumlah total step
            step_weights: Bobot setiap step (harus berjumlah 1.0)
            step_names: Nama setiap step untuk display
        """
        # Update total steps
        progress_state['total_steps'] = max(1, total_steps)
        
        # Update step weights jika diberikan
        if step_weights and len(step_weights) == total_steps:
            # Pastikan jumlah bobot = 1.0
            total_weight = sum(step_weights)
            if abs(total_weight - 1.0) > 0.001:  # Allow small floating point errors
                # Normalize weights
                progress_state['step_weights'] = [w / total_weight for w in step_weights]
            else:
                progress_state['step_weights'] = step_weights
        else:
            # Default equal weights
            progress_state['step_weights'] = [1.0 / total_steps] * total_steps
            
        # Update step names jika diberikan
        if step_names:
            progress_state['step_names'].update(step_names)
            
        # Reset progress state
        progress_state['current_step'] = 0
        progress_state['current_progress'] = 0
        progress_state['overall_progress'] = 0
    
    # Tambahkan fungsi progress dan register ke UI components
    ui_components.update({
        'progress_callback': progress_callback,
        'register_progress_callback': register_progress_callback,
        'update_progress_bar': update_progress_bar,
        'reset_progress_bar': reset_progress_bar,
        'advance_to_step': advance_to_step,
        'configure_progress_steps': configure_progress_steps
    })
    
    # Registrasi langsung jika dataset manager sudah ada
    if 'dataset_manager' in ui_components:
        register_progress_callback(ui_components['dataset_manager'])
    
    return ui_components