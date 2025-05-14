"""
File: smartcash/ui/training_config/training_strategy/handlers/button_handlers.py
Deskripsi: Handler untuk tombol UI pada komponen strategi pelatihan
"""

from typing import Dict, Any, Optional, Callable
import ipywidgets as widgets
from IPython.display import display, HTML

def setup_training_strategy_button_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk tombol pada komponen UI strategi pelatihan.
    
    Args:
        ui_components: Komponen UI
        env: Environment manager
        config: Konfigurasi model
        
    Returns:
        Dict berisi komponen UI dengan handler terpasang
    """
    try:
        # Import dengan penanganan error minimal
        from smartcash.common.config.manager import ConfigManager, get_config_manager
        from smartcash.ui.utils.alert_utils import create_status_indicator, create_info_alert
        
        # Dapatkan logger jika tersedia
        logger = ui_components.get('logger', None)
        
        # Validasi config
        if config is None: config = {}
        
        # Default config sesuai dengan training_config.yaml terbaru
        default_config = {
            # Parameter validasi
            'validation': {
                'frequency': 1,
                'iou_thres': 0.6,
                'conf_thres': 0.001
            },
            
            # Parameter multi-scale training
            'multi_scale': True,
            
            # Konfigurasi tambahan untuk proses training
            'training_utils': {
                'experiment_name': 'efficientnet_b4_training',
                'checkpoint_dir': '/content/runs/train/checkpoints',
                'tensorboard': True,
                'log_metrics_every': 10,
                'visualize_batch_every': 100,
                'gradient_clipping': 1.0,
                'mixed_precision': True
            }
        }
        
        # Update config dari UI
        def update_config_from_ui(current_config=None):
            if current_config is None: current_config = config
            
            # Update parameter validasi
            if 'validation' not in current_config:
                current_config['validation'] = {}
                
            current_config['validation']['frequency'] = ui_components['validation_frequency'].value
            current_config['validation']['iou_thres'] = ui_components['iou_threshold'].value
            current_config['validation']['conf_thres'] = ui_components['conf_threshold'].value
            
            # Update parameter multi-scale training
            current_config['multi_scale'] = ui_components['multi_scale'].value
            
            # Update parameter training_utils
            if 'training_utils' not in current_config:
                current_config['training_utils'] = {}
                
            current_config['training_utils']['experiment_name'] = ui_components['experiment_name'].value
            current_config['training_utils']['checkpoint_dir'] = ui_components['checkpoint_dir'].value
            current_config['training_utils']['tensorboard'] = ui_components['tensorboard'].value
            current_config['training_utils']['log_metrics_every'] = ui_components['log_metrics_every'].value
            current_config['training_utils']['visualize_batch_every'] = ui_components['visualize_batch_every'].value
            current_config['training_utils']['gradient_clipping'] = ui_components['gradient_clipping'].value
            current_config['training_utils']['mixed_precision'] = ui_components['mixed_precision'].value
            current_config['training_utils']['layer_mode'] = ui_components['layer_mode'].value
            
            # Update info strategi pelatihan
            update_training_strategy_info()
            
            return current_config
        
        # Update UI dari config
        def update_ui_from_config(current_config=None):
            if current_config is None: current_config = config
            
            try:
                # Update parameter validasi
                if 'validation' in current_config:
                    if 'frequency' in current_config['validation']:
                        ui_components['validation_frequency'].value = current_config['validation']['frequency']
                    if 'iou_thres' in current_config['validation']:
                        ui_components['iou_threshold'].value = current_config['validation']['iou_thres']
                    if 'conf_thres' in current_config['validation']:
                        ui_components['conf_threshold'].value = current_config['validation']['conf_thres']
                
                # Update parameter multi-scale training
                if 'multi_scale' in current_config:
                    ui_components['multi_scale'].value = current_config['multi_scale']
                
                # Update parameter training_utils
                if 'training_utils' in current_config:
                    if 'experiment_name' in current_config['training_utils']:
                        ui_components['experiment_name'].value = current_config['training_utils']['experiment_name']
                    if 'checkpoint_dir' in current_config['training_utils']:
                        ui_components['checkpoint_dir'].value = current_config['training_utils']['checkpoint_dir']
                    if 'tensorboard' in current_config['training_utils']:
                        ui_components['tensorboard'].value = current_config['training_utils']['tensorboard']
                    if 'log_metrics_every' in current_config['training_utils']:
                        ui_components['log_metrics_every'].value = current_config['training_utils']['log_metrics_every']
                    if 'visualize_batch_every' in current_config['training_utils']:
                        ui_components['visualize_batch_every'].value = current_config['training_utils']['visualize_batch_every']
                    if 'gradient_clipping' in current_config['training_utils']:
                        ui_components['gradient_clipping'].value = current_config['training_utils']['gradient_clipping']
                    if 'mixed_precision' in current_config['training_utils']:
                        ui_components['mixed_precision'].value = current_config['training_utils']['mixed_precision']
                    if 'layer_mode' in current_config['training_utils']:
                        ui_components['layer_mode'].value = current_config['training_utils']['layer_mode']
                
                # Update info strategi pelatihan
                update_training_strategy_info()
                
                if logger: logger.info("✅ UI strategi pelatihan diperbarui dari config")
            except Exception as e:
                if logger: logger.error(f"❌ Error update UI: {e}")
        
        # Update informasi strategi pelatihan
        def update_training_strategy_info():
            try:
                # Dapatkan nilai dari UI
                experiment_name = ui_components['experiment_name'].value
                checkpoint_dir = ui_components['checkpoint_dir'].value
                tensorboard = ui_components['tensorboard'].value
                validation_frequency = ui_components['validation_frequency'].value
                iou_threshold = ui_components['iou_threshold'].value
                conf_threshold = ui_components['conf_threshold'].value
                multi_scale = ui_components['multi_scale'].value
                mixed_precision = ui_components['mixed_precision'].value
                layer_mode = ui_components['layer_mode'].value
                
                # Buat informasi HTML
                info_html = f"""
                <h4>Ringkasan Strategi Pelatihan</h4>
                <ul>
                    <li><b>Experiment Name:</b> {experiment_name}</li>
                    <li><b>Checkpoint Dir:</b> {checkpoint_dir}</li>
                    <li><b>TensorBoard:</b> {'Aktif' if tensorboard else 'Nonaktif'}</li>
                    <li><b>Validasi Setiap:</b> {validation_frequency} epoch</li>
                    <li><b>IoU Threshold:</b> {iou_threshold}</li>
                    <li><b>Conf Threshold:</b> {conf_threshold}</li>
                    <li><b>Multi-scale Training:</b> {'Aktif' if multi_scale else 'Nonaktif'}</li>
                    <li><b>Mixed Precision:</b> {'Aktif' if mixed_precision else 'Nonaktif'}</li>
                    <li><b>Layer Mode:</b> {layer_mode}</li>
                </ul>
                <p><i>Catatan: Pastikan parameter pelatihan sesuai dengan kebutuhan dan kapasitas hardware.</i></p>
                """
                
                ui_components['training_strategy_info'].value = info_html
            except Exception as e:
                ui_components['training_strategy_info'].value = f"<p style='color:red'>❌ Error: {str(e)}</p>"
        
        # Handler buttons
        def on_save_click(b):
            try:
                # Dapatkan config manager
                config_manager = get_config_manager()
                
                # Update config dari UI
                updated_config = update_config_from_ui()
                
                # Simpan ke config manager
                success = config_manager.save_module_config('training_strategy', updated_config)
                
                # Register UI components untuk persistensi
                config_manager.register_ui_components('training_strategy', ui_components)
                
                # Pastikan UI components tetap terdaftar untuk persistensi
                try:
                    from smartcash.ui.utils.persistence_utils import ensure_ui_persistence
                    ensure_ui_persistence(ui_components, 'training_strategy', logger)
                except Exception as persist_error:
                    if logger: logger.warning(f"⚠️ Error saat memastikan persistensi UI: {persist_error}")
                
                # Tampilkan pesan sukses atau warning
                with ui_components['status']:
                    if success:
                        display(create_info_alert("Konfigurasi strategi pelatihan berhasil disimpan", alert_type='success'))
                    else:
                        display(create_info_alert("Konfigurasi strategi pelatihan mungkin tidak tersimpan dengan benar", alert_type='warning'))
                
                if logger: 
                    if success:
                        logger.info("✅ Konfigurasi strategi pelatihan berhasil disimpan")
                    else:
                        logger.warning("⚠️ Konfigurasi strategi pelatihan mungkin tidak tersimpan dengan benar")
            except Exception as e:
                with ui_components['status']:
                    display(create_info_alert(f"Gagal menyimpan konfigurasi: {str(e)}", alert_type='error'))
                if logger: logger.error(f"❌ Error menyimpan konfigurasi strategi pelatihan: {e}")
                
                # Pastikan UI components tetap terdaftar untuk persistensi meskipun terjadi error
                try:
                    from smartcash.ui.utils.persistence_utils import ensure_ui_persistence
                    ensure_ui_persistence(ui_components, 'training_strategy', logger)
                except Exception:
                    pass
        
        def on_reset_click(b):
            try:
                # Dapatkan config manager
                config_manager = get_config_manager()
                
                # Dapatkan konfigurasi saat ini dan update dengan nilai default
                current_config = config_manager.get_module_config('training_strategy') or {}
                current_config.update(default_config)
                
                # Simpan konfigurasi default
                success = config_manager.save_module_config('training_strategy', default_config)
                
                # Update UI dari default config
                update_ui_from_config(default_config)
                
                # Pastikan UI components tetap terdaftar untuk persistensi
                try:
                    from smartcash.ui.utils.persistence_utils import ensure_ui_persistence
                    ensure_ui_persistence(ui_components, 'training_strategy', logger)
                except Exception as persist_error:
                    if logger: logger.warning(f"⚠️ Error saat memastikan persistensi UI: {persist_error}")
                
                # Tampilkan pesan sukses atau warning
                with ui_components['status']:
                    if success:
                        display(create_info_alert("Konfigurasi strategi pelatihan berhasil direset ke default", alert_type='success'))
                    else:
                        display(create_info_alert("Konfigurasi strategi pelatihan direset di UI tetapi mungkin tidak tersimpan ke file", alert_type='warning'))
                
                if logger: 
                    if success:
                        logger.info("✅ Konfigurasi strategi pelatihan berhasil direset ke default")
                    else:
                        logger.warning("⚠️ Konfigurasi direset di UI tetapi mungkin tidak tersimpan ke file")
            except Exception as e:
                with ui_components['status']:
                    display(create_info_alert(f"Gagal mereset konfigurasi: {str(e)}", alert_type='error'))
                if logger: logger.error(f"❌ Error mereset konfigurasi strategi pelatihan: {e}")
                
                # Pastikan UI components tetap terdaftar untuk persistensi meskipun terjadi error
                try:
                    from smartcash.ui.utils.persistence_utils import ensure_ui_persistence
                    ensure_ui_persistence(ui_components, 'training_strategy', logger)
                except Exception:
                    pass
        
        # Register handlers
        ui_components['save_button'].on_click(on_save_click)
        ui_components['reset_button'].on_click(on_reset_click)
        
        # Tambahkan fungsi ke ui_components
        ui_components['update_config_from_ui'] = update_config_from_ui
        ui_components['update_ui_from_config'] = update_ui_from_config
        ui_components['update_training_strategy_info'] = update_training_strategy_info
        
        # Inisialisasi UI dari config yang disimpan
        try:
            # Dapatkan config manager
            config_manager = ConfigManager.get_instance()
            
            # Coba dapatkan konfigurasi yang disimpan
            saved_config = config_manager.get_module_config('training_strategy')
            
            if saved_config:
                # Update UI dari konfigurasi yang disimpan
                update_ui_from_config(saved_config)
                if logger: logger.info("✅ UI strategi pelatihan diinisialisasi dari konfigurasi yang disimpan")
            else:
                # Jika tidak ada konfigurasi yang disimpan, gunakan default
                update_ui_from_config(default_config)
                if logger: logger.info("ℹ️ UI strategi pelatihan diinisialisasi dari konfigurasi default")
                
            # Register UI components untuk persistensi
            config_manager.register_ui_components('training_strategy', ui_components)
        except Exception as e:
            # Fallback ke default jika terjadi error
            update_ui_from_config(default_config)
            if logger: logger.warning(f"⚠️ Error inisialisasi UI strategi pelatihan: {e}, menggunakan default")
        
    except Exception as e:
        # Fallback sederhana jika terjadi error
        if 'status' in ui_components:
            with ui_components['status']: display(HTML(f"<p style='color:red'>❌ Error setup training strategy button handler: {str(e)}</p>"))
        else: print(f"❌ Error setup training strategy button handler: {str(e)}")
    
    return ui_components
