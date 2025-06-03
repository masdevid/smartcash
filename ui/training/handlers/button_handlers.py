"""
File: smartcash/ui/training/handlers/button_handlers.py
Deskripsi: Handler untuk tombol-tombol training dengan integrasi model service
"""

import torch
from typing import Dict, Any
from smartcash.ui.utils.fallback_utils import try_operation_safe
from smartcash.model.config.model_config import ModelConfig

def setup_training_button_handlers(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Setup handlers untuk tombol training dengan one-liner style"""
    
    def handle_start_training(button):
        """Handler untuk tombol mulai training - delegasi ke training service"""
        try:
            # Reset log output
            ui_components.get('log_output') and ui_components['log_output'].clear_output(wait=True)
            
            # Validasi training manager
            training_manager = ui_components.get('training_manager')
            if not training_manager:
                ui_components['logger'].error("❌ Training manager tidak tersedia")
                return
            
            # Disable start button, enable stop button
            button.disabled = True
            ui_components.get('stop_button') and setattr(ui_components['stop_button'], 'disabled', False)
            
            # Show progress container
            progress_container = ui_components.get('progress_container')
            if progress_container and hasattr(progress_container, 'layout'):
                progress_container.layout.display = 'flex'
            
            # Load konfigurasi training
            training_config = _load_training_config()
            
            # Start training melalui training manager
            ui_components['logger'].info("🚀 Memulai training...")
            training_manager.start_training(training_config)
            
        except Exception as e:
            ui_components['logger'].error(f"❌ Error memulai training: {str(e)}")
            # Reset button states
            button.disabled = False
            ui_components.get('stop_button') and setattr(ui_components['stop_button'], 'disabled', True)
    
    def handle_stop_training(button):
        """Handler untuk tombol stop training"""
        try:
            # Reset log output
            ui_components.get('log_output') and ui_components['log_output'].clear_output(wait=True)
            
            training_manager = ui_components.get('training_manager')
            if training_manager:
                ui_components['logger'].info("🛑 Menghentikan training...")
                training_manager.stop_training()
            
            # Reset button states
            button.disabled = True
            ui_components.get('start_button') and setattr(ui_components['start_button'], 'disabled', False)
            
        except Exception as e:
            ui_components['logger'].error(f"❌ Error menghentikan training: {str(e)}")
    
    def handle_reset_metrics(button):
        """Handler untuk reset metrics dan chart"""
        try:
            # Reset log output
            ui_components.get('log_output') and ui_components['log_output'].clear_output(wait=True)
            
            # Reset chart output
            chart_output = ui_components.get('chart_output')
            chart_output and chart_output.clear_output(wait=True)
            
            # Reset metrics output
            metrics_output = ui_components.get('metrics_output')
            metrics_output and metrics_output.clear_output(wait=True)
            
            # Reset melalui training manager jika ada
            training_manager = ui_components.get('training_manager')
            training_manager and hasattr(training_manager, 'reset_metrics') and training_manager.reset_metrics()
            
            ui_components['logger'].success("✅ Metrics dan chart berhasil direset")
            
        except Exception as e:
            ui_components['logger'].error(f"❌ Error reset metrics: {str(e)}")
    
    def handle_validate_model(button):
        """Handler untuk validasi model readiness"""
        try:
            # Reset log output
            ui_components.get('log_output') and ui_components['log_output'].clear_output(wait=True)
            
            model_manager = ui_components.get('model_manager')
            if not model_manager:
                ui_components['logger'].error("❌ Model manager tidak tersedia")
                return
            
            ui_components['logger'].info("🔍 Memvalidasi model readiness...")
            
            # Validasi model sudah dibangun
            if not model_manager.is_model_built():
                ui_components['logger'].info("🔄 Model belum dibangun, membangun model...")
                model_manager.build_model()
            
            # Validasi device
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            ui_components['logger'].info(f"💻 Device: {device}")
            
            # Validasi model structure
            model_info = {
                'model_type': model_manager.get_model_type(),
                'backbone': model_manager.get_backbone_type(),
                'num_classes': model_manager.get_num_classes(),
                'layer_mode': model_manager.get_layer_mode(),
                'detection_layers': model_manager.get_detection_layers()
            }
            
            ui_components['logger'].success("✅ Model validation berhasil:")
            for key, value in model_info.items():
                ui_components['logger'].info(f"   • {key}: {value}")
                
        except Exception as e:
            ui_components['logger'].error(f"❌ Error validasi model: {str(e)}")
    
    def handle_refresh_config(button):
        """Handler untuk refresh konfigurasi dari semua modul dari YAML files"""
        try:
            # Reset log output jika tersedia
            ui_components.get('log_output') and ui_components['log_output'].clear_output(wait=True)
            
            ui_components['logger'].info("🔄 Memuat ulang konfigurasi dari file YAML...")
            
            # Force reload configurations dari file YAML asli
            from smartcash.common.config.manager import get_config_manager
            config_manager = get_config_manager()
            config_manager.reload_all_configs(force_reload=True)  # Force reload dari disk
            
            # Trigger config update callback jika ada
            config_callback = ui_components.get('config_update_callback')
            if config_callback:
                # Load fresh config dari YAML yang diperbarui
                fresh_config = _load_all_configs()
                config_callback(fresh_config)
                ui_components['logger'].success("✅ Konfigurasi berhasil diperbarui dari YAML files")
            else:
                ui_components['logger'].warning("⚠️ Config update callback tidak tersedia")
                
            # Update status panel jika ada
            if 'status_panel' in ui_components:
                from smartcash.ui.utils.alert_utils import update_status_panel
                update_status_panel(ui_components['status_panel'], f"📚 Konfigurasi berhasil dimuat dari YAML files", "success")
                
        except Exception as e:
            ui_components['logger'].error(f"❌ Error saat refresh config dari YAML: {str(e)}")
            if 'status_panel' in ui_components:
                from smartcash.ui.utils.alert_utils import update_status_panel
                update_status_panel(ui_components['status_panel'], f"⚠️ Error: {str(e)}", "error")
    
    # Register handlers dengan one-liner
    ui_components.get('start_button') and ui_components['start_button'].on_click(handle_start_training)
    ui_components.get('stop_button') and ui_components['stop_button'].on_click(handle_stop_training)
    ui_components.get('reset_button') and ui_components['reset_button'].on_click(handle_reset_metrics)
    ui_components.get('validate_button') and ui_components['validate_button'].on_click(handle_validate_model)
    ui_components.get('refresh_button') and ui_components['refresh_button'].on_click(handle_refresh_config)

def _load_training_config() -> Dict[str, Any]:
    """Load konfigurasi training dari file YAML"""
    try:
        from smartcash.common.config.manager import get_config_manager
        config_manager = get_config_manager()
        
        # Load training config
        training_config = config_manager.get_config('training') or {}
        model_config = config_manager.get_config('model') or {}
        hyperparams_config = config_manager.get_config('hyperparameters') or {}
        
        # Merge configs
        merged_config = {
            'epochs': training_config.get('epochs', 100),
            'learning_rate': hyperparams_config.get('learning_rate', 0.001),
            'batch_size': model_config.get('batch_size', 16),
            'weight_decay': hyperparams_config.get('weight_decay', 0.0005),
            'early_stopping': training_config.get('early_stopping', True),
            'patience': training_config.get('patience', 10),
            'save_best': training_config.get('save_best', True),
            'save_interval': training_config.get('save_interval', 10)
        }
        
        return merged_config
        
    except Exception as e:
        # Return default config jika gagal load
        return {
            'epochs': 100,
            'learning_rate': 0.001,
            'batch_size': 16,
            'weight_decay': 0.0005,
            'early_stopping': True,
            'patience': 10,
            'save_best': True,
            'save_interval': 10
        }

def _load_all_configs() -> Dict[str, Any]:
    """Load semua konfigurasi dari berbagai modul berdasarkan struktur YAML terbaru"""
    try:
        from smartcash.common.config.manager import get_config_manager
        config_manager = get_config_manager()
        
        # Load semua configs dari YAML sesuai struktur yang sudah dipisahkan
        # Berdasarkan memory: hyperparameters_config.yaml, model_config.yaml, training_config.yaml
        configs = {
            # Config utama
            'model': config_manager.get_config('model') or {},
            'hyperparameters': config_manager.get_config('hyperparameters') or {},
            'training': config_manager.get_config('training') or {},
            
            # Modul-modul spesifik dari training_config.yaml
            'training_strategy': config_manager.get_config('training_strategy') or {},
            
            # Config lainnya
            'backbone': config_manager.get_config('backbone') or {},
            'detector': config_manager.get_config('detector') or {},
            'paths': config_manager.get_config('paths') or {},
            'augmentation': config_manager.get_config('augmentation') or {}
        }
        
        return configs
        
    except Exception as e:
        # Log error jika diperlukan untuk debugging
        print(f"Error saat loading configs: {str(e)}")
        return {}