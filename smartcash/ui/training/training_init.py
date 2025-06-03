"""\nFile: smartcash/ui/training/training_init.py\nDeskripsi: Training initializer dengan integrasi refresh konfigurasi dan pencegahan tampilan ganda\n"""

from typing import Dict, Any, List
from IPython.display import display

from smartcash.ui.utils.common_initializer import CommonInitializer
from smartcash.ui.utils.logging_utils import suppress_all_outputs
from smartcash.ui.utils.ui_logger_namespace import TRAINING_LOGGER_NAMESPACE, KNOWN_NAMESPACES

MODULE_LOGGER_NAME = KNOWN_NAMESPACES[TRAINING_LOGGER_NAMESPACE]


class TrainingInitializer(CommonInitializer):
    """Training UI initializer dengan config callback integration"""
    
    def __init__(self):
        super().__init__(MODULE_LOGGER_NAME, TRAINING_LOGGER_NAMESPACE)
        self.config_update_callbacks = []
        self._ui_displayed = False  # Flag untuk mencegah tampilan ganda
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create UI components dengan safe error handling"""
        try:
            from smartcash.ui.training.components.training_form import create_training_form
            from smartcash.ui.training.components.training_layout import create_training_layout
            
            # Create form components dengan config
            form_components = create_training_form(config)
            
            # Create layout arrangement
            layout_components = create_training_layout(form_components)
            
            # Setup config callback untuk info display update
            self._setup_config_callback(layout_components)
            
            return layout_components
            
        except Exception as e:
            self.logger.error(f"❌ Error creating UI components: {str(e)}")
            return self._create_fallback_components(str(e))
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers dengan model integration dan progress tracking"""
        try:
            from smartcash.ui.training.handlers.training_handlers import setup_all_training_handlers
            
            # Initialize model services dengan progress callback
            ui_components = self._initialize_model_services(ui_components, config)
            
            # Setup handlers dengan progress integration
            ui_components = setup_all_training_handlers(ui_components, config, env)
            
            return ui_components
            
        except Exception as e:
            self.logger.error(f"❌ Error setting up handlers: {str(e)}")
            return ui_components
    
    def _initialize_model_services(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize model services dengan step-by-step progress"""
        try:
            # Progress update dengan steps
            self._update_initialization_progress(ui_components, 1, 5, "📋 Parsing configuration...")
            
            # Parse config
            training_config = config.get('training', {})
            model_type = training_config.get('model_type', 'efficient_optimized')
            
            # Step 2: Create model manager
            self._update_initialization_progress(ui_components, 2, 5, f"🧠 Creating {model_type} model...")
            model_manager = self._create_model_manager(training_config, model_type)
            
            # Step 3: Create services
            self._update_initialization_progress(ui_components, 3, 5, "🚀 Initializing backend services...")
            services = self._create_training_services(model_manager, config)
            
            # Step 4: Create training service manager
            self._update_initialization_progress(ui_components, 4, 5, "🔄 Initializing training manager...")
            training_manager = self._create_training_manager(ui_components, config)
            
            # Step 5: Complete
            self._update_initialization_progress(ui_components, 5, 5, "✅ Services ready!")
            
            # Update UI components
            ui_components.update({
                'model_manager': model_manager,
                'training_manager': training_manager,
                **services
            })
            
            self.logger.success(f"✅ Model services initialized: {model_type}")
            
        except Exception as e:
            self.logger.error(f"❌ Model services error: {str(e)}")
            
        return ui_components
    
    def _create_model_manager(self, training_config: Dict[str, Any], model_type: str):
        """Create model manager dengan proper config"""
        try:
            from smartcash.model.manager import ModelManager
            
            # Dapatkan parameter konfigurasi
            layer_mode = training_config.get('layer_mode', 'single')
            detection_layers = training_config.get('detection_layers', ['banknote'])
            
            # Create model manager dengan training config
            model_manager = ModelManager(
                config=training_config,
                model_type=model_type,
                layer_mode=layer_mode,
                detection_layers=detection_layers
            )
            
            self.logger.info(f"🧠 Model manager dibuat: {model_type}, layer_mode={layer_mode}")
            return model_manager
            
        except Exception as e:
            self.logger.error(f"❌ Model manager error: {str(e)}")
            return None
    
    def _create_training_services(self, model_manager, config: Dict[str, Any]):
        """Create training services dengan progress callback integration"""
        try:
            # Gunakan training service dari model manager
            backend_training_service = model_manager.get_training_service()
            
            # Buat adapter untuk training service
            from smartcash.ui.training.adapters import TrainingServiceAdapter
            training_service = TrainingServiceAdapter(backend_training_service, self.logger)
            
            # Dapatkan checkpoint service dari training service backend
            checkpoint_service = backend_training_service.checkpoint_service
            
            # Dapatkan metrics tracker dari training service backend
            metrics_tracker = backend_training_service.metrics_tracker
            
            self.logger.info(f"🚀 Training service adapter dibuat untuk model {model_manager.model_type}")
            
            # Return services map
            return {
                'training_service': training_service,
                'checkpoint_service': checkpoint_service,
                'metrics_tracker': metrics_tracker,
                'backend_training_service': backend_training_service
            }
            
        except Exception as e:
            self.logger.error(f"❌ Services error: {str(e)}")
            return {}
    
    def _create_training_manager(self, ui_components: Dict[str, Any], config: Dict[str, Any]):
        """Create training service manager untuk integrasi UI dan backend"""
        try:
            from smartcash.ui.training.services import TrainingServiceManager
            
            # Dapatkan model manager dan training service
            model_manager = ui_components.get('model_manager')
            training_service = ui_components.get('training_service')
            
            if not model_manager or not training_service:
                self.logger.warning("⚠️ Model manager atau training service tidak tersedia untuk training manager")
                return None
            
            # Buat training manager
            training_manager = TrainingServiceManager(ui_components, config, self.logger)
            
            # Register model manager dan config
            training_manager.register_model_manager(model_manager)
            training_manager.register_config(config)
            
            self.logger.info("✅ Training service manager berhasil dibuat")
            return training_manager
            
        except Exception as e:
            self.logger.error(f"❌ Error saat membuat training manager: {str(e)}")
            return None
    
    def _setup_config_callback(self, ui_components: Dict[str, Any]):
        """Setup callback untuk config updates dengan integrasi dari berbagai modul"""
        from smartcash.common.config.manager import SimpleConfigManager, get_config_manager
        from smartcash.ui.training.components.config_tabs import create_config_tabs, update_config_tabs
        
        def config_update_callback(new_config: Dict[str, Any]):
            """Handle config updates dan refresh info display"""
            try:
                # Gabungkan config dari berbagai modul
                merged_config = {}
                
                # Get config manager singleton
                config_manager = get_config_manager()
                
                # Model config dari backbone
                model_config = config_manager.get_config('backbone')
                if model_config:
                    merged_config['model'] = model_config
                
                # Training config (penting untuk UI training)
                training_config = config_manager.get_config('training')
                if training_config:
                    merged_config['training'] = training_config
                
                # Detector config
                detector_config = config_manager.get_config('detector')
                if detector_config:
                    merged_config['detector'] = detector_config
                
                # Hyperparameters config
                hyperparams_config = config_manager.get_config('hyperparameters')
                if hyperparams_config:
                    merged_config['hyperparameters'] = hyperparams_config
                
                # Training strategy config
                training_strategy_config = config_manager.get_config('training_strategy')
                if training_strategy_config:
                    merged_config['training_strategy'] = training_strategy_config
                
                # Paths config
                paths_config = config_manager.get_config('paths')
                if paths_config:
                    merged_config['paths'] = paths_config
                
                # Tambahkan config baru
                if new_config:
                    merged_config.update(new_config)
                
                # Update tabs jika sudah ada di ui_components
                if 'config_tabs' in ui_components:
                    update_config_tabs(ui_components['config_tabs'], merged_config)
                
                # Pastikan logger memanggil info
                if hasattr(self, 'logger') and self.logger:
                    self.logger.info("🔄 Configuration updated in training UI")
                elif 'logger' in ui_components and ui_components['logger']:
                    ui_components['logger'].info("🔄 Configuration updated in training UI")
                
            except Exception as e:
                # Catat error dan pastikan logger dipanggil
                error_msg = f"⚠️ Config callback error: {str(e)}"
                if hasattr(self, 'logger') and self.logger:
                    self.logger.warning(error_msg)
                elif 'logger' in ui_components and ui_components['logger']:
                    ui_components['logger'].warning(error_msg)
        
        # Register callback
        self.config_update_callbacks.append(config_update_callback)
        ui_components['config_update_callback'] = config_update_callback
    
    def _update_initialization_progress(self, ui_components: Dict[str, Any], current: int, total: int, message: str):
        """Update initialization progress step-by-step"""
        try:
            from smartcash.ui.training.utils.training_progress_utils import update_model_loading_progress
            update_model_loading_progress(ui_components, current, total, message)
        except Exception:
            # Silent fail untuk progress updates
            pass
    
    def _create_fallback_components(self, error_msg: str) -> Dict[str, Any]:
        """Create minimal fallback components"""
        from smartcash.ui.training.components.fallback_component import create_fallback_training_form
        return create_fallback_training_form(error_msg)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default training configuration"""
        return {
            'training': {
                'model_type': 'efficient_optimized',
                'backbone': 'efficientnet_b4',
                'epochs': 100,
                'batch_size': 16,
                'learning_rate': 0.001,
                'image_size': 640,
                'detection_layers': ['banknote'],
                'num_classes': 7
            },
            'paths': {
                'checkpoint_dir': 'runs/train/checkpoints'
            },
            'model_optimization': {
                'use_attention': True,
                'use_residual': True,
                'use_ciou': True
            }
        }
    
    def _get_critical_components(self) -> List[str]:
        """Critical components yang harus ada"""
        return ['main_container', 'start_button', 'stop_button', 'status_panel']
    
    def trigger_config_update(self, new_config: Dict[str, Any]):
        """Trigger config update callbacks"""
        [callback(new_config) for callback in self.config_update_callbacks]


def initialize_training_ui(env=None, config=None, **kwargs):
    """Factory function untuk training UI dengan pencegahan tampilan ganda"""
    suppress_all_outputs()
    
    try:
        # Check apakah sudah ada initializer yang aktif
        initializer = None
        for obj in globals().values():
            if isinstance(obj, TrainingInitializer) and hasattr(obj, '_ui_displayed') and obj._ui_displayed:
                initializer = obj
                break
        
        # Jika belum ada, buat yang baru
        if not initializer:
            initializer = TrainingInitializer('training', 'smartcash.ui.training')
        
        # Initialize UI
        result = initializer.initialize(env, config, **kwargs)
        
        # Set flag untuk mencegah tampilan ganda
        initializer._ui_displayed = True
        
        # Auto-display dengan single display check
        if not getattr(result, '_displayed', False):
            if hasattr(result, 'children') or hasattr(result, 'layout'):
                display(result)
                result._displayed = True
            elif isinstance(result, dict) and 'ui' in result:
                ui_widget = result['ui']
                if not getattr(ui_widget, '_displayed', False):
                    display(ui_widget)
                    ui_widget._displayed = True
            elif isinstance(result, dict) and 'main_container' in result:
                main_widget = result['main_container']
                if not getattr(main_widget, '_displayed', False):
                    display(main_widget)
                    main_widget._displayed = True
        
        return result
        
    except Exception as e:
        from smartcash.ui.utils.fallback_utils import create_fallback_ui
        fallback = create_fallback_ui(f"Training init error: {str(e)}", 'training')
        display(fallback.get('ui', fallback))
        return fallback


# Global instance dan public API
_training_initializer = TrainingInitializer()

# One-liner factory untuk compatibility dengan cell entry
create_training_ui = lambda env=None, config=None, **kw: initialize_training_ui(env, config, **kw)
get_training_initializer = lambda: _training_initializer