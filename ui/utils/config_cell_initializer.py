"""
File: smartcash/ui/utils/config_cell_initializer.py
Deskripsi: Simplified base class untuk config cells dengan aggressive log suppression
"""

from typing import Dict, Any, Optional, Callable
from abc import ABC, abstractmethod
import ipywidgets as widgets
from IPython.display import display
import logging
import sys
import warnings

from smartcash.common.config.manager import get_config_manager
from smartcash.common.logger import get_logger


class ConfigCellInitializer(ABC):
    """Simplified base class untuk config cells dengan focus pada config operations"""
    
    def __init__(self, module_name: str, config_filename: str):
        self.module_name = module_name
        self.config_filename = config_filename
        self.logger = get_logger(f"smartcash.ui.{module_name}")
        self.config_callbacks = []
    
    def initialize(self, env=None, config=None, **kwargs) -> Any:
        """Main initialization dengan aggressive suppression"""
        try:
            self._suppress_all_logs()
            config = self._load_config(config)
            ui_components = self._create_config_ui(config, env, **kwargs)
            
            if not self._validate_ui(ui_components):
                return self._error_ui("❌ Required components missing")
            
            self._setup_handlers(ui_components, config)
            self._update_status(ui_components, f"✅ {self.module_name} ready", "success")
            return ui_components.get('main_container', ui_components)
        except Exception as e:
            return self._error_ui(f"❌ Error: {str(e)}")
    
    # Abstract methods
    @abstractmethod
    def _create_config_ui(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Buat komponen UI untuk config"""
        pass
    
    @abstractmethod
    def _extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dari UI components"""
        pass
    
    @abstractmethod
    def _update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dari config"""
        pass
    
    @abstractmethod
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default config"""
        pass
    
    def _load_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Load config dari file atau default"""
        if config:
            return config
        try:
            saved = get_config_manager().get_config(self.config_filename)
            return saved if saved else self._get_default_config()
        except:
            return self._get_default_config()
    
    def _validate_ui(self, ui_components: Dict[str, Any]) -> bool:
        """Validate required UI components ada"""
        required = ['save_button', 'reset_button', 'status_panel']
        return all(comp in ui_components for comp in required)
    
    def _setup_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Setup save/reset handlers"""
        ui_components.update({'module_name': self.module_name, 'config': config})
        
        # Setup button handlers
        save_btn = ui_components.get('save_button')
        if save_btn and hasattr(save_btn, 'on_click'):
            save_btn.on_click(lambda b: self._save_config(ui_components, b))
        
        reset_btn = ui_components.get('reset_button')
        if reset_btn and hasattr(reset_btn, 'on_click'):
            reset_btn.on_click(lambda b: self._reset_config(ui_components, b))
        
        # Setup custom handlers jika ada
        custom_handler = getattr(self, '_setup_custom_handlers', None)
        if custom_handler:
            custom_handler(ui_components, config)
    
    def _save_config(self, ui_components: Dict[str, Any], button) -> None:
        """Save config handler"""
        try:
            button.disabled = True
            self._update_status(ui_components, "💾 Saving...", "info")
            
            config = self._extract_config(ui_components)
            success = get_config_manager().save_config(config, self.config_filename)
            
            if success:
                ui_components['config'] = config
                self._update_status(ui_components, "✅ Saved", "success")
                [cb(config) for cb in self.config_callbacks]  # Notify callbacks
            else:
                self._update_status(ui_components, "❌ Save failed", "error")
        except Exception as e:
            self._update_status(ui_components, f"❌ Error: {str(e)}", "error")
        finally:
            button.disabled = False
    
    def _reset_config(self, ui_components: Dict[str, Any], button) -> None:
        """Reset config handler"""
        try:
            button.disabled = True
            self._update_status(ui_components, "🔄 Resetting...", "info")
            
            default_config = self._get_default_config()
            self._update_ui(ui_components, default_config)
            get_config_manager().save_config(default_config, self.config_filename)
            
            ui_components['config'] = default_config
            self._update_status(ui_components, "✅ Reset complete", "success")
            [cb(default_config) for cb in self.config_callbacks]  # Notify callbacks
        except Exception as e:
            self._update_status(ui_components, f"❌ Error: {str(e)}", "error")
        finally:
            button.disabled = False
    
    def _update_status(self, ui_components: Dict[str, Any], message: str, status_type: str = "info") -> None:
        """Update status panel"""
        try:
            from smartcash.ui.components.status_panel import update_status_panel
            update_status_panel(ui_components.get('status_panel'), message, status_type)
        except:
            pass  # Silent fail
    
    def _suppress_all_logs(self) -> None:
        """Aggressive log suppression untuk clean config UI"""
        # Clear root logger
        root = logging.getLogger()
        root.handlers.clear()
        root.setLevel(logging.CRITICAL)
        root.propagate = False
        
        # Suppress 50+ common targets dengan one-liner
        targets = [
            'requests', 'urllib3', 'tensorflow', 'torch', 'sklearn', 'ipywidgets',
            'google', 'yaml', 'tqdm', 'matplotlib', 'pandas', 'numpy', 'PIL',
            'smartcash.dataset', 'smartcash.model', 'smartcash.training',
            'smartcash.common', 'smartcash.ui.dataset', 'smartcash.detection',
            'IPython', 'traitlets', 'tornado', 'seaborn', 'cv2', 'pathlib'
        ]
        
        [setattr(logging.getLogger(t), 'level', logging.CRITICAL) or 
         setattr(logging.getLogger(t), 'propagate', False) or
         logging.getLogger(t).handlers.clear() for t in targets]
        
        # Suppress stdout dengan simple anonymous class
        if not hasattr(sys, '_original_stdout_saved'):
            sys._original_stdout_saved = sys.stdout
            sys.stdout = type('', (), {
                'write': lambda self, x: None,
                'flush': lambda self: None,
                'isatty': lambda self: False,
                'fileno': lambda self: sys._original_stdout_saved.fileno()
            })()
        
        # Suppress stderr
        if not hasattr(sys, '_original_stderr_saved'):
            sys._original_stderr_saved = sys.stderr
            sys.stderr = type('', (), {
                'write': lambda self, x: None,
                'flush': lambda self: None,
                'isatty': lambda self: False,
                'fileno': lambda self: sys._original_stderr_saved.fileno()
            })()
        
        # Suppress warnings
        warnings.filterwarnings('ignore')
    
    def _error_ui(self, message: str):
        """Create error UI dengan simple styling"""
        return widgets.HTML(f"""
        <div style="padding:15px; background:#f8d7da; border:1px solid #dc3545; 
                    border-radius:5px; color:#721c24; margin:10px 0;">
            <h4>⚠️ {self.module_name} Error</h4>
            <p>{message}</p>
            <small>💡 Try restarting cell atau check config</small>
        </div>""")
    
    # Callback system dengan one-liner methods
    add_callback = lambda self, cb: self.config_callbacks.append(cb) if cb not in self.config_callbacks else None
    remove_callback = lambda self, cb: self.config_callbacks.remove(cb) if cb in self.config_callbacks else None


def create_config_cell(initializer_class, module_name: str, config_filename: str, 
                      env=None, config=None, **kwargs) -> Any:
    """Factory untuk create config cell"""
    try:
        initializer = initializer_class(module_name, config_filename)
        return initializer.initialize(env, config, **kwargs)
    except Exception as e:
        return widgets.HTML(f"<div style='color:red'>❌ {str(e)}</div>")


def restore_stdout():
    """Utility untuk restore stdout jika diperlukan"""
    if hasattr(sys, '_original_stdout_saved'):
        sys.stdout = sys._original_stdout_saved
        delattr(sys, '_original_stdout_saved')
    if hasattr(sys, '_original_stderr_saved'):
        sys.stderr = sys._original_stderr_saved
        delattr(sys, '_original_stderr_saved')