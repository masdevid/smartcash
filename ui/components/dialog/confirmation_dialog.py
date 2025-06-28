"""
File: smartcash/ui/components/dialog/confirmation_dialog.py
Deskripsi: Komponen dialog modern dengan glass morphism yang responsive dan optimized untuk show/hide behavior
"""

from typing import Dict, Any, Callable, Optional, Tuple
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

# Modern glass morphism CSS dengan optimasi responsif
GLASS_MORPHISM_CSS = """
<style>
.smartcash-dialog-container {
    position: relative;
    max-width: 1200px;
    width: 100%;
    margin: 0 auto;
    padding: 0;
    z-index: 1000;
}

.glass-confirmation-card {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.15);
    box-shadow: 
        0 8px 32px 0 rgba(31, 38, 135, 0.12),
        inset 0 1px 0 rgba(255, 255, 255, 0.3);
    padding: 24px;
    max-width: 100%;
    max-height: 200px;
    overflow: hidden;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
}

.glass-confirmation-card.danger {
    background: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(239, 68, 68, 0.2);
    box-shadow: 
        0 8px 32px 0 rgba(239, 68, 68, 0.12),
        inset 0 1px 0 rgba(255, 255, 255, 0.3);
}

.dialog-header {
    color: #1a202c;
    font-size: 1.2rem;
    font-weight: 600;
    margin: 0 0 8px 0;
    line-height: 1.4;
}

.dialog-message {
    color: #4a5568;
    font-size: 0.95rem;
    line-height: 1.5;
    margin: 0 0 20px 0;
    white-space: pre-line;
    max-height: 60px;
    overflow-y: auto;
}

.dialog-actions {
    display: flex;
    gap: 12px;
    justify-content: center;
    align-items: center;
    flex-wrap: wrap;
}

.glass-btn {
    background: rgba(255, 255, 255, 0.2);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.3);
    border-radius: 10px;
    padding: 8px 20px;
    font-size: 0.9rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.25s ease;
    min-width: 90px;
    text-align: center;
    user-select: none;
    outline: none;
}

.glass-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
    background: rgba(255, 255, 255, 0.3);
}

.glass-btn:active {
    transform: translateY(0);
}

.glass-btn.primary {
    background: rgba(79, 70, 229, 0.8);
    color: white;
    border-color: rgba(79, 70, 229, 0.9);
}

.glass-btn.primary:hover {
    background: rgba(79, 70, 229, 0.9);
    box-shadow: 0 4px 20px rgba(79, 70, 229, 0.3);
}

.glass-btn.danger {
    background: rgba(239, 68, 68, 0.8);
    color: white;
    border-color: rgba(239, 68, 68, 0.9);
}

.glass-btn.danger:hover {
    background: rgba(239, 68, 68, 0.9);
    box-shadow: 0 4px 20px rgba(239, 68, 68, 0.3);
}

.glass-btn.secondary {
    background: rgba(107, 114, 128, 0.1);
    color: #374151;
    border-color: rgba(107, 114, 128, 0.3);
}

.glass-btn.secondary:hover {
    background: rgba(107, 114, 128, 0.2);
    color: #1f2937;
}

@keyframes slideInDown {
    from {
        opacity: 0;
        transform: translateY(-20px) scale(0.95);
    }
    to {
        opacity: 1;
        transform: translateY(0) scale(1);
    }
}

@keyframes slideOutUp {
    from {
        opacity: 1;
        transform: translateY(0) scale(1);
    }
    to {
        opacity: 0;
        transform: translateY(-20px) scale(0.95);
    }
}

.dialog-show {
    animation: slideInDown 0.3s cubic-bezier(0.4, 0, 0.2, 1) forwards;
}

.dialog-hide {
    animation: slideOutUp 0.25s cubic-bezier(0.4, 0, 0.2, 1) forwards;
}

/* Responsive design */
@media (max-width: 768px) {
    .smartcash-dialog-container {
        max-width: 95%;
        padding: 0 10px;
    }
    
    .glass-confirmation-card {
        padding: 20px 16px;
        border-radius: 16px;
    }
    
    .dialog-header {
        font-size: 1.1rem;
    }
    
    .dialog-message {
        font-size: 0.9rem;
    }
    
    .glass-btn {
        padding: 8px 16px;
        font-size: 0.85rem;
        min-width: 80px;
    }
    
    .dialog-actions {
        gap: 10px;
    }
}

@media (max-width: 480px) {
    .glass-confirmation-card {
        padding: 16px 12px;
        max-height: 180px;
    }
    
    .dialog-actions {
        flex-direction: column;
        gap: 8px;
        width: 100%;
    }
    
    .glass-btn {
        width: 100%;
        max-width: 200px;
    }
}

/* Hide scrollbars but keep functionality */
.dialog-message::-webkit-scrollbar {
    width: 3px;
}

.dialog-message::-webkit-scrollbar-track {
    background: transparent;
}

.dialog-message::-webkit-scrollbar-thumb {
    background: rgba(0, 0, 0, 0.2);
    border-radius: 2px;
}

.dialog-message::-webkit-scrollbar-thumb:hover {
    background: rgba(0, 0, 0, 0.3);
}
</style>
"""

class GlassDialogManager:
    """Manager untuk glass morphism dialog dengan state management yang robust"""
    
    def __init__(self):
        self.current_dialog = None
        self.is_visible = False
        self._ensure_css_loaded()
    
    def _ensure_css_loaded(self):
        """Ensure CSS is loaded hanya sekali"""
        if not hasattr(self, '_css_loaded'):
            display(HTML(GLASS_MORPHISM_CSS))
            self._css_loaded = True
    
    def create_confirmation_area(self, ui_components: Dict[str, Any]) -> widgets.Output:
        """Buat area konfirmasi yang optimized untuk show/hide behavior"""
        
        if 'confirmation_area' not in ui_components:
            confirmation_area = widgets.Output(
                layout=widgets.Layout(
                    width='100%',
                    min_height='0px',
                    max_height='250px',
                    margin='10px 0',
                    padding='0',
                    overflow='visible',
                    display='none'  # Initially hidden
                )
            )
            ui_components['confirmation_area'] = confirmation_area
        
        return ui_components['confirmation_area']
    
    def show_dialog(self, 
                   ui_components: Dict[str, Any],
                   title: str,
                   message: str,
                   on_confirm: Optional[Callable] = None,
                   on_cancel: Optional[Callable] = None,
                   confirm_text: str = "Konfirmasi",
                   cancel_text: str = "Batal",
                   danger_mode: bool = False) -> None:
        """Show glass morphism dialog dengan improved state management"""
        
        try:
            # Ensure confirmation area exists
            confirmation_area = self.create_confirmation_area(ui_components)
            
            # Force clear dan reset layout
            self._reset_dialog_area(confirmation_area)
            
            # Setup button handlers dengan proper cleanup
            def handle_confirm():
                self._handle_button_click(confirmation_area, on_confirm)
            
            def handle_cancel():
                self._handle_button_click(confirmation_area, on_cancel)
            
            # Generate unique button IDs untuk avoid conflicts
            import uuid
            confirm_id = f"confirm_btn_{uuid.uuid4().hex[:8]}"
            cancel_id = f"cancel_btn_{uuid.uuid4().hex[:8]}"
            
            # Determine button classes
            confirm_class = "glass-btn danger" if danger_mode else "glass-btn primary"
            card_class = "glass-confirmation-card danger" if danger_mode else "glass-confirmation-card"
            
            # Build HTML dengan improved structure
            dialog_html = f"""
            <div class="smartcash-dialog-container">
                <div class="{card_class} dialog-show">
                    <div class="dialog-header">{title}</div>
                    <div class="dialog-message">{message}</div>
                    <div class="dialog-actions">
                        <button id="{confirm_id}" class="{confirm_class}">
                            {confirm_text}
                        </button>
                        <button id="{cancel_id}" class="glass-btn secondary">
                            {cancel_text}
                        </button>
                    </div>
                </div>
            </div>
            
            <script>
            (function() {{
                // Setup event listeners dengan error handling
                const confirmBtn = document.getElementById('{confirm_id}');
                const cancelBtn = document.getElementById('{cancel_id}');
                
                if (confirmBtn) {{
                    confirmBtn.onclick = function() {{
                        try {{
                            // Add hide animation
                            const card = confirmBtn.closest('.glass-confirmation-card');
                            if (card) {{
                                card.classList.remove('dialog-show');
                                card.classList.add('dialog-hide');
                            }}
                            
                            // Trigger Python callback setelah animation
                            setTimeout(() => {{
                                window.dispatchEvent(new CustomEvent('smartcash_confirm_clicked'));
                            }}, 200);
                        }} catch (e) {{
                            console.error('Error in confirm handler:', e);
                            window.dispatchEvent(new CustomEvent('smartcash_confirm_clicked'));
                        }}
                    }};
                }}
                
                if (cancelBtn) {{
                    cancelBtn.onclick = function() {{
                        try {{
                            // Add hide animation
                            const card = cancelBtn.closest('.glass-confirmation-card');
                            if (card) {{
                                card.classList.remove('dialog-show');
                                card.classList.add('dialog-hide');
                            }}
                            
                            // Trigger Python callback setelah animation
                            setTimeout(() => {{
                                window.dispatchEvent(new CustomEvent('smartcash_cancel_clicked'));
                            }}, 200);
                        }} catch (e) {{
                            console.error('Error in cancel handler:', e);
                            window.dispatchEvent(new CustomEvent('smartcash_cancel_clicked'));
                        }}
                    }};
                }}
            }})();
            </script>
            """
            
            # Display dialog dengan proper timing
            with confirmation_area:
                clear_output(wait=True)
                display(HTML(dialog_html))
            
            # Show confirmation area dengan improved visibility
            self._show_confirmation_area(confirmation_area)
            
            # Store handlers untuk cleanup
            self.current_handlers = {
                'confirm': handle_confirm,
                'cancel': handle_cancel
            }
            self.is_visible = True
            
            # Setup JavaScript event listeners
            self._setup_js_listeners(handle_confirm, handle_cancel)
            
        except Exception as e:
            print(f"⚠️ Error showing dialog: {str(e)}")
            # Fallback ke simple print confirmation
            response = input(f"{title}: {message} (y/N): ").lower().strip()
            if response in ['y', 'yes', 'ya'] and on_confirm:
                on_confirm()
            elif on_cancel:
                on_cancel()
    
    def _reset_dialog_area(self, dialog_area: widgets.Output) -> None:
        """Reset dialog area dengan proper state cleanup"""
        if hasattr(dialog_area, 'layout'):
            dialog_area.layout.display = 'block'
            dialog_area.layout.visibility = 'visible'
            dialog_area.layout.height = 'auto'
            dialog_area.layout.overflow = 'visible'
        
        self.is_visible = False
        self.current_handlers = None
    
    def _show_confirmation_area(self, confirmation_area: widgets.Output) -> None:
        """Show confirmation area dengan proper layout management"""
        if hasattr(confirmation_area, 'layout'):
            confirmation_area.layout.display = 'block'
            confirmation_area.layout.visibility = 'visible'
            confirmation_area.layout.height = 'auto'
            confirmation_area.layout.max_height = '250px'
            confirmation_area.layout.overflow = 'visible'
    
    def _handle_button_click(self, dialog_area: widgets.Output, callback: Optional[Callable]) -> None:
        """Handle button click dengan proper cleanup"""
        try:
            # Hide dialog area
            self.hide_dialog(dialog_area)
            
            # Execute callback jika ada
            if callback:
                callback()
                
        except Exception as e:
            print(f"⚠️ Error in button handler: {str(e)}")
    
    def _setup_js_listeners(self, confirm_handler: Callable, cancel_handler: Callable):
        """Setup JavaScript event listeners untuk dialog buttons"""
        try:
            from IPython.display import Javascript
            
            js_code = f"""
            // Clean up existing listeners
            window.removeEventListener('smartcash_confirm_clicked', window.smartcash_confirm_handler);
            window.removeEventListener('smartcash_cancel_clicked', window.smartcash_cancel_handler);
            
            // Setup new listeners
            window.smartcash_confirm_handler = function() {{
                // Trigger Python callback
                if (window.jupyter && window.jupyter.notebook) {{
                    window.jupyter.notebook.kernel.execute('_handle_confirm_from_js()');
                }}
            }};
            
            window.smartcash_cancel_handler = function() {{
                // Trigger Python callback  
                if (window.jupyter && window.jupyter.notebook) {{
                    window.jupyter.notebook.kernel.execute('_handle_cancel_from_js()');
                }}
            }};
            
            window.addEventListener('smartcash_confirm_clicked', window.smartcash_confirm_handler);
            window.addEventListener('smartcash_cancel_clicked', window.smartcash_cancel_handler);
            """
            
            display(Javascript(js_code))
            
            # Store handlers in global namespace untuk JavaScript access
            import __main__
            __main__._handle_confirm_from_js = confirm_handler
            __main__._handle_cancel_from_js = cancel_handler
            
        except Exception as e:
            print(f"⚠️ Warning: JS listeners setup failed: {str(e)}")
    
    def hide_dialog(self, dialog_area: widgets.Output) -> None:
        """Hide dialog dengan proper cleanup"""
        try:
            if hasattr(dialog_area, 'layout'):
                dialog_area.layout.display = 'none'
                dialog_area.layout.visibility = 'hidden'
            
            # Clear content
            with dialog_area:
                clear_output(wait=True)
            
            self.is_visible = False
            self.current_handlers = None
            
        except Exception as e:
            print(f"⚠️ Error hiding dialog: {str(e)}")
    
    def clear_dialog_area(self, ui_components: Dict[str, Any]) -> None:
        """Clear dialog area dengan improved cleanup"""
        confirmation_area = ui_components.get('confirmation_area')
        if confirmation_area:
            self.hide_dialog(confirmation_area)

# Global dialog manager instance
dialog_manager = GlassDialogManager()

def show_confirmation_dialog(ui_components: Dict[str, Any],
                           title: str,
                           message: str,
                           on_confirm: Optional[Callable] = None,
                           on_cancel: Optional[Callable] = None,
                           confirm_text: str = "Konfirmasi",
                           cancel_text: str = "Batal",
                           danger_mode: bool = False) -> None:
    """Show glass morphism confirmation dialog dengan optimized behavior"""
    dialog_manager.show_dialog(
        ui_components=ui_components,
        title=title,
        message=message,
        on_confirm=on_confirm,
        on_cancel=on_cancel,
        confirm_text=confirm_text,
        cancel_text=cancel_text,
        danger_mode=danger_mode
    )

def show_info_dialog(ui_components: Dict[str, Any],
                    title: str,
                    message: str,
                    on_ok: Optional[Callable] = None,
                    ok_text: str = "OK") -> None:
    """Show glass morphism info dialog"""
    dialog_manager.show_dialog(
        ui_components=ui_components,
        title=title,
        message=message,
        on_confirm=on_ok,
        on_cancel=None,
        confirm_text=ok_text,
        cancel_text="",
        danger_mode=False
    )

def clear_dialog_area(ui_components: Dict[str, Any]) -> None:
    """Clear dialog area menggunakan dialog manager"""
    dialog_manager.clear_dialog_area(ui_components)

def is_dialog_visible(ui_components: Dict[str, Any]) -> bool:
    """Check apakah dialog sedang visible"""
    return dialog_manager.is_visible

def create_confirmation_area(ui_components: Dict[str, Any]) -> widgets.Output:
    """Create atau get existing confirmation area"""
    return dialog_manager.create_confirmation_area(ui_components)