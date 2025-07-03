"""
Confirmation dialog component for displaying modal dialogs with confirmation actions.
"""

import uuid
from typing import Dict, Any, Callable, Optional
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

from smartcash.ui.components.base_component import BaseUIComponent


class ConfirmationDialog(BaseUIComponent):
    """A confirmation dialog component with glass morphism styling."""
    
    def __init__(self, 
                 component_name: str = "confirmation_dialog",
                 **kwargs):
        """Initialize the confirmation dialog.
        
        Args:
            component_name: Unique name for this component
            **kwargs: Additional arguments to pass to BaseUIComponent
        """
        self._callbacks: Dict[str, Callable] = {}
        self._is_visible = False
        
        # Initialize base class
        super().__init__(component_name, **kwargs)
    
    def _create_ui_components(self) -> None:
        """Create and initialize UI components."""
        # Create main container
        self._ui_components['container'] = widgets.Output(
            layout=widgets.Layout(
                width='100%',
                max_width='100%',
                min_height='50px',
                max_height='500px',
                margin='10px 0',
                padding='10px',
                border='1px solid #e0e0e0',
                border_radius='4px',
                overflow_y='auto',
                overflow_x='hidden',
                display='none',  # Initially hidden
                flex='1 1 auto'  # Allow expansion
            )
        )
        
        # Add class for styling
        if hasattr(self._ui_components['container'], 'add_class'):
            self._ui_components['container'].add_class('confirmation-area')
    
    def show(self, 
             title: str,
             message: str,
             on_confirm: Optional[Callable] = None,
             on_cancel: Optional[Callable] = None,
             confirm_text: str = "Konfirmasi",
             cancel_text: str = "Batal",
             danger_mode: bool = False) -> None:
        """Show the confirmation dialog.
        
        Args:
            title: Dialog title
            message: Dialog message
            on_confirm: Callback for confirm action
            on_cancel: Callback for cancel action
            confirm_text: Text for confirm button
            cancel_text: Text for cancel button
            danger_mode: Whether to use danger styling
        """
        if not self._initialized:
            self.initialize()
        
        # Store callbacks
        self._callbacks['confirm'] = on_confirm
        self._callbacks['cancel'] = on_cancel
        
        # Generate unique IDs
        dialog_id = f"dialog_{uuid.uuid4().hex[:8]}"
        confirm_id = f"confirm_{uuid.uuid4().hex[:8]}"
        cancel_id = f"cancel_{uuid.uuid4().hex[:8]}"
        
        # Define colors based on mode
        if danger_mode:
            confirm_bg = "#ef4444"
            confirm_hover = "#dc2626"
            card_accent = "rgba(239, 68, 68, 0.1)"
            card_border = "rgba(239, 68, 68, 0.2)"
        else:
            confirm_bg = "#4f46e5"
            confirm_hover = "#4338ca"
            card_accent = "rgba(79, 70, 229, 0.05)"
            card_border = "rgba(79, 70, 229, 0.15)"
        
        # Build HTML with inline styles
        dialog_html = f"""
        <style>
        .smartcash-dialog-{dialog_id} {{
            position: relative;
            max-width: 100%;
            width: 100%;
            margin: 0 auto;
            padding: 0;
            overflow: hidden;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            height: 100%;
            display: flex;
            flex-direction: column;
        }}
        
        .glass-card-{dialog_id} {{
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05));
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-radius: 12px;
            border: 1px solid {card_border};
            box-shadow: 
                0 4px 16px 0 rgba(31, 38, 135, 0.1),
                inset 0 1px 0 rgba(255, 255, 255, 0.2);
            padding: 28px 24px;
            width: 100%;
            max-width: 420px;
            max-height: 90vh;
            overflow-y: auto;
            overflow-x: hidden;
            transition: all 0.3s ease;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            position: relative;
            box-sizing: border-box;
            margin: 20px auto;
        }}
        
        .glass-card-{dialog_id}::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: {card_accent};
            border-radius: 8px;
            z-index: -1;
        }}
        
        .dialog-header-{dialog_id} {{
            color: #1a202c;
            font-size: 1.1rem;
            font-weight: 600;
            line-height: 1.4;
            margin: 0 0 8px 0;
            padding: 0 8px;
            width: 100%;
        }}
        
        .dialog-message-{dialog_id} {{
            color: #4b5563;
            font-size: 0.95rem;
            line-height: 1.5;
            margin: 12px 0 24px 0;
            padding: 0 8px;
            width: 100%;
            max-height: 50vh;
            overflow-y: auto;
            white-space: pre-line;
            box-sizing: border-box;
        }}
        
        .dialog-actions-{dialog_id} {{
            display: flex;
            gap: 12px;
            justify-content: center;
            align-items: center;
            flex-wrap: nowrap;
            width: 100%;
            margin-top: 8px;
            padding-top: 8px;
            border-top: 1px solid rgba(0, 0, 0, 0.05);
        }}
        
        .glass-btn-{dialog_id} {{
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 0.9rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
            min-width: 100px;
            text-align: center;
            user-select: none;
            outline: none;
            position: relative;
            overflow: hidden;
            flex: 1;
            max-width: 160px;
        }}
        
        .glass-btn-{dialog_id}:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        }}
        
        .glass-btn-{dialog_id}:active {{
            transform: translateY(-1px);
        }}
        
        .confirm-btn-{dialog_id} {{
            background: {confirm_bg};
            color: white;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }}
        
        .confirm-btn-{dialog_id}:hover {{
            background: {confirm_hover};
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.25);
        }}
        
        .cancel-btn-{dialog_id} {{
            background: rgba(107, 114, 128, 0.1);
            color: #374151;
            border: 1px solid rgba(107, 114, 128, 0.3);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }}
        
        .cancel-btn-{dialog_id}:hover {{
            background: rgba(107, 114, 128, 0.2);
            color: #1f2937;
        }}
        
        /* Responsive design */
        @media (max-width: 768px) {{
            .smartcash-dialog-{dialog_id} {{
                max-width: 95%;
                padding: 0 10px;
            }}
            
            .glass-card-{dialog_id} {{
                padding: 24px 20px;
                border-radius: 12px;
                margin: 10px auto;
            }}
            
            .dialog-header-{dialog_id} {{
                font-size: 1.1rem;
                margin-bottom: 12px;
            }}
            
            .dialog-message-{dialog_id} {{
                font-size: 0.95rem;
                margin: 12px 0 20px 0;
            }}
            
            .glass-btn-{dialog_id} {{
                padding: 10px 16px;
                font-size: 0.9rem;
                min-width: 90px;
            }}
            
            .dialog-actions-{dialog_id} {{
                gap: 10px;
                padding-top: 12px;
                margin-top: 12px;
            }}
        }}
        
        @media (max-width: 480px) {{
            .glass-card-{dialog_id} {{
                padding: 20px 16px;
                margin: 10px;
                max-height: 90vh;
            }}
            
            .dialog-actions-{dialog_id} {{
                flex-direction: row;
                justify-content: space-between;
                gap: 10px;
                width: 100%;
                margin-top: 12px;
                padding-top: 12px;
            }}
            
            .glass-btn-{dialog_id} {{
                flex: 1;
                min-width: auto;
                padding: 10px 12px;
                font-size: 0.85rem;
            }}
            
            .dialog-message-{dialog_id} {{
                margin: 12px 0 20px 0;
                font-size: 0.9rem;
                padding: 0 4px;
            }}
        }}
        
        /* Animation */
        @keyframes slideIn-{dialog_id} {{
            from {{
                opacity: 0;
                transform: translateY(-20px) scale(0.95);
            }}
            to {{
                opacity: 1;
                transform: translateY(0) scale(1);
            }}
        }}
        
        .glass-card-{dialog_id} {{
            animation: slideIn-{dialog_id} 0.3s cubic-bezier(0.4, 0, 0.2, 1) forwards;
        }}
        </style>
        
        <div class="smartcash-dialog-{dialog_id}">
            <div class="glass-card-{dialog_id}">
                <div class="dialog-header-{dialog_id}">{title}</div>
                <div class="dialog-message-{dialog_id}">{message}</div>
                <div class="dialog-actions-{dialog_id}">
                    <button id="{confirm_id}" class="glass-btn-{dialog_id} confirm-btn-{dialog_id}">
                        {confirm_text}
                    </button>
                    {f'<button id="{cancel_id}" class="glass-btn-{dialog_id} cancel-btn-{dialog_id}">{cancel_text}</button>' if cancel_text else ''}
                </div>
            </div>
        </div>
        
        <script>
        (function() {{
            const confirmBtn = document.getElementById('{confirm_id}');
            const cancelBtn = document.getElementById('{cancel_id}');
            
            function handleAction(actionType) {{
                // Add fade out animation
                const card = document.querySelector('.glass-card-{dialog_id}');
                if (card) {{
                    card.style.animation = 'none';
                    card.style.transition = 'opacity 0.2s ease';
                    card.style.opacity = '0';
                    card.style.transform = 'scale(0.95)';
                }}
                
                setTimeout(() => {{
                    // Hide the dialog
                    const container = document.querySelector('.confirmation-area');
                    if (container) {{
                        container.style.transition = 'all 0.3s ease';
                        container.style.height = '0';
                        container.style.minHeight = '0';
                        container.style.maxHeight = '0';
                        container.style.padding = '0';
                        container.style.margin = '0';
                        container.style.overflow = 'hidden';
                        container.style.border = 'none';
                    }}
                    
                    // Execute the appropriate callback
                    if (window.jupyter && window.jupyter.notebook && window.jupyter.notebook.kernel) {{
                        const code = `
from IPython.display import clear_output
from smartcash.ui.components.dialog.confirmation_dialog import clear_dialog_area

# Clear the output
if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'clear_output'):
    with ui_components['confirmation_area']:
        clear_output(wait=True)

# Execute the callback
if '${action_type}' in ui_components._callbacks and ui_components._callbacks['${action_type}']:
    try:
        ui_components._callbacks['${action_type}']()
    except Exception as e:
        print(f"⚠️ Error in ${action_type} callback: {str(e) if 'e' in locals() else 'Unknown error'}")

# Clear callbacks
ui_components._callbacks = {{}}
`;
                        window.jupyter.notebook.kernel.execute(code);
                    }}
                }}, 200);
            }}
            
            if (confirmBtn) {{
                confirmBtn.onclick = () => handleAction('confirm');
            }}
            
            if (cancelBtn) {{
                cancelBtn.onclick = () => handleAction('cancel');
            }}
        }})();
        </script>
        """.format(
            dialog_id=dialog_id,
            confirm_id=confirm_id,
            cancel_id=cancel_id,
            title=title,
            message=message,
            confirm_text=confirm_text,
            cancel_text=cancel_text,
            card_border=card_border,
            card_accent=card_accent,
            confirm_bg=confirm_bg,
            confirm_hover=confirm_hover
        )
        
        # Show the dialog
        self._show_dialog(dialog_html)
    
    def show_info(self,
                 title: str,
                 message: str,
                 on_ok: Optional[Callable] = None,
                 ok_text: str = "OK") -> None:
        """Show an info dialog with a single OK button.
        
        Args:
            title: Dialog title
            message: Dialog message
            on_ok: Callback for OK button
            ok_text: Text for OK button
        """
        self.show(
            title=title,
            message=message,
            on_confirm=on_ok,
            on_cancel=None,
            confirm_text=ok_text,
            cancel_text="",
            danger_mode=False
        )
    
    def _show_dialog(self, html_content: str) -> None:
        """Display the dialog with the given HTML content."""
        if not self._initialized:
            self.initialize()
        
        container = self._ui_components['container']
        
        # Update container styles
        container.layout.display = 'flex'
        container.layout.visibility = 'visible'
        container.layout.height = 'auto'
        container.layout.min_height = '200px'
        container.layout.max_height = '300px'
        container.layout.padding = '10px 15px'
        container.layout.margin = '5px 0'
        container.layout.overflow = 'hidden'
        container.layout.flex = '0 0 auto'
        
        # Display the dialog
        with container:
            clear_output(wait=True)
            display(HTML(html_content))
        
        self._is_visible = True
    
    def hide(self) -> None:
        """Hide the dialog."""
        if not self._initialized:
            return
        
        container = self._ui_components.get('container')
        if container:
            # Clear the output
            with container:
                clear_output(wait=True)
            
            # Reset container styles
            container.layout.display = 'none'
            container.layout.visibility = 'hidden'
            container.layout.height = None
            container.layout.min_height = '50px'
            container.layout.max_height = '500px'
            container.layout.padding = '10px'
            container.layout.margin = '10px 0'
            container.layout.overflow = 'auto'
            container.layout.border = '1px solid #e0e0e0'
            container.layout.border_radius = '4px'
        
        self._is_visible = False
        self._callbacks = {}
    
    def is_visible(self) -> bool:
        """Check if the dialog is currently visible."""
        if not self._initialized:
            return False
        
        container = self._ui_components.get('container')
        if container and hasattr(container, 'layout'):
            return container.layout.display != 'none'
        
        return self._is_visible


# Backward compatibility functions
def create_confirmation_area(ui_components: Dict[str, Any]) -> Any:
    """Legacy function to create a confirmation area."""
    if 'confirmation_dialog' not in ui_components:
        ui_components['confirmation_dialog'] = ConfirmationDialog("legacy_dialog")
    
    if 'confirmation_area' not in ui_components:
        dialog = ui_components['confirmation_dialog']
        dialog.initialize()
        ui_components['confirmation_area'] = dialog._ui_components['container']
    
    return ui_components['confirmation_area']

def show_confirmation_dialog(ui_components: Dict[str, Any],
                           title: str,
                           message: str,
                           on_confirm: Optional[Callable] = None,
                           on_cancel: Optional[Callable] = None,
                           confirm_text: str = "Konfirmasi",
                           cancel_text: str = "Batal",
                           danger_mode: bool = False) -> None:
    """Legacy function to show a confirmation dialog."""
    # Ensure the dialog exists in ui_components
    if 'confirmation_dialog' not in ui_components:
        ui_components['confirmation_dialog'] = ConfirmationDialog("legacy_dialog")
    
    # Get the dialog instance
    dialog = ui_components['confirmation_dialog']
    
    # Show the dialog
    dialog.show(
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
    """Legacy function to show an info dialog."""
    # Ensure the dialog exists in ui_components
    if 'confirmation_dialog' not in ui_components:
        ui_components['confirmation_dialog'] = ConfirmationDialog("legacy_dialog")
    
    # Get the dialog instance and show info
    dialog = ui_components['confirmation_dialog']
    dialog.show_info(
        title=title,
        message=message,
        on_ok=on_ok,
        ok_text=ok_text
    )

def clear_dialog_area(ui_components: Dict[str, Any]) -> None:
    """Legacy function to clear the dialog area."""
    if 'confirmation_dialog' in ui_components:
        dialog = ui_components['confirmation_dialog']
        dialog.hide()
    elif 'confirmation_area' in ui_components:
        # Fallback for old-style usage
        confirmation_area = ui_components['confirmation_area']
        if hasattr(confirmation_area, 'clear_output'):
            with confirmation_area:
                clear_output(wait=True)
        
        # Reset styles if possible
        if hasattr(confirmation_area, 'layout'):
            confirmation_area.layout.display = 'none'
            confirmation_area.layout.visibility = 'hidden'

def is_dialog_visible(ui_components: Dict[str, Any]) -> bool:
    """Legacy function to check if dialog is visible."""
    if 'confirmation_dialog' in ui_components:
        return ui_components['confirmation_dialog'].is_visible()
    elif 'confirmation_area' in ui_components:
        confirmation_area = ui_components['confirmation_area']
        return (hasattr(confirmation_area, 'layout') and 
                confirmation_area.layout.display != 'none')
    return False
