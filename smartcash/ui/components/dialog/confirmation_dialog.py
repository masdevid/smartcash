"""
File: smartcash/ui/components/dialog/confirmation_dialog.py
Deskripsi: Simple HTML glass morphism dialog di dalam confirmation area (tidak trigger dialog asli Colab)
"""

from turtle import width
from typing import Dict, Any, Callable, Optional
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import uuid

def create_confirmation_area(ui_components: Dict[str, Any]) -> widgets.Output:
    """Buat area konfirmasi yang simple"""
    
    if 'confirmation_area' not in ui_components or not isinstance(ui_components['confirmation_area'], widgets.Output):
        # Create the output widget first
        confirmation_area = widgets.Output()
        
        # Set layout with proper CSS classes
        confirmation_area.layout = widgets.Layout(
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
        
        # Add class using the proper method
        if hasattr(confirmation_area, 'add_class'):
            confirmation_area.add_class('confirmation-area')
        else:
            # Fallback for older ipywidgets versions
            try:
                confirmation_area.add_class('confirmation-area')
            except Exception:
                pass  # Ignore if add_class is not available
                
        ui_components['confirmation_area'] = confirmation_area
    
    return ui_components['confirmation_area']

def show_confirmation_dialog(ui_components: Dict[str, Any],
                           title: str,
                           message: str,
                           on_confirm: Optional[Callable] = None,
                           on_cancel: Optional[Callable] = None,
                           confirm_text: str = "Konfirmasi",
                           cancel_text: str = "Batal",
                           danger_mode: bool = False) -> None:
    """Show simple HTML glass morphism dialog dalam confirmation area"""
    
    try:
        # Ensure confirmation area exists
        confirmation_area = create_confirmation_area(ui_components)
        
        # Show confirmation area with animation
        confirmation_area.layout.display = 'block'
        confirmation_area.layout.visibility = 'visible'
        confirmation_area.layout.height = '250px'  # Set to 250px as requested
        confirmation_area.layout.min_height = '250px'
        confirmation_area.layout.max_height = '250px'
        confirmation_area.layout.padding = '10px 0'  # Reduced padding for more compact look
        confirmation_area.layout.margin = '5px 0'  # Reduced margin
        confirmation_area.layout.overflow = 'hidden'
        
        # Generate unique IDs untuk buttons
        dialog_id = f"dialog_{uuid.uuid4().hex[:8]}"
        confirm_id = f"confirm_{uuid.uuid4().hex[:8]}"
        cancel_id = f"cancel_{uuid.uuid4().hex[:8]}"
        
        # Store callbacks in ui_components untuk JavaScript access
        ui_components[f'_confirm_callback_{confirm_id}'] = on_confirm
        ui_components[f'_cancel_callback_{cancel_id}'] = on_cancel
        ui_components[f'_dialog_id'] = dialog_id
        
        # Define colors berdasarkan mode
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
        
        # Build HTML dengan inline styles complete
        dialog_html = f"""
        <style>
        .smartcash-dialog-{dialog_id} {{
            position: relative;
            max-width: 80%;
            width: 80%;
            margin: 0 auto;
            padding: 0;
            overflow-x: hidden;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
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
            padding: 20px;
            height: 300px;
            overflow-y: auto;
            transition: all 0.3s ease;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            align-items: center;
            text-align: center;
            position: relative;
        }}
        
        .glass-card-{dialog_id}::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: {card_accent};
            border-radius: 20px;
            z-index: -1;
        }}
        
        .dialog-header-{dialog_id} {{
            color: #1a202c;
            font-size: 1rem;
            font-weight: 600;
            line-height: 1.3;
            padding: 0 4px;
        }}
        
        .dialog-message-{dialog_id} {{
            color: #4b5563;
            font-size: 14px;  # Slightly smaller font
            line-height: 1.3;  # Tighter line height
            margin: 10px 0 15px;  # Reduced margins
            padding: 0 10px;
            max-height: 120px;  # Limit height for scrolling
            overflow-y: auto;  # Add scroll if content is too long
            white-space: pre-line;
        }}
        
        .dialog-actions-{dialog_id} {{
            display: flex;
            gap: 12px;
            justify-content: center;
            align-items: center;
            flex-wrap: wrap;
        }}
        
        .glass-btn-{dialog_id} {{
            border: none;
            border-radius: 6px;
            padding: 8px 16px;
            font-size: 0.8rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.15s ease;
            min-width: 80px;
            text-align: center;
            user-select: none;
            outline: none;
            position: relative;
            overflow: hidden;
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
                padding: 20px 16px;
                border-radius: 16px;
            }}
            
            .dialog-header-{dialog_id} {{
                font-size: 1.1rem;
            }}
            
            .dialog-message-{dialog_id} {{
                font-size: 0.9rem;
            }}
            
            .glass-btn-{dialog_id} {{
                padding: 8px 16px;
                font-size: 0.85rem;
                min-width: 80px;
            }}
            
            .dialog-actions-{dialog_id} {{
                gap: 10px;
            }}
        }}
        
        @media (max-width: 480px) {{
            .glass-card-{dialog_id} {{
                padding: 15px;  # Reduced padding
                margin: 0;
            }}
            
            .dialog-actions-{dialog_id} {{
                flex-direction: row;  # Buttons side by side
                justify-content: flex-end;  # Align to right
                gap: 10px;  # Space between buttons
                width: 100%;
                margin-top: 10px;  # Add some space above buttons
            }}
            
            .glass-btn-{dialog_id} {{
                width: 100%;
                max-width: 200px;
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
                    <button id="{cancel_id}" class="glass-btn-{dialog_id} cancel-btn-{dialog_id}">
                        {cancel_text}
                    </button>
                </div>
            </div>
        </div>
        
        <script>
        (function() {{
            const confirmBtn = document.getElementById('{confirm_id}');
            const cancelBtn = document.getElementById('{cancel_id}');
            
            function handleConfirm() {{
                try {{
                    // Add fade out animation
                    const card = document.querySelector('.glass-card-{dialog_id}');
                    if (card) {{
                        card.style.animation = 'none';
                        card.style.transition = 'opacity 0.2s ease';
                        card.style.opacity = '0';
                    }}
                    
                    setTimeout(() => {{
                        // First, reset the confirmation area styles in the frontend
                        const confirmationArea = document.querySelector('.confirmation-area');
                        if (confirmationArea) {{
                            confirmationArea.style.transition = 'all 0.3s ease';
                            confirmationArea.style.height = '0';
                            confirmationArea.style.minHeight = '0';
                            confirmationArea.style.maxHeight = '0';
                            confirmationArea.style.padding = '0';
                            confirmationArea.style.margin = '0';
                            confirmationArea.style.overflow = 'hidden';
                            confirmationArea.style.border = 'none';
                        }}
                        
                        // Then trigger Python callback after a short delay
                        if (window.jupyter && window.jupyter.notebook && window.jupyter.notebook.kernel) {{
                            const code = `
try:
    callback = ui_components.get('_confirm_callback_{confirm_id}')
    if callback:
        callback()
    confirmation_area = ui_components.get('confirmation_area')
    if confirmation_area:
        # Reset to default values
        confirmation_area.layout.display = 'none'
        confirmation_area.layout.visibility = 'hidden'
        confirmation_area.layout.height = None  # Reset to auto
        confirmation_area.layout.min_height = '50px'  # Default min height
        confirmation_area.layout.max_height = '500px'  # Default max height
        confirmation_area.layout.padding = '10px'  # Default padding
        confirmation_area.layout.margin = '10px 0'  # Default margin
        confirmation_area.layout.overflow = 'hidden'  # Keep hidden when not in use
        confirmation_area.layout.border = '1px solid #e0e0e0'  # Default border
        confirmation_area.layout.border_radius = '4px'  # Default border radius
    ui_components.pop('_confirm_callback_{confirm_id}', None)
    ui_components.pop('_cancel_callback_{cancel_id}', None)
    # Clear the output after resetting styles
    if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'clear_output'):
        with ui_components['confirmation_area']:
            from IPython.display import clear_output
            clear_output(wait=True)
except Exception as e:
    print(f"⚠️ Error in confirm callback: {{e}}")
`;
                            window.jupyter.notebook.kernel.execute(code);
                        }}
                    }}, 200);
                }} catch (e) {{
                    console.error('Error in confirm handler:', e);
                }}
            }}
            
            function handleCancel() {{
                try {{
                    // Add fade out animation
                    const card = document.querySelector('.glass-card-{dialog_id}');
                    if (card) {{
                        card.style.animation = 'none';
                        card.style.transition = 'all 0.3s ease';
                        card.style.opacity = '0';
                        card.style.transform = 'scale(0.95)';
                        card.style.margin = '0';
                        card.style.padding = '0';
                        card.style.border = 'none';
                        card.style.height = '0';
                    }}
                    
                    setTimeout(() => {{
                        // First, reset the confirmation area styles in the frontend
                        const confirmationArea = document.querySelector('.confirmation-area');
                        if (confirmationArea) {{
                            confirmationArea.style.transition = 'all 0.3s ease';
                            confirmationArea.style.height = '0';
                            confirmationArea.style.minHeight = '0';
                            confirmationArea.style.maxHeight = '0';
                            confirmationArea.style.padding = '0';
                            confirmationArea.style.margin = '0';
                            confirmationArea.style.overflow = 'hidden';
                            confirmationArea.style.border = 'none';
                        }}
                        
                        // Then trigger Python callback after a short delay
                        if (window.jupyter && window.jupyter.notebook && window.jupyter.notebook.kernel) {{
                            const code = `
try:
    callback = ui_components.get('_cancel_callback_{cancel_id}')
    if callback:
        callback()
    confirmation_area = ui_components.get('confirmation_area')
    if confirmation_area:
        # Reset to default values instead of 0
        confirmation_area.layout.display = 'none'
        confirmation_area.layout.visibility = 'hidden'
        confirmation_area.layout.height = None  # Reset to auto
        confirmation_area.layout.min_height = '50px'  # Default min height
        confirmation_area.layout.max_height = '500px'  # Default max height
        confirmation_area.layout.padding = '10px'  # Default padding
        confirmation_area.layout.margin = '10px 0'  # Default margin
        confirmation_area.layout.overflow = 'hidden'  # Keep hidden when not in use
        confirmation_area.layout.border = '1px solid #e0e0e0'  # Default border
        confirmation_area.layout.border_radius = '4px'  # Default border radius
    ui_components.pop('_confirm_callback_{confirm_id}', None)
    ui_components.pop('_cancel_callback_{cancel_id}', None)
    # Clear the output after resetting styles
    if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'clear_output'):
        with ui_components['confirmation_area']:
            from IPython.display import clear_output
            clear_output(wait=True)
except Exception as e:
    print(f"⚠️ Error in cancel callback: {{e}}")
`;
                            window.jupyter.notebook.kernel.execute(code);
                        }}
                    }}, 300);
                }} catch (e) {{
                    console.error('Error in cancel handler:', e);
                }}
            }}
            
            if (confirmBtn) {{
                confirmBtn.onclick = handleConfirm;
            }}
            
            if (cancelBtn) {{
                cancelBtn.onclick = handleCancel;
            }}
        }})();
        </script>
        """
        
        # Display HTML di confirmation area
        with confirmation_area:
            clear_output(wait=True)
            display(HTML(dialog_html))
        
    except Exception as e:
        print(f"⚠️ Error showing dialog: {str(e)}")
        # Simple fallback
        response = input(f"{title}: {message} (y/N): ").lower().strip()
        if response in ['y', 'yes', 'ya'] and on_confirm:
            on_confirm()
        elif on_cancel:
            on_cancel()

def show_info_dialog(ui_components: Dict[str, Any],
                    title: str,
                    message: str,
                    on_ok: Optional[Callable] = None,
                    ok_text: str = "OK") -> None:
    """Show simple info dialog"""
    show_confirmation_dialog(
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
    """Clear dialog area"""
    confirmation_area = ui_components.get('confirmation_area')
    if confirmation_area:
        confirmation_area.layout.display = 'none'
        with confirmation_area:
            clear_output(wait=True)
        
        # Clean up callbacks
        keys_to_remove = [k for k in ui_components.keys() 
                         if k.startswith('_confirm_callback_') or k.startswith('_cancel_callback_')]
        for key in keys_to_remove:
            ui_components.pop(key, None)

def is_dialog_visible(ui_components: Dict[str, Any]) -> bool:
    """Check apakah dialog sedang visible"""
    confirmation_area = ui_components.get('confirmation_area')
    if confirmation_area and hasattr(confirmation_area, 'layout'):
        return confirmation_area.layout.display != 'none'
    return False