"""
File: smartcash/ui/components/dialog/confirmation_dialog.py
Deskripsi: Komponen dialog modern dengan glass morphism untuk konfirmasi dan interaksi pengguna
"""

from typing import Dict, Any, Callable, Optional, Tuple
import ipywidgets as widgets
from IPython.display import display, clear_output

# CSS untuk glass morphism effect
GLASS_STYLE = """
.glass-card {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.18);
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
    padding: 20px;
    transition: all 0.3s ease;
}

.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.2);
}

.dialog-title {
    color: #2d3748;
    font-size: 1.4em;
    font-weight: 600;
    margin: 0 0 12px 0;
    text-align: center;
}

.dialog-message {
    color: #4a5568;
    font-size: 1em;
    line-height: 1.5;
    margin: 0 0 20px 0;
    text-align: center;
}

.dialog-button {
    border: none;
    border-radius: 8px;
    cursor: pointer;
    font-weight: 500;
    padding: 10px 20px;
    transition: all 0.2s ease;
    min-width: 100px;
    text-align: center;
}

.dialog-button.primary {
    background: #4f46e5;
    color: white;
}

.dialog-button.primary:hover {
    background: #4338ca;
    transform: translateY(-1px);
}

.dialog-button.secondary {
    background: rgba(255, 255, 255, 0.7);
    color: #4f46e5;
    border: 1px solid #e2e8f0;
}

.dialog-button.secondary:hover {
    background: rgba(255, 255, 255, 0.9);
    transform: translateY(-1px);
}

.dialog-button.danger {
    background: #ef4444;
    color: white;
}

.dialog-button.danger:hover {
    background: #dc2626;
    transform: translateY(-1px);
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.dialog-container {
    animation: fadeIn 0.3s ease-out;
    max-width: 100%;
    width: 450px;
    margin: 0 auto;
}

@media (max-width: 480px) {
    .dialog-container {
        width: 90%;
        padding: 0 10px;
    }
    
    .dialog-button {
        padding: 8px 16px;
        min-width: 80px;
    }
}
"""

# Inject the CSS
widgets.HTML(f"""
<style>
{GLASS_STYLE}
</style>
""")

def create_confirmation_area(
    width: str = '100%',
    min_height: str = '0px',
    max_height: str = '90vh',
    margin: str = '10px 0',
    padding: str = '20px',
    border_radius: str = '16px',
    background_color: str = 'rgba(255, 255, 255, 0.15)',
    backdrop_filter: str = 'blur(12px)',
    border: str = '1px solid rgba(255, 255, 255, 0.18)',
    box_shadow: str = '0 8px 32px 0 rgba(31, 38, 135, 0.15)',
    overflow: str = 'auto',
    visibility: str = 'hidden'
) -> Tuple[widgets.Output, Dict[str, str]]:
    """
    Membuat area konfirmasi modern dengan glass morphism effect.
    
    Args:
        width: Lebar area konfirmasi (contoh: '100%', '500px')
        min_height: Tinggi minimum area
        max_height: Tinggi maksimum area (contoh: '80vh', '600px')
        margin: Margin area (contoh: '10px 0', '20px auto')
        padding: Padding area
        border_radius: Sudut border untuk efek rounded corner
        background_color: Warna latar belakang dengan transparansi
        backdrop_filter: Filter untuk efek glass (contoh: 'blur(12px)')
        border: Gaya border dengan transparansi
        box_shadow: Bayangan untuk efek kedalaman
        overflow: Pengaturan overflow
        visibility: Visibilitas awal ('hidden' atau 'visible')
        overflow: Pengaturan overflow
        visibility: Visibilitas awal ('hidden' atau 'visible')
        
    Returns:
        Tuple berisi:
        - Widget Output yang dapat digunakan untuk menampilkan konten
        - Dictionary berisi layout yang digunakan (untuk referensi)
    """
    layout = {
        'width': width,
        'min_height': min_height,
        'max_height': max_height,
        'margin': margin,
        'padding': padding,
        'border': border,
        'border_radius': border_radius,
        'background_color': background_color,
        'overflow': overflow,
        'visibility': visibility
    }
    
    confirmation_area = widgets.Output(
        layout=widgets.Layout(**{
            k: v for k, v in layout.items() 
            if k not in ['visibility']  # visibility diatur melalui display property
        })
    )
    
    # Set initial visibility
    confirmation_area.layout.display = 'none' if visibility == 'hidden' else 'flex'
    
    return confirmation_area, layout

def show_confirmation_dialog(
    ui_components: Dict[str, Any],
    title: str,
    message: str,
    on_confirm: Callable = None,
    on_cancel: Callable = None,
    confirm_text: str = "Ya",
    cancel_text: str = "Batal",
    danger_mode: bool = False
) -> None:
    """Tampilkan dialog konfirmasi dengan desain glass morphism modern"""
    dialog_area = ui_components.get('confirmation_area') or ui_components.get('dialog_area')
    if not dialog_area:
        print(f"⚠️ {title}: {message}")
        if on_confirm:
            on_confirm()
        return
    
    try:
        # Pastikan dialog area visible dan bersihkan konten yang ada
        if hasattr(dialog_area, 'layout') and hasattr(dialog_area.layout, 'display'):
            dialog_area.layout.display = 'flex'
        
        with dialog_area:
            clear_output(wait=True)
        
        # Buat callback untuk tombol dengan error handling
        def handle_confirm(btn):
            try:
                with dialog_area:
                    clear_output(wait=True)
                    if hasattr(dialog_area, 'layout') and hasattr(dialog_area.layout, 'display'):
                        dialog_area.layout.display = 'none'
                if on_confirm:
                    on_confirm()
            except Exception as e:
                print(f"Error in confirm handler: {str(e)}")
        
        def handle_cancel(btn):
            try:
                with dialog_area:
                    clear_output(wait=True)
                    if hasattr(dialog_area, 'layout') and hasattr(dialog_area.layout, 'display'):
                        dialog_area.layout.display = 'none'
                if on_cancel:
                    on_cancel()
            except Exception as e:
                print(f"Error in cancel handler: {str(e)}")
        
        # Tentukan gaya tombol berdasarkan mode
        confirm_btn_class = 'danger' if danger_mode else 'primary'
        
        # Buat HTML untuk dialog
        dialog_html = f"""
        <div class="dialog-container">
            <div class="glass-card">
                <h3 class="dialog-title">{title}</h3>
                <div class="dialog-message">{message}</div>
                <div style="display: flex; justify-content: center; gap: 12px; margin-top: 20px;">
                    <button class="dialog-button {confirm_btn_class}" id="confirmBtn">{confirm_text}</button>
                    <button class="dialog-button secondary" id="cancelBtn">{cancel_text}</button>
                </div>
            </div>
        </div>
        """
        
        # Buat widget HTML
        dialog = widgets.HTML(dialog_html)
        
        # Tambahkan event handlers
        display(HTML(
            f"""
            <script>
            document.getElementById('confirmBtn').addEventListener('click', function(e) {{
                var kernel = IPython.notebook.kernel;
                kernel.execute('confirm_clicked = True');
            }});
            document.getElementById('cancelBtn').addEventListener('click', function(e) {{
                var kernel = IPython.notebook.kernel;
                kernel.execute('cancel_clicked = True');
            }});
            </script>
            """
        ))
        
        # Tampilkan dialog
        with dialog_area:
            display(dialog)
            
        # Simpan referensi ke fungsi handler
        dialog._confirm_handler = handle_confirm
        dialog._cancel_handler = handle_cancel
        
        # Atur event handler untuk tombol
        display(HTML(
            f"""
            <script>
            document.getElementById('confirmBtn').onclick = function() {{
                var kernel = IPython.notebook.kernel;
                kernel.execute('from IPython.display import display, clear_output; ' +
                             'with ({(dialog_area._repr_mimebundle_()["text/plain"])}): ' +
                             'clear_output(wait=True)');
                {handle_confirm.__code__.co_consts[0]};
            }};
            document.getElementById('cancelBtn').onclick = function() {{
                var kernel = IPython.notebook.kernel;
                kernel.execute('from IPython.display import display, clear_output; ' +
                             'with ({(dialog_area._repr_mimebundle_()["text/plain"]).replace("'", "\\'")}): ' +
                             'clear_output(wait=True)');
                {handle_cancel.__code__.co_consts[0]};
            }};
            </script>
            """
        ))
        
    except Exception as e:
        print(f"Error showing confirmation dialog: {str(e)}")
        # Fallback ke print sederhana jika dialog gagal
        print(f"⚠️ {title}: {message} [Confirm: {confirm_text} / {cancel_text}]")

def show_info_dialog(
    ui_components: Dict[str, Any],
    title: str,
    message: str,
    on_close: Callable = None,
    close_text: str = "Tutup",
    dialog_type: str = "info"
) -> None:
    """Tampilkan dialog info dengan desain glass morphism modern"""
    """Show info dialog with consistent styling and better state management"""
    dialog_area = ui_components.get('confirmation_area') or ui_components.get('dialog_area')
    if not dialog_area:
        print(f"ℹ️ {title}: {message}")
        if on_close:
            on_close()
        return
    
    try:
        # Ensure dialog area is visible
        if hasattr(dialog_area, 'layout') and hasattr(dialog_area.layout, 'display'):
            dialog_area.layout.display = 'flex'
        
        def handle_close(btn):
            try:
                with dialog_area:
                    clear_output(wait=True)
                    if hasattr(dialog_area, 'layout') and hasattr(dialog_area.layout, 'display'):
                        dialog_area.layout.display = 'none'
                if on_close:
                    on_close()
            except Exception as e:
                print(f"Error in close handler: {str(e)}")
        
        # Modern color scheme with better contrast
        colors = {
            'info': '#3b82f6',    # blue-500
            'success': '#10b981', # emerald-500
            'warning': '#f59e0b', # amber-500
            'error': '#ef4444'    # red-500
        }
        
        # Icons for each dialog type
        icons = {
            'info': 'ℹ️',
            'success': '✅',
            'warning': '⚠️',
            'error': '❌'
        }
        
        color = colors.get(dialog_type.lower(), '#3b82f6')
        icon = icons.get(dialog_type.lower(), 'ℹ️')
        
        # Create styled close button with hover effect
        close_btn = widgets.Button(
            description=close_text,
            layout=widgets.Layout(
                width='120px',
                margin='10px 0 0 0',
                padding='8px 16px',
                border_radius='6px',
                border='none',
                background_color=color,
                font_weight='500',
                cursor='pointer',
                color='white',
                box_shadow='0 2px 4px rgba(0,0,0,0.1)'
            )
        )
        close_btn.on_click(handle_close)
        
        # Add hover effect using CSS
        close_btn.add_class('dialog-button')
        
        # Create dialog with glass morphism effect
        dialog = widgets.VBox([
            widgets.HTML(f"""
                <div style='text-align:center; margin-bottom:12px; font-size:2.5em; color:{color};'>{icon}</div>
                <h3 style='color:{color}; text-align:center; margin:0 0 12px 0; font-size:1.4em;'>{title}</h3>
                <div style='color:#4b5563; text-align:center; line-height:1.5;'>{message}</div>
            """),
            widgets.HBox([close_btn], layout=widgets.Layout(justify_content='center'))
        ], layout=widgets.Layout(
            padding='24px',
            border_radius='12px',
            margin='10px auto',
            width='auto',
            max_width='500px',
            background='rgba(255, 255, 255, 0.9)',
            backdrop_filter='blur(10px)',
            box_shadow='0 8px 32px 0 rgba(31, 38, 135, 0.15)',
            border=f'1px solid rgba(255, 255, 255, 0.2)'
        ))
        
        # Add responsive styles and effects
        display(HTML(f"""
        <style>
            .dialog-button {{
                transition: all 0.2s ease;
            }}
            .dialog-button:hover {{
                opacity: 0.9;
                transform: translateY(-1px);
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15) !important;
            }}
            .dialog-button:active {{
                transform: translateY(0);
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
            }}
            
            /* Responsive adjustments */
            @media (max-width: 600px) {{
                .p-Widget {{
                    max-width: 95% !important;
                    margin: 10px auto !important;
                }}
                .p-Widget h3 {{
                    font-size: 1.2em !important;
                    margin-bottom: 8px !important;
                }}
                .p-Widget .p-Widget {{
                    padding: 16px !important;
                }}
            }}
            
            /* Animation for dialog appearance */
            @keyframes dialogFadeIn {{
                from {{ opacity: 0; transform: translateY(20px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            
            .p-Widget > div[data-mime-type="application/vnd.jupyter.widget-view+json"] {{
                animation: dialogFadeIn 0.3s ease-out forwards;
            }}
        </style>
        """))
        
        with dialog_area:
            clear_output(wait=True)
            display(dialog)
            
    except Exception as e:
        print(f"Error showing info dialog: {str(e)}")
        # Fallback to simple print if dialog fails
        print(f"ℹ️ {title}: {message}")

def clear_dialog_area(ui_components: Dict[str, Any]) -> None:
    """Clear dialog area"""
    dialog_area = ui_components.get('confirmation_area') or ui_components.get('dialog_area')
    if dialog_area:
        with dialog_area:
            clear_output(wait=True)

def is_dialog_visible(ui_components: Dict[str, Any]) -> bool:
    """Periksa apakah dialog sedang terlihat"""
    dialog_area = ui_components.get('confirmation_area') or ui_components.get('dialog_area')
    if not dialog_area:
        return False
    
    try:
        if hasattr(dialog_area, 'layout') and hasattr(dialog_area.layout, 'display'):
            return dialog_area.layout.display != 'none' and dialog_area.layout.display is not None
    except Exception:
        pass
    return False