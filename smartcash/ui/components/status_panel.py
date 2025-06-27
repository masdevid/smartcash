"""
File: smartcash/ui/components/status_panel.py
Deskripsi: Status panel dengan single emoji consistency
"""

import re
import ipywidgets as widgets
from typing import Dict, Any, Optional

def _filter_emoji(message: str) -> str:
    """Filter pesan untuk menghapus semua emoji kecuali yang pertama"""
    # Regex untuk mendeteksi emoji (mencocokkan sebagian besar emoji Unicode)
    emoji_pattern = re.compile(
        r'[\U0001F1E0-\U0001F1FF]|'  # bendera
        r'[\U0001F300-\U0001F5FF]|'  # simbol & piktograf
        r'[\U0001F600-\U0001F64F]|'  # emosi
        r'[\U0001F680-\U0001F6FF]|'  # transportasi & simbol
        r'[\U0001F700-\U0001F77F]|'  # alchemy
        r'[\U0001F780-\U0001F7FF]|'  # Geometric Shapes
        r'[\U0001F800-\U0001F8FF]|'  # Supplemental Arrows-C
        r'[\U0001F900-\U0001F9FF]|'  # Supplemental Symbols and Pictographs
        r'[\U0001FA00-\U0001FA6F]|'  # Chess Symbols
        r'[\U0001FA70-\U0001FAFF]|'  # Symbols and Pictographs Extended-A
        r'[\U00002702-\U000027B0]|'  # Dingbats
        r'[\U000024C2-\U0001F251]'    # Enclosed characters
    )
    
    # Temukan semua emoji dalam pesan
    emojis = emoji_pattern.findall(message)
    
    # Jika ada emoji, hapus semua kecuali yang pertama
    if emojis:
        first_emoji = emojis[0]
        # Hapus semua emoji dari pesan
        cleaned = emoji_pattern.sub('', message).strip()
        # Kembalikan emoji pertama diikuti pesan yang sudah dibersihkan
        return f"{first_emoji} {cleaned}"
    return message

def create_status_panel(message: str = "", status_type: str = "info", layout: Optional[Dict[str, Any]] = None) -> widgets.HTML:
    """Buat status panel dengan single emoji consistency
    
    Args:
        message: Pesan yang akan ditampilkan (hanya emoji pertama yang akan ditampilkan)
        status_type: Tipe status (info, success, warning, error)
        layout: Konfigurasi layout opsional
    """
    from smartcash.ui.utils.constants import get_alert_style
    
    try:
        # Dapatkan style dengan fallback yang aman
        style_info = get_alert_style(status_type)
        
        # Gunakan nilai style dengan key yang dijamin ada
        bg_color = style_info['bg_color']
        text_color = style_info['text_color']
        icon = style_info['icon']
        
        # Filter pesan untuk memastikan hanya satu emoji
        filtered_message = _filter_emoji(f"{icon} {message}")
        
        # Buat konten HTML
        html_content = f"""
        <div style="
            padding: 6px 10px;
            background-color: {bg_color};
            color: {text_color};
            border-radius: 6px;
            margin: 3px 0;
            border: 1px solid rgba(0,0,0,0.08);
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            font-size: 13px;
            line-height: 1.4;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        ">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 14px;">{icon}</span>
                <span style="flex: 1;">{filtered_message}</span>
            </div>
        </div>"""
        
        # Setup layout
        default_layout = {'width': '100%', 'margin': '4px 0'}
        if layout:
            default_layout.update(layout)
            
        return widgets.HTML(value=html_content, layout=widgets.Layout(**default_layout))
        
    except Exception as e:
        # Fallback minimal jika terjadi error
        return widgets.HTML(
            value=f'''
            <div style="
                padding: 6px 10px;
                background: #f8f9fa;
                color: #495057;
                border-radius: 6px;
                margin: 4px 0;
                border: 1px solid rgba(0,0,0,0.08);
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
                font-size: 13px;
                line-height: 1.4;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            ">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 14px;">ℹ️</span>
                    <span style="flex: 1;">{filtered_message}</span>
                </div>
            </div>''',
            layout=widgets.Layout(width='100%', margin='4px 0')
        )

def update_status_panel(panel: widgets.HTML, message: str, status_type: str = "info") -> None:
    """Update status panel dengan single emoji consistency
    
    Args:
        panel: Widget HTML yang akan diupdate
        message: Pesan baru (hanya emoji pertama yang akan ditampilkan)
        status_type: Tipe status (info, success, warning, error)
    """
    from smartcash.ui.utils.constants import get_alert_style
    
    try:
        # Dapatkan style dengan fallback yang aman
        style_info = get_alert_style(status_type)
        
        # Gunakan nilai style dengan key yang dijamin ada
        bg_color = style_info['bg_color']
        text_color = style_info['text_color']
        icon = style_info['icon']
        
        # Filter pesan untuk memastikan hanya satu emoji
        filtered_message = _filter_emoji(f"{icon} {message}")
        
        # Buat konten HTML
        html_content = f"""
        <div style="
            padding: 6px 10px;
            background-color: {bg_color};
            color: {text_color};
            border-radius: 6px;
            margin: 3px 0;
            border: 1px solid rgba(0,0,0,0.08);
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            font-size: 13px;
            line-height: 1.4;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        ">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 14px;">{icon}</span>
                <span style="flex: 1;">{filtered_message}</span>
            </div>
        </div>"""
        
        panel.value = html_content
        
    except Exception as e:
        # Fallback minimal jika terjadi error
        panel.value = f'''
        <div style="
            padding: 10px;
            background-color: #f8f9fa;
            color: #212529;
            border-radius: 4px;
            margin: 5px 0;
            border-left: 4px solid #6c757d;
        ">
            <p style="margin: 5px 0">ℹ️ {filtered_message}</p>
        </div>'''
