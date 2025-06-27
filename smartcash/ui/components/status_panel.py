"""
File: smartcash/ui/components/status_panel.py
Deskripsi: Status panel dengan single emoji consistency
"""

import re
import ipywidgets as widgets
from typing import Dict, Any, Optional

def _filter_emoji(message: str) -> tuple[str, str]:
    """
    Ekstrak emoji pertama dari pesan jika ada.
    
    Returns:
        tuple: (emoji, cleaned_message) - Emoji pertama yang ditemukan dan pesan yang sudah dibersihkan
    """
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
    
    # Jika ada emoji, ambil yang pertama dan bersihkan pesan
    if emojis:
        first_emoji = emojis[0]
        # Hapus semua emoji dari pesan
        cleaned = emoji_pattern.sub('', message).strip()
        return first_emoji, cleaned
    
    return "", message.strip()

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
        
        # Ekstrak emoji dari pesan atau gunakan default
        custom_emoji, cleaned_message = _filter_emoji(message)
        display_emoji = custom_emoji or icon
        
        # Tentukan gradient berdasarkan tipe status
        gradients = {
            'success': 'linear-gradient(135deg, #28a745, #34ce57)',
            'info': 'linear-gradient(135deg, #007bff, #17a2b8)',
            'warning': 'linear-gradient(135deg, #ffc107, #fd7e14)',
            'error': 'linear-gradient(135deg, #dc3545, #c82333)'
        }
        gradient = gradients.get(status_type, f'linear-gradient(135deg, {bg_color}, {bg_color})')
        
        # Buat konten HTML dengan gradient background
        html_content = f"""
        <div style="
            padding: 8px 12px;
            background: {gradient};
            color: white;
            border-radius: 4px;
            margin: 5px 0;
            font-weight: 500;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            background-size: 200% 200%;
            animation: gradient 3s ease infinite;
        ">
            {display_emoji} {cleaned_message}
            <style>
                @keyframes gradient {{
                    0% {{ background-position: 0% 50%; }}
                    50% {{ background-position: 100% 50%; }}
                    100% {{ background-position: 0% 50%; }}
                }}
            </style>
        </div>"""
        
        # Setup layout
        default_layout = {'width': '100%', 'margin': '4px 0'}
        if layout:
            default_layout.update(layout)
            
        return widgets.HTML(value=html_content, layout=widgets.Layout(**default_layout))
        
    except Exception as e:
        # Fallback minimal jika terjadi error
        custom_emoji, cleaned_message = _filter_emoji(message)
        display_emoji = custom_emoji or 'ℹ️'
        
        return widgets.HTML(
            value=f'''
            <div style="
                padding: 8px 12px;
                background: linear-gradient(135deg, #007bff, #17a2b8);
                color: white;
                border-radius: 4px;
                margin: 5px 0;
                font-weight: 500;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            ">
                {display_emoji} {cleaned_message}
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
        
        # Ekstrak emoji dari pesan atau gunakan default
        custom_emoji, cleaned_message = _filter_emoji(message)
        display_emoji = custom_emoji or icon
        
        # Tentukan gradient berdasarkan tipe status
        gradients = {
            'success': 'linear-gradient(135deg, #28a745, #34ce57)',
            'info': 'linear-gradient(135deg, #007bff, #17a2b8)',
            'warning': 'linear-gradient(135deg, #ffc107, #fd7e14)',
            'error': 'linear-gradient(135deg, #dc3545, #c82333)'
        }
        gradient = gradients.get(status_type, f'linear-gradient(135deg, {bg_color}, {bg_color})')
        
        # Buat konten HTML dengan gradient background
        html_content = f"""
        <div style="
            padding: 8px 12px;
            background: {gradient};
            color: white;
            border-radius: 4px;
            margin: 5px 0;
            font-weight: 500;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            background-size: 200% 200%;
            animation: gradient 3s ease infinite;
        ">
            {display_emoji} {cleaned_message}
            <style>
                @keyframes gradient {{
                    0% {{ background-position: 0% 50%; }}
                    50% {{ background-position: 100% 50%; }}
                    100% {{ background-position: 0% 50%; }}
                }}
            </style>
        </div>"""
        
        panel.value = html_content
        
    except Exception as e:
        # Fallback minimal jika terjadi error
        custom_emoji, cleaned_message = _filter_emoji(message)
        display_emoji = custom_emoji or 'ℹ️'
        
        panel.value = f'''
        <div style="
            padding: 8px 12px;
            background: linear-gradient(135deg, #007bff, #17a2b8);
            color: white;
            border-radius: 4px;
            margin: 5px 0;
            font-weight: 500;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        ">
            {display_emoji} {cleaned_message}
        </div>'''
