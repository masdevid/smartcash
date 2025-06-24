# File: smartcash/ui/setup/env_config/utils/ui_helpers.py
# Deskripsi: Helper functions untuk UI operations

from typing import Dict, Any

def create_status_html(is_ready: bool) -> str:
    """Create status panel HTML"""
    if is_ready:
        return """
        <div style="background: #d4edda; color: #155724; padding: 10px; border-radius: 5px;">
            ✅ Environment sudah terkonfigurasi dengan baik
        </div>
        """
    else:
        return """
        <div style="background: #fff3cd; color: #856404; padding: 10px; border-radius: 5px;">
            🔧 Environment perlu dikonfigurasi - Klik tombol setup untuk memulai
        </div>
        """

def format_log_message(timestamp: str, emoji: str, message: str, color: str) -> str:
    """Format log message dengan timestamp dan color"""
    return f'<span style="color: {color};">[{timestamp}] {emoji} {message}</span><br>'
