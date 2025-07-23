"""
file_path: /Users/masdevid/Projects/smartcash/smartcash/ui/core/ui_utils.py

Modul utilitas untuk operasi UI yang umum digunakan di seluruh aplikasi.
"""
import logging
from typing import Any, Dict, Optional, TypeVar, Type

from IPython.display import display as ipy_display

# Buat logger untuk modul ini
logger = logging.getLogger(__name__)

T = TypeVar('T')

def display_ui_module(
    module: Any,
    module_name: str,
    auto_display: bool = True,
    **kwargs
) -> Optional[Dict[str, Any]]:
    """
    Menampilkan modul UI dengan penanganan error yang konsisten.

    Args:
        module: Instance modul UI yang akan ditampilkan
        module_name: Nama modul untuk keperluan logging
        auto_display: Jika True, akan memanggil method display_ui() pada modul
        **kwargs: Argumen tambahan yang akan diteruskan ke display_ui()

    Returns:
        Dict yang berisi hasil operasi display_ui() jika auto_display=True,
        atau None jika auto_display=False

    Raises:
        RuntimeError: Jika terjadi kesalahan saat menampilkan UI
    """
    try:
        logger.debug(f"Membuat dan menampilkan {module_name} UI")
        
        if auto_display:
            logger.debug(f"Displaying {module_name} UI...")
            if not hasattr(module, 'display_ui'):
                error_msg = f"Modul {module_name} tidak memiliki method display_ui()"
                logger.error(error_msg)
                raise AttributeError(error_msg)
                
            display_result = module.display_ui(**kwargs)
            if not display_result.get('success', False):
                error_msg = display_result.get('message', f'Gagal menampilkan {module_name} UI')
                logger.error(error_msg)
                raise RuntimeError(error_msg)
                
            logger.debug(f"✅ {module_name} UI displayed successfully")
            return display_result
        else:
            logger.debug(f"✅ {module_name} UI module created (auto-display disabled)")
            # Only display the module object when auto_display is disabled
            ipy_display(module)
            return None
        
    except Exception as e:
        error_msg = f"Gagal membuat dan menampilkan {module_name} UI: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise

def create_and_display_ui(
    module_class: Type[T],
    module_name: str,
    config: Optional[Dict] = None,
    auto_display: bool = True,
    **kwargs
) -> Optional[T]:
    """
    Factory function untuk membuat dan menampilkan modul UI.

    Args:
        module_class: Kelas modul UI yang akan diinstantiasi
        module_name: Nama modul untuk keperluan logging
        config: Konfigurasi untuk modul
        auto_display: Jika True, akan memanggil method display_ui() pada modul
        **kwargs: Argumen tambahan yang akan diteruskan ke constructor modul

    Returns:
        Instance modul UI yang sudah dibuat

    Raises:
        RuntimeError: Jika terjadi kesalahan saat membuat atau menampilkan UI
    """
    try:
        logger.debug(f"Membuat instance {module_name}")
        module = module_class(config=config, **kwargs)
        
        display_ui_module(
            module=module,
            module_name=module_name,
            auto_display=auto_display,
            **kwargs
        )
        
        return module
        
    except Exception as e:
        error_msg = f"Gagal membuat {module_name}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise


def convert_summary_to_html(content: str) -> str:
    """
    Convert markdown-like content to proper HTML for Colab display.
    
    This utility function converts markdown-style content to HTML that renders
    properly in Jupyter/Colab environments, with proper line breaks and styling.
    
    Args:
        content: Raw content string (may contain markdown or plain text)
        
    Returns:
        HTML formatted content with proper line breaks and styling
        
    Example:
        >>> content = "## Results\\n- Success: 15 files\\n- **Total**: 100MB"
        >>> html = convert_summary_to_html(content)
        >>> # Returns formatted HTML with proper styling
    """
    if not content:
        return "<p>No summary available</p>"
    
    # Handle different content types
    html_lines = []
    lines = content.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Convert markdown headers to HTML
        if line.startswith('### '):
            html_lines.append(f'<h5 style="color: #2c3e50; margin: 10px 0 5px 0;">{line[4:]}</h5>')
        elif line.startswith('## '):
            html_lines.append(f'<h4 style="color: #2c3e50; margin: 15px 0 8px 0;">{line[3:]}</h4>')
        elif line.startswith('# '):
            html_lines.append(f'<h3 style="color: #2c3e50; margin: 20px 0 10px 0;">{line[2:]}</h3>')
        
        # Convert markdown lists to HTML
        elif line.startswith('- '):
            html_lines.append(f'<div style="margin: 3px 0; padding-left: 15px;">• {line[2:]}</div>')
        elif line.startswith('* '):
            html_lines.append(f'<div style="margin: 3px 0; padding-left: 15px;">• {line[2:]}</div>')
        
        # Convert markdown emphasis
        elif '**' in line:
            # Bold text - handle multiple bold sections
            import re
            line = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', line)
            html_lines.append(f'<p style="margin: 5px 0;">{line}</p>')
        
        # Handle status indicators with colors
        elif '✅' in line or '🟢' in line:
            html_lines.append(f'<div style="margin: 5px 0; color: #28a745; font-weight: 500;">{line}</div>')
        elif '❌' in line or '🔴' in line:
            html_lines.append(f'<div style="margin: 5px 0; color: #dc3545; font-weight: 500;">{line}</div>')
        elif '⚠️' in line or '🟡' in line:
            html_lines.append(f'<div style="margin: 5px 0; color: #ffc107; font-weight: 500;">{line}</div>')
        elif '📊' in line or '📈' in line or '📉' in line:
            html_lines.append(f'<div style="margin: 5px 0; color: #17a2b8; font-weight: 500;">{line}</div>')
        
        # Regular text
        else:
            # Handle key-value pairs (common in summaries)
            if ':' in line and not line.startswith('http'):
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    html_lines.append(f'<div style="margin: 3px 0;"><strong>{key}:</strong> {value}</div>')
                else:
                    html_lines.append(f'<div style="margin: 5px 0;">{line}</div>')
            else:
                html_lines.append(f'<div style="margin: 5px 0;">{line}</div>')
    
    # Join with proper spacing
    return '\n'.join(html_lines) if html_lines else "<p>No content to display</p>"


def format_operation_summary(content: str, title: str = "Operation Summary", 
                           icon: str = "📊", border_color: str = "#28a745") -> str:
    """
    Format operation summary content with consistent styling.
    
    Args:
        content: Raw summary content
        title: Summary title to display
        icon: Icon to show in title
        border_color: Left border color (default: green for success)
        
    Returns:
        Formatted HTML content ready for display
    """
    html_content = convert_summary_to_html(content)
    
    return f"""
    <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
        <h4 style="color: #2c3e50; margin: 0 0 10px 0;">{icon} {title}</h4>
        <div style="background: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid {border_color}; line-height: 1.6;">
            {html_content}
        </div>
    </div>
    """
