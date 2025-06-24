"""
File: smartcash/ui/components/header/header.py
Modern header component with gradient styling and responsive typography.
"""

from typing import Optional, List, Dict, Any
import ipywidgets as widgets
from IPython.display import display, HTML
from smartcash.ui.utils.constants import COLORS, ICONS

# Add Google Fonts for modern typography
FONT_IMPORT = """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
    :root {
        --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    body {
        font-family: var(--font-sans);
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
</style>
"""


def create_header(title: str, description: Optional[str] = None, icon: Optional[str] = None) -> widgets.HTML:
    """
    Create a modern header with gradient background.
    
    Args:
        title: Header title text
        description: Optional description text
        icon: Optional emoji or icon character
        
    Returns:
        HTML widget containing the header
    """
    try:
        # Set gradient colors - blue to emerald
        blue_color = '#3b82f6'  # Blue-500
        emerald_color = '#10b981'  # Emerald-500
        
        # Get text color from constants or use default
        text_color = COLORS.get('text', '#ffffff') if COLORS else '#ffffff'
        
        # Add icon if provided
        title_with_icon = f"{icon} {title}" if icon else title
        
        # Create gradient background - blue to emerald
        gradient = f"linear-gradient(135deg, {blue_color} 0%, {emerald_color} 100%)"
        
        # Build HTML
        html_content = f"""
        <div style="
            background: {gradient};
            color: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        ">
            <h2 style="
                margin: 0 0 0.5rem 0;
                font-size: 1.5rem;
                font-weight: 600;
                color: white;
            ">
                {title_with_icon}
            </h2>
        """.format(
            gradient=gradient,
            title_with_icon=title_with_icon,
            text_color=text_color
        )
        
        if description:
            html_content += f"""
            <p style="
                margin: 0;
                opacity: 0.9;
                font-size: 1rem;
                line-height: 1.5;
            ">
                {description}
            </p>
            """.format(description=description)
        
        html_content += "</div>"
        
        return widgets.HTML(html_content)
        
    except Exception as e:
        # Fallback minimal header
        print(f"⚠️ Error creating header: {str(e)}")
        title_with_icon = f"{icon} {title}" if icon else title
        return widgets.HTML(
            FONT_IMPORT + 
            f'<div style="padding:1rem;margin-bottom:1.5rem;background:#f0f8ff;border-radius:0.5rem">'
            f'<h2 style="margin:0 0 0.5rem 0;color:#1f2937;font-family:var(--font-sans)">{title_with_icon}</h2>'
            f'<p style="margin:0;color:#4b5563;font-family:var(--font-sans)">{description or ""}</p>'
            '</div>'
        )


def create_section_title(title: str, level: int = 3, icon: Optional[str] = None) -> widgets.HTML:
    """
    Create a modern section title with consistent styling.
    
    Args:
        title: Section title text
        level: Heading level (2-6)
        icon: Optional emoji or icon character
        
    Returns:
        HTML widget containing the section title
    """
    try:
        # Validate level
        level = max(2, min(6, level))
        
        # Get colors from constants or use defaults
        primary_color = COLORS.get('primary', '#4f46e5') if COLORS else '#4f46e5'
        text_color = COLORS.get('text', '#1f2937') if COLORS else '#1f2937'
        muted_color = COLORS.get('muted', '#6b7280') if COLORS else '#6b7280'
        
        # Add icon if provided
        title_with_icon = f"{icon} {title}" if icon else title
        
        # Base styles for all levels
        base_styles = {
            'fontFamily': "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            'fontWeight': '600',
            'lineHeight': '1.25',
            'margin': '1.5rem 0 1rem',
            'color': text_color,
        }
        
        # Level-specific styles
        level_styles = {
            2: {
                'fontSize': '1.5rem',
                'fontWeight': '700',
                'margin': '2rem 0 1.25rem',
                'paddingBottom': '0.5rem',
                'borderBottom': f'2px solid {primary_color}',
                'display': 'inline-block',
                'position': 'relative',
            },
            3: {
                'fontSize': '1.25rem',
                'margin': '1.75rem 0 1rem',
                'color': text_color,
            },
            4: {
                'fontSize': '1.1rem',
                'margin': '1.5rem 0 0.75rem',
                'color': text_color,
                'opacity': '0.9',
            },
            5: {
                'fontSize': '1rem',
                'margin': '1.25rem 0 0.5rem',
                'color': muted_color,
                'textTransform': 'uppercase',
                'letterSpacing': '0.05em',
                'fontWeight': '600',
            },
            6: {
                'fontSize': '0.9rem',
                'margin': '1rem 0 0.5rem',
                'color': muted_color,
                'fontWeight': '600',
            }
        }
        
        # Get styles for the specified level
        styles = {**base_styles, **level_styles.get(level, level_styles[3])}
        
        # Generate CSS string
        style_str = '; '.join(f"{k}: {v}" for k, v in styles.items() if v is not None)
        
        # Add accent line for h2
        if level == 2:
            title_with_icon = f"""
                {title_with_icon}
                <span style="
                    position: absolute;
                    left: 0;
                    bottom: -2px;
                    width: 100%;
                    height: 2px;
                    background: linear-gradient(90deg, {primary_color} 0%, {primary_color}33 100%);
                    content: '';
                "></span>
            """.format(primary_color=primary_color)
        
        return widgets.HTML(f'<h{level} style="{style_str}">{title_with_icon}</h{level}>')
        
    except Exception as e:
        # Fallback minimal title
        print(f"⚠️ Error creating section title: {str(e)}")
        title_with_icon = f"{icon} {title}" if icon else title
        return widgets.HTML(
            FONT_IMPORT + 
            f'<h{level} style="margin:1.5rem 0 1rem;font-weight:600;font-family:var(--font-sans)">'
            f'{title_with_icon}</h{level}>'
        )


# Module exports
__all__ = [
    'create_header',
    'create_section_title'
]
