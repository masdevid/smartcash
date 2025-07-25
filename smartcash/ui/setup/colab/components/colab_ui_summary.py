"""
Summary components for Colab UI.

This module contains summary content and related functions for the Colab setup interface.
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

def create_module_summary_content(config: Dict[str, Any]) -> Optional[widgets.Widget]:
    """Create Colab-specific summary content.
    
    Args:
        config: Module configuration dictionary
        
    Returns:
        Widget containing summary content or None if not applicable
    """
    # Create summary content based on config
    summary_items = []
    
    # Add environment summary
    env_summary = [
        f"<strong>Environment:</strong> {config.get('environment', 'Not configured')}",
        f"<strong>Drive Path:</strong> {config.get('drive_path', 'Not set')}",
        f"<strong>Project Name:</strong> {config.get('project_name', 'Not set')}"
    ]
    
    # Add configuration summary
    if 'auto_detect' in config:
        auto_detect = "Enabled" if config['auto_detect'] else "Disabled"
        env_summary.append(f"<strong>Auto-detect:</strong> {auto_detect}")
    
    # Create summary widget
    if env_summary:
        summary_html = "<div style='line-height: 1.6;'>"
        summary_html += "<h4 style='margin-top: 0;'>Configuration Summary</h4>"
        summary_html += "<ul style='margin: 0; padding-left: 20px;'>"
        for item in env_summary:
            summary_html += f"<li>{item}</li>"
        summary_html += "</ul>"
        summary_html += "</div>"
        
        return widgets.HTML(summary_html)
    
    return None

# For backward compatibility
_create_module_summary_content = create_module_summary_content
