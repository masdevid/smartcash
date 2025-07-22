"""
File: smartcash/ui/dataset/augmentation/components/info_accordion.py
Description: Info accordion widget for augmentation UI.
"""
from typing import List, Dict, Any

import ipywidgets as widgets
from IPython.display import display


def create_info_accordion() -> widgets.Widget:
    """
    Create the info accordion for the footer with augmentation tips.
    
    Returns:
        Widget containing the info accordion content
    """
    # Define accordion sections with their titles and content
    sections = [
        {
            'title': '💡 Augmentation Tips',
            'items': [
                "Use 'Combined' type for balanced position + lighting effects",
                'Monitor target count to maintain dataset balance',
                'Preview results before running full augmentation',
                'Backup data before cleanup operations'
            ]
        },
        {
            'title': '📊 Recommended Settings',
            'items': [
                'Start with low intensity (0.1-0.3) and gradually increase',
                'Use 2-4 augmentation types for balanced results',
                'Check class distribution after augmentation'
            ]
        }
    ]
    
    # Create HTML content for each section
    accordion_children = []
    for section in sections:
        items_html = '\n'.join(f'<li>{item}</li>' for item in section['items'])
        content = f"""
        <div style="font-size: 0.9em; line-height: 1.5; padding: 8px 0;">
            <ul style="margin: 0 0 0 15px; padding: 0;">
                {items_html}
            </ul>
        </div>
        """
        accordion_children.append(widgets.HTML(content))
    
    # Create and configure the accordion
    accordion = widgets.Accordion(children=accordion_children)
    
    # Set section titles
    for i, section in enumerate(sections):
        accordion.set_title(i, section['title'])
    
    # Apply styling
    accordion.add_class('info-accordion')
    accordion.layout = widgets.Layout(
        width='100%',
        margin='0 0 10px 0',
        border='1px solid #e0e0e0',
        border_radius='8px',
        overflow='hidden'
    )
    
    # Add CSS for the accordion
    display(widgets.HTML("""
    <style>
        .info-accordion .p-Accordion-header {
            background-color: #f5f5f5;
            padding: 8px 12px;
            cursor: pointer;
            font-weight: 600;
            border-bottom: 1px solid #e0e0e0;
        }
        .info-accordion .p-Accordion-header:hover {
            background-color: #ebebeb;
        }
        .info-accordion .p-Accordion-contents {
            padding: 8px 12px;
            background-color: #fafafa;
        }
    </style>
    """))
    
    return accordion
