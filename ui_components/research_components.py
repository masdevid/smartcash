"""
File: ui_components/research_components.py
Author: Alfrida Sabar
Deskripsi: Komponen UI untuk evaluasi perbandingan skenario penelitian.
"""

import ipywidgets as widgets
from IPython.display import display, HTML

def create_scenario_checkboxes(scenarios):
    """
    Buat checkboxes untuk memilih skenario penelitian.
    
    Args:
        scenarios: List skenario penelitian
        
    Returns:
        List checkbox widgets
    """
    scenario_checkboxes = []
    for scenario in scenarios:
        checkbox = widgets.Checkbox(
            value=True,
            description=f"{scenario['name']}: {scenario['description']}",
            layout=widgets.Layout(width='auto'),
            style={'description_width': 'initial'}
        )
        scenario_checkboxes.append(checkbox)
    
    return scenario_checkboxes

def create_evaluation_controls():
    """
    Buat kontrol untuk menjalankan evaluasi.
    
    Returns:
        Dictionary berisi widget tombol evaluasi
    """
    # Button untuk menjalankan evaluasi
    run_button = widgets.Button(
        description='Jalankan Evaluasi',
        button_style='success',
        icon='play'
    )
    
    return {
        'run_button': run_button
    }

def create_research_ui(scenarios):
    """
    Buat UI lengkap untuk evaluasi skenario penelitian.
    
    Args:
        scenarios: List skenario penelitian
        
    Returns:
        Dictionary berisi komponen UI untuk skenario penelitian
    """
    # Buat header dan deskripsi
    header = widgets.HTML("<h2>🔬 Evaluasi Skenario Penelitian</h2>")
    description = widgets.HTML("<p>Pilih skenario yang akan dievaluasi:</p>")
    
    # Buat checkboxes untuk skenario
    scenario_checkboxes = create_scenario_checkboxes(scenarios)
    
    # Buat tombol evaluasi
    controls = create_evaluation_controls()
    
    # Buat output area
    output_area = widgets.Output()
    
    # Gabungkan dalam layout utama
    checkboxes_layout = widgets.VBox(scenario_checkboxes)
    
    main_ui = widgets.VBox([
        header,
        description,
        checkboxes_layout,
        controls['run_button'],
        output_area
    ])
    
    # Return struktur UI dan komponen untuk handler
    return {
        'ui': main_ui,
        'output': output_area,
        'run_button': controls['run_button'],
        'scenario_checkboxes': scenario_checkboxes,
        'scenarios': scenarios  # Original scenarios list
    }