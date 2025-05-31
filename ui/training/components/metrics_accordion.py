"""
File: smartcash/ui/training/components/metrics_accordion.py
Deskripsi: Komponen accordion untuk menampilkan metrik dan chart training
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_metrics_accordion(height='300px') -> Dict[str, Any]:
    """Create metrics accordion untuk menampilkan chart dan metrics"""
    from smartcash.ui.components.accordion_factory import create_accordion
    
    # Chart output
    chart_output = widgets.Output(
        layout=widgets.Layout(
            width='100%',
            min_height='200px',
            max_height='400px',
            overflow_y='auto',
            margin='10px 0',
            padding='10px'
        )
    )
    
    # Metrics output
    metrics_output = widgets.Output(
        layout=widgets.Layout(
            width='100%',
            min_height='50px',
            max_height='150px',
            overflow_y='auto',
            margin='5px 0',
            padding='10px'
        )
    )
    
    # Create accordion
    metrics_container = widgets.VBox([chart_output, metrics_output])
    
    # Membuat dengan format yang sesuai dengan create_accordion
    accordion = create_accordion(
        [('📈 Metrik & Visualisasi', metrics_container)],
        selected_index=0  # Open by default
    )
    
    # Atur layout tambahan
    accordion.layout.height = height
    
    return {
        'metrics_accordion': accordion,
        'chart_output': chart_output,
        'metrics_output': metrics_output
    }

def update_metrics_chart(chart_output: widgets.Output, chart_data: Any) -> None:
    """Update chart dengan data baru"""
    import matplotlib.pyplot as plt
    
    with chart_output:
        chart_output.clear_output(wait=True)
        if chart_data is not None:
            plt.figure(figsize=(10, 6))
            # Render chart sesuai dengan data
            if isinstance(chart_data, dict):
                for key, values in chart_data.items():
                    if isinstance(values, list) and len(values) > 0:
                        plt.plot(values, label=key)
                plt.legend()
                plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

def update_metrics_text(metrics_output: widgets.Output, metrics_data: Dict[str, Any]) -> None:
    """Update metrics text dengan data baru"""
    from IPython.display import display, HTML
    
    with metrics_output:
        metrics_output.clear_output(wait=True)
        if metrics_data:
            metrics_html = """
            <div style="display: flex; flex-wrap: wrap; gap: 10px; padding: 5px;">
            """
            
            for key, value in metrics_data.items():
                if key.lower() in ['loss', 'val_loss']:
                    color = '#d32f2f'  # Red for loss
                elif 'accuracy' in key.lower() or 'precision' in key.lower() or 'recall' in key.lower() or 'map' in key.lower():
                    color = '#2e7d32'  # Green for accuracy metrics
                else:
                    color = '#1976d2'  # Blue for others
                
                metrics_html += f"""
                <div style="flex: 1; min-width: 120px; padding: 8px; background: rgba({','.join(map(str, [int(int(color[1:3], 16)), int(color[3:5], 16), int(color[5:7], 16)]))}, 0.1); border-radius: 4px;">
                    <div style="font-size: 12px; color: #555;">{key}</div>
                    <div style="font-size: 16px; font-weight: 600; color: {color};">{value:.4f if isinstance(value, float) else value}</div>
                </div>
                """
            
            metrics_html += "</div>"
            display(HTML(metrics_html))
