"""smartcash/ui/training/components/confusion_matrix.py

Komponen Confusion Matrix untuk UI Training.
"""

import plotly.graph_objs as go
from ipywidgets import VBox, Output, Accordion

class ConfusionMatrixAccordion:
    """Accordion interaktif untuk visualisasi confusion matrix"""
    
    def __init__(self):
        self.accordion = Accordion(
            titles=['🧮 Confusion Matrix'],
            selected_index=None
        )
        self.output = Output()
        self.figure = None
        self.accordion.children = [self.output]
        
    def update_matrix(self, cm, class_labels):
        """Update heatmap confusion matrix"""
        with self.output:
            self.output.clear_output(wait=True)
            
            # Normalisasi untuk persentase
            cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Buat heatmap
            self.figure = go.Figure(
                data=go.Heatmap(
                    z=cm_perc,
                    x=class_labels,
                    y=class_labels,
                    colorscale='Blues',
                    hoverongaps=False,
                    texttemplate="%{z:.1%}",
                    textfont={"size":10}
                )
            )
            
            self.figure.update_layout(
                title='Confusion Matrix (Normalized)',
                xaxis_title='Predicted',
                yaxis_title='Actual',
                autosize=True
            )
            
            # Tampilkan plot
            display(self.figure)
