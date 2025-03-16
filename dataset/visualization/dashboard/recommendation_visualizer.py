"""
File: smartcash/dataset/visualization/helpers/recommendation_visualizer.py
Deskripsi: Helper untuk visualisasi rekomendasi dataset
"""

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any


class RecommendationVisualizer:
    """Helper untuk visualisasi rekomendasi dataset."""
    
    def __init__(self):
        """Inisialisasi RecommendationVisualizer."""
        # Setup style
        self.palette = sns.color_palette("viridis", 12)
        self.accent_palette = sns.color_palette("Set2", 8)
    
    def plot_recommendations(
        self, 
        ax, 
        recommendations: List[str],
        title: str = "Rekomendasi Dataset"
    ) -> None:
        """
        Plot visualisasi rekomendasi.
        
        Args:
            ax: Axes untuk plot
            recommendations: List rekomendasi
            title: Judul visualisasi
        """
        if not recommendations:
            ax.text(0.5, 0.5, "Tidak ada rekomendasi", ha='center', va='center')
            ax.set_title(title)
            ax.axis('off')
            return
            
        # Batasi jumlah rekomendasi yang ditampilkan
        max_recs = 5
        displayed_recommendations = recommendations[:max_recs]
        remaining = len(recommendations) - max_recs
        
        # Buat text untuk ditampilkan
        text = ""
        for i, rec in enumerate(displayed_recommendations, 1):
            text += f"{i}. {rec}\n\n"
            
        if remaining > 0:
            text += f"... dan {remaining} rekomendasi lainnya."
        
        # Plot text dalam box
        text_box = ax.text(
            0.5, 0.5, text, 
            ha='center', va='center', 
            bbox=dict(boxstyle="round,pad=0.5", facecolor='#f5f5f5', alpha=0.5),
            wrap=True
        )
        
        # Setting untuk text box
        text_box.set_fontsize(10)
        
        # Styling
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
    
    def plot_recommendations_summary(self, ax, report: Dict[str, Any]) -> None:
        """
        Plot ringkasan rekomendasi dari laporan.
        
        Args:
            ax: Axes untuk plot
            report: Data laporan dataset
        """
        # Extract recommendations
        recommendations = report.get('recommendations', [])
        
        if not recommendations:
            ax.text(0.5, 0.5, "Tidak ada rekomendasi", ha='center', va='center')
            ax.set_title('Rekomendasi Utama')
            ax.axis('off')
            return
            
        # Create a text box with recommendations
        recommendations_text = "Rekomendasi:\n\n"
        for i, rec in enumerate(recommendations[:3], 1):  # Limit to top 3
            recommendations_text += f"{i}. {rec}\n"
            
        if len(recommendations) > 3:
            recommendations_text += f"\n...dan {len(recommendations) - 3} rekomendasi lainnya."
        
        # Plot text
        ax.text(0.5, 0.5, recommendations_text, ha='center', va='center', 
               bbox=dict(boxstyle="round,pad=0.5", facecolor='#f5f5f5', alpha=0.5),
               wrap=True, fontsize=10)
        
        # Remove axis
        ax.axis('off')
        ax.set_title('Rekomendasi Utama')