# File: smartcash/handlers/dataset/visualizations/layer_distribution_visualizer.py
# Author: Alfrida Sabar
# Deskripsi: Visualisasi distribusi layer dalam dataset

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from smartcash.utils.logger import get_logger
from smartcash.handlers.dataset.visualizations.visualization_base import VisualizationBase


class LayerDistributionVisualizer(VisualizationBase):
    """
    Visualisasi khusus untuk distribusi layer dalam dataset SmartCash.
    """
    
    def __init__(
        self,
        config: Dict,
        data_dir: str,
        output_dir: Optional[str] = None,
        logger=None
    ):
        """
        Inisialisasi visualizer distribusi layer.
        
        Args:
            config: Konfigurasi dataset
            data_dir: Direktori dataset
            output_dir: Direktori output untuk visualisasi
            logger: Logger kustom (opsional)
        """
        super().__init__(
            config=config,
            data_dir=data_dir,
            output_dir=output_dir,
            logger=logger or get_logger("layer_distribution_visualizer")
        )
        
        self.logger.info(f"📊 LayerDistributionVisualizer diinisialisasi: {self.data_dir}")
    
    def visualize_layer_distribution(
        self,
        split: str,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6),
        show_percentages: bool = True,
        pie_chart: bool = True
    ) -> str:
        """
        Visualisasikan distribusi layer dalam dataset.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            save_path: Path untuk menyimpan visualisasi
            figsize: Ukuran figur
            show_percentages: Tampilkan persentase 
            pie_chart: Gunakan pie chart alih-alih bar chart
            
        Returns:
            Path ke file visualisasi yang disimpan
        """
        self.logger.info(f"📊 Memvisualisasikan distribusi layer untuk split: {split}")
        
        try:
            # Analisis dataset untuk mendapatkan distribusi layer
            analysis = self.explorer.analyze_dataset(split, sample_size=0)
            
            # Pastikan hasil analisis memiliki data layer
            if 'layer_balance' not in analysis:
                raise ValueError(f"Hasil analisis tidak memiliki data distribusi layer")
                
            layer_balance = analysis['layer_balance']
            layer_percentages = layer_balance.get('layer_percentages', {})
            
            if not layer_percentages:
                raise ValueError(f"Tidak ada data persentase layer dalam hasil analisis")
            
            # Konversi ke DataFrame untuk visualisasi
            df = pd.DataFrame({
                'layer': list(layer_percentages.keys()),
                'persentase': list(layer_percentages.values())
            })
            
            # Urutkan berdasarkan persentase (descending)
            df = df.sort_values('persentase', ascending=False)
            
            # Buat visualisasi
            fig, ax = plt.subplots(figsize=figsize)
            
            if pie_chart:
                # Buat pie chart
                wedges, texts, autotexts = ax.pie(
                    df['persentase'],
                    labels=df['layer'],
                    autopct='%1.1f%%' if show_percentages else None,
                    startangle=90,
                    colors=self.color_palette_alt[:len(df)],
                    wedgeprops={'edgecolor': 'w', 'linewidth': 1}
                )
                
                # Customisasi font
                for text in texts:
                    text.set_fontsize(12)
                    
                if show_percentages:
                    for autotext in autotexts:
                        autotext.set_fontsize(11)
                        autotext.set_fontweight('bold')
                
                # Equal aspect ratio menjaga pie tetap lingkaran
                ax.axis('equal')
                plt.title(f'Distribusi Layer - {split.capitalize()}')
            else:
                # Buat bar plot
                bars = sns.barplot(
                    x='layer',
                    y='persentase',
                    data=df,
                    palette=self.color_palette_alt[:len(df)],
                    ax=ax
                )
                
                # Tambahkan label persentase
                if show_percentages:
                    for i, p in enumerate(bars.patches):
                        percentage = f"{p.get_height():.1f}%"
                        ax.annotate(
                            percentage,
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center',
                            va='bottom',
                            fontsize=12,
                            fontweight='bold'
                        )
                
                # Tambahkan detail plot
                ax.set_xlabel('Layer')
                ax.set_ylabel('Persentase (%)')
                ax.set_title(f'Distribusi Layer - {split.capitalize()}')
            
            # Tambahkan info ke judul
            plt.suptitle(
                f"Ketidakseimbangan: {layer_balance.get('imbalance_score', 0):.1f}/10",
                fontsize=12,
                y=0.02
            )
            
            # Tight layout
            plt.tight_layout()
            
            # Simpan plot jika path disediakan
            if save_path is None:
                save_path = str(self.output_dir / f"{split}_layer_distribution.png")
                
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"✅ Visualisasi distribusi layer tersimpan di: {save_path}")
            
            return save_path
            
        except Exception as e:
            self.logger.error(f"❌ Gagal membuat visualisasi distribusi layer: {str(e)}")
            # Return path dummy jika gagal
            return ""
    
    def compare_layer_distribution(
        self,
        splits: List[str] = ['train', 'valid', 'test'],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8)
    ) -> str:
        """
        Bandingkan distribusi layer antar split dataset.
        
        Args:
            splits: List split dataset yang akan dibandingkan
            save_path: Path untuk menyimpan visualisasi
            figsize: Ukuran figur
            
        Returns:
            Path ke file visualisasi yang disimpan
        """
        self.logger.info(f"📊 Membandingkan distribusi layer antar split: {', '.join(splits)}")
        
        try:
            # Kumpulkan data dari semua split
            combined_data = []
            
            for split in splits:
                try:
                    # Analisis dataset
                    analysis = self.explorer.analyze_dataset(split, sample_size=0)
                    
                    # Ekstrak distribusi layer
                    if 'layer_balance' not in analysis:
                        self.logger.warning(f"⚠️ Split {split} tidak memiliki data distribusi layer")
                        continue
                        
                    layer_balance = analysis['layer_balance']
                    percentages = layer_balance.get('layer_percentages', {})
                    
                    # Tambahkan ke data gabungan
                    for name, percentage in percentages.items():
                        combined_data.append({
                            'split': split,
                            'layer': name,
                            'persentase': percentage
                        })
                except Exception as e:
                    self.logger.warning(f"⚠️ Gagal menganalisis split {split}: {str(e)}")
            
            if not combined_data:
                raise ValueError("Tidak ada data valid untuk visualisasi perbandingan")
            
            # Konversi ke DataFrame
            df = pd.DataFrame(combined_data)
            
            # Buat visualisasi
            plt.figure(figsize=figsize)
            
            # Gunakan barplot yang dikelompokkan berdasarkan layer
            ax = sns.barplot(
                x='layer',
                y='persentase',
                hue='split',
                data=df,
                palette=self.color_palette[:len(splits)]
            )
            
            # Tambahkan detail plot
            plt.title(f'Perbandingan Distribusi Layer Antar Split')
            plt.xlabel('Layer')
            plt.ylabel('Persentase (%)')
            
            # Legend
            plt.legend(title='Split')
            
            # Tight layout
            plt.tight_layout()
            
            # Simpan plot
            if save_path is None:
                save_path = str(self.output_dir / "comparison_layer_distribution.png")
                
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"✅ Perbandingan distribusi layer tersimpan di: {save_path}")
            
            return save_path
            
        except Exception as e:
            self.logger.error(f"❌ Gagal membuat perbandingan distribusi layer: {str(e)}")
            # Return path dummy jika gagal
            return ""
    
    def create_class_layer_heatmap(
        self,
        split: str,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10),
        normalize: bool = True,
        cmap: str = "YlGnBu"
    ) -> str:
        """
        Visualisasikan heatmap korelasi antara kelas dan layer.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            save_path: Path untuk menyimpan visualisasi
            figsize: Ukuran figur
            normalize: Normalisasi nilai heatmap
            cmap: Colormap untuk heatmap
            
        Returns:
            Path ke file visualisasi yang disimpan
        """
        self.logger.info(f"🔥 Membuat heatmap kelas-layer untuk split: {split}")
        
        try:
            # Analisis dataset untuk mendapatkan statistik kelas per layer
            analysis = self.explorer.analyze_dataset(split, sample_size=0)
            
            # Pastikan hasil analisis memiliki data yang diperlukan
            if 'class_layer_matrix' not in analysis:
                raise ValueError(f"Hasil analisis tidak memiliki data matriks kelas-layer")
                
            class_layer_matrix = analysis['class_layer_matrix']
            
            # Konversi ke DataFrame untuk visualisasi
            matrix_df = pd.DataFrame(class_layer_matrix)
            
            # Normalisasi jika diminta
            if normalize and matrix_df.values.sum() > 0:
                matrix_df = matrix_df / matrix_df.values.sum() * 100
            
            # Buat visualisasi
            plt.figure(figsize=figsize)
            
            # Buat heatmap
            ax = sns.heatmap(
                matrix_df,
                annot=True,
                fmt=".1f" if normalize else "d",
                cmap=cmap,
                linewidths=0.5,
                cbar_kws={'label': 'Persentase (%)' if normalize else 'Jumlah Objek'}
            )
            
            # Tambahkan detail plot
            plt.title(f'Heatmap Kelas-Layer - {split.capitalize()}')
            plt.tight_layout()
            
            # Simpan plot jika path disediakan
            if save_path is None:
                save_path = str(self.output_dir / f"{split}_class_layer_heatmap.png")
                
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"✅ Heatmap kelas-layer tersimpan di: {save_path}")
            
            return save_path
            
        except Exception as e:
            self.logger.error(f"❌ Gagal membuat heatmap kelas-layer: {str(e)}")
            # Return path dummy jika gagal
            return ""