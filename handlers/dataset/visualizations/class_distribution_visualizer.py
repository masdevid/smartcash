# File: smartcash/handlers/dataset/visualizations/class_distribution_visualizer.py
# Author: Alfrida Sabar
# Deskripsi: Visualisasi distribusi kelas dalam dataset

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from smartcash.utils.logger import get_logger
from smartcash.handlers.dataset.core.dataset_explorer import DatasetExplorer
from smartcash.handlers.dataset.visualizations.visualization_base import VisualizationBase


class ClassDistributionVisualizer(VisualizationBase):
    """
    Visualisasi khusus untuk distribusi kelas dalam dataset SmartCash.
    """
    
    def __init__(
        self,
        config: Dict,
        data_dir: str,
        output_dir: Optional[str] = None,
        logger=None
    ):
        """
        Inisialisasi visualizer distribusi kelas.
        
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
            logger=logger or get_logger("class_distribution_visualizer")
        )
        
        self.logger.info(f"📊 ClassDistributionVisualizer diinisialisasi: {self.data_dir}")
    
    def visualize_class_distribution(
        self,
        split: str,
        save_path: Optional[str] = None,
        top_n: int = 10,
        figsize: Tuple[int, int] = (12, 8),
        show_percentages: bool = True
    ) -> str:
        """
        Visualisasikan distribusi kelas dalam dataset.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            save_path: Path untuk menyimpan visualisasi
            top_n: Jumlah kelas teratas untuk ditampilkan
            figsize: Ukuran figur
            show_percentages: Tampilkan persentase pada bar
            
        Returns:
            Path ke file visualisasi yang disimpan
        """
        self.logger.info(f"📊 Memvisualisasikan distribusi kelas untuk split: {split}")
        
        try:
            # Analisis dataset untuk mendapatkan distribusi kelas
            analysis = self.explorer.analyze_dataset(split, sample_size=0)
            
            # Pastikan hasil analisis memiliki data kelas
            if 'class_balance' not in analysis:
                raise ValueError(f"Hasil analisis tidak memiliki data distribusi kelas")
                
            class_balance = analysis['class_balance']
            class_percentages = class_balance.get('class_percentages', {})
            
            if not class_percentages:
                raise ValueError(f"Tidak ada data persentase kelas dalam hasil analisis")
            
            # Konversi ke DataFrame untuk visualisasi
            df = pd.DataFrame({
                'kelas': list(class_percentages.keys()),
                'persentase': list(class_percentages.values())
            })
            
            # Urutkan berdasarkan persentase (descending)
            df = df.sort_values('persentase', ascending=False)
            
            # Ambil N kelas teratas
            if len(df) > top_n:
                df_top = df.head(top_n)
                df_other = pd.DataFrame({
                    'kelas': ['Lainnya'],
                    'persentase': [df['persentase'][top_n:].sum()]
                })
                df = pd.concat([df_top, df_other], ignore_index=True)
            
            # Buat visualisasi
            fig, ax = plt.subplots(figsize=figsize)
            
            # Buat bar plot
            bars = sns.barplot(
                x='kelas',
                y='persentase',
                data=df,
                palette=self.color_palette[:len(df)],
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
            ax.set_xlabel('Kelas')
            ax.set_ylabel('Persentase (%)')
            ax.set_title(f'Distribusi Kelas - {split.capitalize()}')
            
            # Rotasi label x
            plt.xticks(rotation=45, ha='right')
            
            # Tight layout
            plt.tight_layout()
            
            # Simpan plot jika path disediakan
            if save_path is None:
                save_path = str(self.output_dir / f"{split}_class_distribution.png")
                
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"✅ Visualisasi distribusi kelas tersimpan di: {save_path}")
            
            return save_path
            
        except Exception as e:
            self.logger.error(f"❌ Gagal membuat visualisasi distribusi kelas: {str(e)}")
            # Return path dummy jika gagal
            return ""
    
    def compare_class_distribution(
        self,
        splits: List[str] = ['train', 'valid', 'test'],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 10),
        top_n: int = 8
    ) -> str:
        """
        Bandingkan distribusi kelas antar split dataset.
        
        Args:
            splits: List split dataset yang akan dibandingkan
            save_path: Path untuk menyimpan visualisasi
            figsize: Ukuran figur
            top_n: Jumlah kelas teratas untuk ditampilkan
            
        Returns:
            Path ke file visualisasi yang disimpan
        """
        self.logger.info(f"📊 Membandingkan distribusi kelas antar split: {', '.join(splits)}")
        
        try:
            # Kumpulkan data dari semua split
            combined_data = []
            
            for split in splits:
                try:
                    # Analisis dataset
                    analysis = self.explorer.analyze_dataset(split, sample_size=0)
                    
                    # Ekstrak distribusi kelas
                    if 'class_balance' not in analysis:
                        self.logger.warning(f"⚠️ Split {split} tidak memiliki data distribusi kelas")
                        continue
                        
                    class_balance = analysis['class_balance']
                    percentages = class_balance.get('class_percentages', {})
                    
                    # Tambahkan ke data gabungan
                    for name, percentage in percentages.items():
                        combined_data.append({
                            'split': split,
                            'kelas': name,
                            'persentase': percentage
                        })
                except Exception as e:
                    self.logger.warning(f"⚠️ Gagal menganalisis split {split}: {str(e)}")
            
            if not combined_data:
                raise ValueError("Tidak ada data valid untuk visualisasi perbandingan")
            
            # Konversi ke DataFrame
            df = pd.DataFrame(combined_data)
            
            # Filter hanya top N kelas dengan persentase tertinggi (aggregate antar split)
            if len(df['kelas'].unique()) > top_n:
                top_classes = df.groupby('kelas')['persentase'].mean().nlargest(top_n).index.tolist()
                df = df[df['kelas'].isin(top_classes)]
            
            # Buat visualisasi
            plt.figure(figsize=figsize)
            
            # Gunakan barplot yang dikelompokkan berdasarkan kelas
            ax = sns.barplot(
                x='kelas',
                y='persentase',
                hue='split',
                data=df,
                palette=self.color_palette[:len(splits)]
            )
            
            # Tambahkan detail plot
            plt.title(f'Perbandingan Distribusi Kelas Antar Split')
            plt.xlabel('Kelas')
            plt.ylabel('Persentase (%)')
            
            # Rotasi label sumbu-x jika terlalu banyak
            if len(df['kelas'].unique()) > 5:
                plt.xticks(rotation=45, ha='right')
            
            # Legend
            plt.legend(title='Split')
            
            # Tight layout
            plt.tight_layout()
            
            # Simpan plot
            if save_path is None:
                save_path = str(self.output_dir / "comparison_class_distribution.png")
                
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"✅ Perbandingan distribusi kelas tersimpan di: {save_path}")
            
            return save_path
            
        except Exception as e:
            self.logger.error(f"❌ Gagal membuat perbandingan distribusi kelas: {str(e)}")
            # Return path dummy jika gagal
            return ""