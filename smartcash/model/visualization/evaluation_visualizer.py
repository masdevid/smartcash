# File: smartcash/model/visualization/evaluation_visualizer.py
# Author: Alfrida Sabar
# Deskripsi: Komponen visualisasi untuk hasil evaluasi model

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from smartcash.common.logger import SmartCashLogger, get_logger

class EvaluationVisualizer:
    """
    Komponen visualisasi untuk hasil evaluasi model.
    Digunakan untuk menghasilkan visualisasi perbandingan model, metrik, backbone, dll.
    """
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi visualizer evaluasi.
        
        Args:
            output_dir: Direktori output untuk visualisasi (opsional)
            logger: Logger kustom (opsional)
        """
        self.logger = logger or get_logger("evaluation_visualizer")
        
        # Output directory
        self.output_dir = Path(output_dir) if output_dir else Path("results/evaluation/plots")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.debug(f"🔧 EvaluationVisualizer diinisialisasi (output_dir={self.output_dir})")
    
    def create_all_plots(
        self,
        metrics_data: Union[pd.DataFrame, List[Dict]],
        prefix: str = "",
        **kwargs
    ) -> Dict[str, str]:
        """
        Buat semua visualisasi yang tersedia.
        
        Args:
            metrics_data: DataFrame atau list dicts berisi metrik evaluasi
            prefix: Awalan nama file (opsional)
            **kwargs: Parameter tambahan untuk plotting
            
        Returns:
            Dictionary berisi paths ke plots yang dibuat
        """
        # Konversi ke DataFrame jika list
        if isinstance(metrics_data, list):
            metrics_df = pd.DataFrame(metrics_data)
        else:
            metrics_df = metrics_data
            
        # Pastikan ada data yang valid
        if metrics_df.empty:
            self.logger.warning(f"⚠️ Tidak ada data metrik untuk visualisasi")
            return {}
            
        # Hasil plots
        plots = {}
        
        try:
            # Plot mAP dan F1 score comparison
            map_f1_path = self.plot_map_f1_comparison(metrics_df, prefix, **kwargs)
            if map_f1_path:
                plots['map_f1_comparison'] = map_f1_path
                
            # Plot inference time
            inference_path = self.plot_inference_time(metrics_df, prefix, **kwargs)
            if inference_path:
                plots['inference_time'] = inference_path
                
            # Plot backbone comparison
            backbone_path = self.plot_backbone_comparison(metrics_df, prefix, **kwargs)
            if backbone_path:
                plots['backbone_comparison'] = backbone_path
                
            # Plot test condition comparison
            condition_path = self.plot_condition_comparison(metrics_df, prefix, **kwargs)
            if condition_path:
                plots['condition_comparison'] = condition_path
                
            # Plot combined heatmap
            heatmap_path = self.plot_combined_heatmap(metrics_df, prefix, **kwargs)
            if heatmap_path:
                plots['combined_heatmap'] = heatmap_path
                
        except Exception as e:
            self.logger.error(f"❌ Gagal membuat visualisasi: {str(e)}")
            
        return plots
    
    def plot_map_f1_comparison(
        self,
        metrics_df: pd.DataFrame,
        prefix: str = "",
        **kwargs
    ) -> Optional[str]:
        """
        Buat plot perbandingan mAP dan F1 score antar model/skenario.
        
        Args:
            metrics_df: DataFrame berisi metrik evaluasi
            prefix: Awalan nama file (opsional)
            **kwargs: Parameter tambahan untuk plotting
            
        Returns:
            Path ke plot yang dibuat atau None jika gagal
        """
        try:
            # Setup figure
            plt.figure(figsize=(10, 6))
            
            # Tentukan kolom x (model atau skenario)
            if 'scenario' in metrics_df.columns:
                x_col = 'scenario'
            elif 'model' in metrics_df.columns:
                x_col = 'model'
            else:
                x_col = metrics_df.columns[0]  # Gunakan kolom pertama sebagai fallback
                
            # Sort data berdasarkan mAP
            metrics_df = metrics_df.sort_values('mAP', ascending=False)
            
            # Create a bar plot
            ax = sns.barplot(x=x_col, y='value', hue='metric', 
                           data=pd.melt(metrics_df, id_vars=[x_col], 
                                      value_vars=['mAP', 'F1'],
                                      var_name='metric', value_name='value'))
            
            plt.title(f'Perbandingan mAP dan F1 Score')
            plt.xlabel(x_col.capitalize())
            plt.ylabel('Nilai')
            plt.ylim(0, 1.0)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save plot
            filename = f"{prefix}_map_f1_comparison.png" if prefix else "map_f1_comparison.png"
            plot_path = self.output_dir / filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.debug(f"✅ Plot mAP/F1 dibuat: {plot_path}")
            return str(plot_path)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal membuat plot mAP/F1: {str(e)}")
            return None
    
    def plot_inference_time(
        self,
        metrics_df: pd.DataFrame,
        prefix: str = "",
        **kwargs
    ) -> Optional[str]:
        """
        Buat plot perbandingan waktu inferensi.
        
        Args:
            metrics_df: DataFrame berisi metrik evaluasi
            prefix: Awalan nama file (opsional)
            **kwargs: Parameter tambahan untuk plotting
            
        Returns:
            Path ke plot yang dibuat atau None jika gagal
        """
        try:
            # Periksa jika kolom inference_time ada
            if 'inference_time' not in metrics_df.columns:
                return None
                
            # Setup figure
            plt.figure(figsize=(10, 6))
            
            # Tentukan kolom x (model atau skenario)
            if 'scenario' in metrics_df.columns:
                x_col = 'scenario'
            elif 'model' in metrics_df.columns:
                x_col = 'model'
            else:
                x_col = metrics_df.columns[0]  # Gunakan kolom pertama sebagai fallback
                
            # Sort data berdasarkan inference time
            metrics_df = metrics_df.sort_values('inference_time')
            
            # Create a bar plot for inference time
            ax = sns.barplot(x=x_col, y='inference_time', data=metrics_df, palette='viridis')
            
            # Annotate bars with values
            for p in ax.patches:
                ax.annotate(f"{p.get_height():.1f} ms", 
                           (p.get_x() + p.get_width() / 2., p.get_height()), 
                           ha='center', va='bottom', rotation=0, 
                           xytext=(0, 5), textcoords='offset points')
            
            plt.title('Waktu Inferensi')
            plt.xlabel(x_col.capitalize())
            plt.ylabel('Waktu Inferensi (ms)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save plot
            filename = f"{prefix}_inference_time.png" if prefix else "inference_time.png"
            plot_path = self.output_dir / filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.debug(f"✅ Plot waktu inferensi dibuat: {plot_path}")
            return str(plot_path)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal membuat plot waktu inferensi: {str(e)}")
            return None
    
    def plot_backbone_comparison(
        self,
        metrics_df: pd.DataFrame,
        prefix: str = "",
        **kwargs
    ) -> Optional[str]:
        """
        Buat plot perbandingan backbone.
        
        Args:
            metrics_df: DataFrame berisi metrik evaluasi
            prefix: Awalan nama file (opsional)
            **kwargs: Parameter tambahan untuk plotting
            
        Returns:
            Path ke plot yang dibuat atau None jika gagal
        """
        try:
            # Periksa jika kolom description atau backbone ada
            if 'description' not in metrics_df.columns and 'backbone' not in metrics_df.columns:
                return None
                
            # Setup figure
            plt.figure(figsize=(8, 6))
            
            # Extract backbone info jika belum ada
            if 'backbone' not in metrics_df.columns and 'description' in metrics_df.columns:
                metrics_df['backbone'] = metrics_df['description'].apply(
                    lambda x: 'EfficientNet' if 'efficientnet' in str(x).lower() 
                    else 'CSPDarknet' if 'cspdarknet' in str(x).lower() or 'darknet' in str(x).lower() or 'csp' in str(x).lower() 
                    else 'Unknown'
                )
            
            # Filter hanya backbone yang dikenal
            backbone_df = metrics_df[metrics_df['backbone'] != 'Unknown']
            
            if backbone_df.empty:
                return None
                
            # Group by backbone and calculate averages
            backbone_agg = backbone_df.groupby('backbone').agg({
                'mAP': 'mean',
                'F1': 'mean',
                'inference_time': 'mean' if 'inference_time' in backbone_df.columns else 'count'
            }).reset_index()
            
            # Create a bar plot
            ax = sns.barplot(x='backbone', y='value', hue='metric', 
                           data=pd.melt(backbone_agg, id_vars=['backbone'], 
                                      value_vars=['mAP', 'F1'],
                                      var_name='metric', value_name='value'))
            
            plt.title('Perbandingan Backbone: EfficientNet vs CSPDarknet')
            plt.xlabel('Backbone')
            plt.ylabel('Nilai')
            plt.ylim(0, 1.0)
            plt.tight_layout()
            
            # Save plot
            filename = f"{prefix}_backbone_comparison.png" if prefix else "backbone_comparison.png"
            plot_path = self.output_dir / filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.debug(f"✅ Plot perbandingan backbone dibuat: {plot_path}")
            return str(plot_path)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal membuat plot perbandingan backbone: {str(e)}")
            return None
    
    def plot_condition_comparison(
        self,
        metrics_df: pd.DataFrame,
        prefix: str = "",
        **kwargs
    ) -> Optional[str]:
        """
        Buat plot perbandingan kondisi pengujian.
        
        Args:
            metrics_df: DataFrame berisi metrik evaluasi
            prefix: Awalan nama file (opsional)
            **kwargs: Parameter tambahan untuk plotting
            
        Returns:
            Path ke plot yang dibuat atau None jika gagal
        """
        try:
            # Periksa jika kolom description ada
            if 'description' not in metrics_df.columns and 'condition' not in metrics_df.columns:
                return None
                
            # Extract test condition jika belum ada
            if 'condition' not in metrics_df.columns and 'description' in metrics_df.columns:
                metrics_df['condition'] = metrics_df['description'].apply(
                    lambda x: 'Posisi Bervariasi' if 'posisi' in str(x).lower() 
                    else 'Pencahayaan Bervariasi' if 'pencahayaan' in str(x).lower() or 'lighting' in str(x).lower() 
                    else 'Other'
                )
            
            # Filter hanya kondisi yang dikenal
            condition_df = metrics_df[metrics_df['condition'] != 'Other']
            
            if condition_df.empty:
                return None
                
            # Setup figure
            plt.figure(figsize=(8, 6))
            
            # Group by condition and calculate averages
            condition_agg = condition_df.groupby('condition').agg({
                'mAP': 'mean',
                'F1': 'mean'
            }).reset_index()
            
            # Create a bar plot
            ax = sns.barplot(x='condition', y='value', hue='metric', 
                           data=pd.melt(condition_agg, id_vars=['condition'], 
                                      value_vars=['mAP', 'F1'],
                                      var_name='metric', value_name='value'))
            
            plt.title('Perbandingan Kondisi Pengujian')
            plt.xlabel('Kondisi')
            plt.ylabel('Nilai')
            plt.ylim(0, 1.0)
            plt.tight_layout()
            
            # Save plot
            filename = f"{prefix}_condition_comparison.png" if prefix else "condition_comparison.png"
            plot_path = self.output_dir / filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.debug(f"✅ Plot perbandingan kondisi pengujian dibuat: {plot_path}")
            return str(plot_path)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal membuat plot perbandingan kondisi: {str(e)}")
            return None
    
    def plot_combined_heatmap(
        self,
        metrics_df: pd.DataFrame,
        prefix: str = "",
        **kwargs
    ) -> Optional[str]:
        """
        Buat heatmap kombinasi backbone dan kondisi pengujian.
        
        Args:
            metrics_df: DataFrame berisi metrik evaluasi
            prefix: Awalan nama file (opsional)
            **kwargs: Parameter tambahan untuk plotting
            
        Returns:
            Path ke plot yang dibuat atau None jika gagal
        """
        try:
            # Periksa jika kolom backbone dan condition ada
            has_backbone = 'backbone' in metrics_df.columns
            has_condition = 'condition' in metrics_df.columns
            has_description = 'description' in metrics_df.columns
            
            if not has_backbone and not has_description:
                return None
                
            if not has_condition and not has_description:
                return None
                
            # Extract backbone info jika belum ada
            if not has_backbone and has_description:
                metrics_df['backbone'] = metrics_df['description'].apply(
                    lambda x: 'EfficientNet' if 'efficientnet' in str(x).lower() 
                    else 'CSPDarknet' if 'cspdarknet' in str(x).lower() or 'darknet' in str(x).lower() or 'csp' in str(x).lower()
                    else 'Unknown'
                )
            
            # Extract test condition jika belum ada
            if not has_condition and has_description:
                metrics_df['condition'] = metrics_df['description'].apply(
                    lambda x: 'Posisi Bervariasi' if 'posisi' in str(x).lower() 
                    else 'Pencahayaan Bervariasi' if 'pencahayaan' in str(x).lower() or 'lighting' in str(x).lower() 
                    else 'Other'
                )
            
            # Filter rows
            filtered_df = metrics_df[(metrics_df['backbone'] != 'Unknown') & (metrics_df['condition'] != 'Other')]
            
            if filtered_df.empty:
                return None
                
            # Setup figure
            plt.figure(figsize=(12, 8))
            
            # Create pivot table
            try:
                heatmap_df = filtered_df.pivot_table(
                    index='backbone', 
                    columns='condition', 
                    values='mAP',
                    aggfunc='mean'
                )
                
                # Create heatmap
                ax = sns.heatmap(heatmap_df, annot=True, cmap='viridis', fmt=".3f", vmin=0, vmax=1)
                
                plt.title('mAP Score berdasarkan Backbone dan Kondisi')
                plt.tight_layout()
                
                # Save plot
                filename = f"{prefix}_combined_heatmap.png" if prefix else "combined_heatmap.png"
                plot_path = self.output_dir / filename
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                self.logger.debug(f"✅ Heatmap kombinasi dibuat: {plot_path}")
                return str(plot_path)
            except Exception as e:
                self.logger.warning(f"⚠️ Gagal membuat pivot table: {str(e)}")
                return None
            
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal membuat heatmap kombinasi: {str(e)}")
            return None