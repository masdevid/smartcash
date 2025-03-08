# File: smartcash/utils/model_visualizer.py
# Author: Alfrida Sabar
# Deskripsi: Modul untuk visualisasi hasil evaluasi model dengan konteks yang lebih baik

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from IPython.display import display, HTML, clear_output

from smartcash.utils.logger import SmartCashLogger, get_logger

class ModelVisualizer:
    """Visualizer untuk hasil evaluasi model dengan grafik dan tabel informatif."""
    
    def __init__(
        self,
        logger: Optional[SmartCashLogger] = None,
        output_dir: str = "runs/evaluation"
    ):
        """
        Inisialisasi visualizer.
        
        Args:
            logger: Logger untuk logging (opsional)
            output_dir: Direktori output untuk menyimpan visualisasi
        """
        self.logger = logger or get_logger("model_visualizer")
        self.output_dir = output_dir
        
        # Setup style untuk visualisasi yang konsisten
        self.setup_style()
    
    def setup_style(self):
        """Setup style untuk visualisasi yang konsisten."""
        try:
            # Set style seaborn dengan palette yang cocok untuk color blind
            sns.set_style("whitegrid")
            plt.rcParams['figure.figsize'] = (12, 8)
            plt.rcParams['axes.titlesize'] = 14
            plt.rcParams['axes.labelsize'] = 12
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal mengatur style: {str(e)}")
    
    def visualize_evaluation_results(
        self,
        results: Dict[str, Any],
        title: str = "Hasil Evaluasi Model",
        show_confusion_matrix: bool = True,
        show_class_metrics: bool = True,
        save_plots: bool = False
    ):
        """
        Visualisasikan hasil evaluasi model.
        
        Args:
            results: Dict hasil evaluasi dari EvaluationHandler
            title: Judul untuk visualisasi
            show_confusion_matrix: Tampilkan confusion matrix
            show_class_metrics: Tampilkan metrik per kelas
            save_plots: Simpan plot ke file
        """
        if not results:
            self.logger.warning("⚠️ Tidak ada hasil evaluasi untuk divisualisasikan")
            return
        
        try:
            # Ekstrak metrik dasar
            metrics = results.get('metrics', {})
            
            # Ubah nilai ke persentase untuk tampilan yang lebih baik
            accuracy = metrics.get('accuracy', 0) * 100
            precision = metrics.get('precision', 0) * 100
            recall = metrics.get('recall', 0) * 100
            f1 = metrics.get('f1', 0) * 100
            mAP = metrics.get('mAP', 0) * 100
            inference_time = metrics.get('inference_time', 0)
            
            # Tampilkan judul
            display(HTML(f"<h2>{title}</h2>"))
            
            # 1. Plot metrik utama
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1.1 Metrik akurasi (barplot)
            ax1 = axes[0, 0]
            metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'mAP']
            metric_values = [accuracy, precision, recall, f1, mAP]
            
            bars = ax1.bar(
                metric_names, 
                metric_values,
                color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']
            )
            
            ax1.set_title('Metrik Evaluasi (%)')
            ax1.set_xlabel('Metrik')
            ax1.set_ylabel('Nilai (%)')
            ax1.set_ylim(0, 105)
            ax1.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Tambahkan nilai di atas bar
            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                ax1.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 1,
                    f"{value:.1f}%",
                    ha='center', 
                    va='bottom',
                    fontweight='bold'
                )
            
            # 1.2 Confusion matrix (heatmap)
            ax2 = axes[0, 1]
            
            if 'confusion_matrix' in metrics and show_confusion_matrix:
                cm = metrics['confusion_matrix']
                try:
                    # Ambil class names dari konfigurasi jika tersedia
                    class_names = results.get('class_names', [])
                    
                    # Fallback ke indeks jika class_names tidak tersedia atau tidak cocok
                    if not class_names or len(class_names) != cm.shape[0]:
                        class_names = [str(i) for i in range(cm.shape[0])]
                    
                    # Normalisasi confusion matrix untuk persentase
                    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                    cm_norm = np.nan_to_num(cm_norm)  # Handle division by zero
                    
                    # Plot heatmap
                    sns.heatmap(
                        cm_norm, 
                        annot=cm,  # Tampilkan nilai asli sebagai anotasi
                        fmt='d', 
                        cmap='Blues',
                        xticklabels=class_names,
                        yticklabels=class_names,
                        ax=ax2
                    )
                    
                    ax2.set_title('Confusion Matrix')
                    ax2.set_ylabel('True Label')
                    ax2.set_xlabel('Predicted Label')
                    
                    # Rotasi label untuk readability
                    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                except Exception as e:
                    self.logger.warning(f"⚠️ Gagal membuat confusion matrix: {str(e)}")
                    ax2.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
            else:
                ax2.text(0.5, 0.5, "Confusion Matrix tidak tersedia", ha='center', va='center')
                ax2.set_title('Confusion Matrix')
            
            # 1.3 Performance metrics (waktu inferensi dan FPS)
            ax3 = axes[1, 0]
            
            # Tambahkan inferensi per device jika tersedia
            device_times = []
            device_labels = []
            
            # Default inference time
            device_times.append(inference_time)
            device_labels.append('GPU' if results.get('gpu_inference', True) else 'CPU')
            
            # Tambahkan CPU inference jika tersedia
            if 'cpu_inference_time' in metrics:
                device_times.append(metrics['cpu_inference_time'])
                device_labels.append('CPU')
                
            # Calculate FPS untuk semua devices
            fps_values = [1000 / max(t, 0.001) for t in device_times]  # Convert ms to FPS
            
            # Bar plot untuk inference time
            time_bars = ax3.bar(
                device_labels,
                device_times,
                color=['#3498db', '#e74c3c'][:len(device_labels)],
                alpha=0.7
            )
            
            ax3.set_title('Waktu Inferensi (ms)')
            ax3.set_xlabel('Device')
            ax3.set_ylabel('Waktu (ms)', color='#3498db')
            ax3.tick_params(axis='y', labelcolor='#3498db')
            ax3.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add FPS on secondary y-axis
            ax3b = ax3.twinx()
            ax3b.set_ylabel('FPS', color='#e74c3c')
            ax3b.plot(device_labels, fps_values, 'ro-', linewidth=2)
            ax3b.tick_params(axis='y', labelcolor='#e74c3c')
            
            # Tambahkan nilai di atas bar
            for i, (bar, time_val, fps_val) in enumerate(zip(time_bars, device_times, fps_values)):
                height = bar.get_height()
                ax3.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 0.1,
                    f"{time_val:.1f} ms",
                    ha='center',
                    va='bottom',
                    color='#3498db',
                    fontweight='bold'
                )
                ax3b.text(
                    i,
                    fps_val + 1,
                    f"{fps_val:.1f} FPS",
                    ha='center',
                    va='bottom',
                    color='#e74c3c',
                    fontweight='bold'
                )
            
            # 1.4 Diagram radar untuk metrik per kelas (opsional)
            ax4 = axes[1, 1]
            
            # Persiapkan metrik per kelas jika tersedia
            class_metrics = {}
            for key, value in metrics.items():
                if key.startswith('precision_cls_') or key.startswith('recall_cls_') or key.startswith('f1_cls_'):
                    parts = key.split('_')
                    metric_type = parts[0]
                    class_id = int(parts[-1])
                    
                    if class_id not in class_metrics:
                        class_metrics[class_id] = {}
                    
                    class_metrics[class_id][metric_type] = value
            
            if class_metrics and len(class_metrics) >= 3:
                # Coba buat radar chart jika ada cukup kelas
                try:
                    self._create_radar_chart(ax4, class_metrics, class_names)
                except Exception as e:
                    self.logger.warning(f"⚠️ Gagal membuat radar chart: {str(e)}")
                    ax4.text(0.5, 0.5, "Radar chart memerlukan minimal 3 kelas", ha='center', va='center')
            else:
                # Tambahkan placeholder text
                ax4.text(0.5, 0.5, "Metrik per kelas tidak tersedia", ha='center', va='center')
                ax4.set_title('Metrik Per Kelas')
            
            # Finalisasi layout
            plt.tight_layout()
            
            # Tampilkan plot
            plt.show()
            
            # Simpan plot jika diminta
            if save_plots:
                try:
                    import os
                    os.makedirs(self.output_dir, exist_ok=True)
                    fig.savefig(f"{self.output_dir}/evaluation_metrics.png", dpi=300, bbox_inches='tight')
                    self.logger.info(f"✅ Plot disimpan: {self.output_dir}/evaluation_metrics.png")
                except Exception as e:
                    self.logger.warning(f"⚠️ Gagal menyimpan plot: {str(e)}")
            
            # 2. Tampilkan metrik per kelas jika tersedia
            if show_class_metrics and class_metrics:
                self._show_class_metrics(class_metrics, class_names, save_plots)
            
            # 3. Tampilkan ringkasan informasi model
            self._show_model_summary(results, metrics, save_plots)
            
            # 4. Tampilkan contoh deteksi jika tersedia
            self._show_sample_detections(results, save_plots)
            
        except Exception as e:
            self.logger.error(f"❌ Error saat visualisasi hasil evaluasi: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _create_radar_chart(self, ax, class_metrics, class_names):
        """
        Buat radar chart untuk metrik per kelas.
        
        Args:
            ax: Matplotlib axes untuk plot
            class_metrics: Metrik per kelas
            class_names: Nama kelas
        """
        # Set polar projection
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Tentukan metrik yang akan ditampilkan
        metrics = ['precision', 'recall', 'f1']
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Tutup polygon
        
        # Label chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(['Precision', 'Recall', 'F1-Score'])
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
        ax.set_ylim(0, 1)
        ax.set_title('Metrik Per Kelas')
        
        # Plot data untuk setiap kelas
        for class_id, metrics_dict in class_metrics.items():
            # Skip jika tidak semua metrik tersedia
            if not all(m in metrics_dict for m in metrics):
                continue
                
            # Get class name
            class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
            
            # Get values
            values = [metrics_dict[m] for m in metrics]
            values += values[:1]  # Tutup polygon
            
            # Plot
            ax.plot(angles, values, linewidth=1, linestyle='solid', label=class_name)
            ax.fill(angles, values, alpha=0.1)
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    def _show_class_metrics(self, class_metrics, class_names, save_plots=False):
        """
        Tampilkan metrik per kelas dalam bentuk tabel dan heatmap.
        
        Args:
            class_metrics: Metrik per kelas
            class_names: Nama kelas
            save_plots: Simpan plot ke file
        """
        # Persiapkan data untuk tabel
        class_data = []
        
        for class_id, metrics_dict in class_metrics.items():
            # Get class name
            class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
            
            # Tambahkan ke data
            class_data.append({
                'Class': class_name,
                'Precision': metrics_dict.get('precision', 0) * 100,
                'Recall': metrics_dict.get('recall', 0) * 100,
                'F1-Score': metrics_dict.get('f1', 0) * 100
            })
        
        # Buat DataFrame
        class_df = pd.DataFrame(class_data)
        
        # Tampilkan tabel
        display(HTML("<h3>📊 Metrik per Kelas</h3>"))
        display(class_df)
        
        # Visualisasikan dengan heatmap
        plt.figure(figsize=(10, len(class_data) * 0.5 + 2))
        
        # Persiapkan data untuk heatmap
        metrics_heatmap = class_df.set_index('Class')
        
        # Buat heatmap
        sns.heatmap(
            metrics_heatmap, 
            annot=True, 
            fmt='.1f', 
            cmap='YlGnBu', 
            linewidths=.5,
            cbar_kws={'label': 'Nilai (%)'}
        )
        
        plt.title('Metrik per Kelas (%)')
        plt.tight_layout()
        
        # Tampilkan plot
        plt.show()
        
        # Simpan plot jika diminta
        if save_plots:
            try:
                plt.savefig(f"{self.output_dir}/class_metrics.png", dpi=300, bbox_inches='tight')
                self.logger.info(f"✅ Plot disimpan: {self.output_dir}/class_metrics.png")
            except Exception as e:
                self.logger.warning(f"⚠️ Gagal menyimpan plot: {str(e)}")
    
    def _show_model_summary(self, results, metrics, save_plots=False):
        """
        Tampilkan ringkasan informasi model.
        
        Args:
            results: Dict hasil evaluasi
            metrics: Dict metrik evaluasi
            save_plots: Simpan informasi ke file
        """
        # Tampilkan ringkasan model
        display(HTML("<h3>📋 Ringkasan Model</h3>"))
        
        # Buat tabel informasi
        model_info = [
            {"Parameter": "Backbone", "Nilai": results.get('backbone', 'Unknown')},
            {"Parameter": "Mode Deteksi", "Nilai": results.get('detection_mode', 'Single-layer')},
            {"Parameter": "Ukuran Input", "Nilai": f"{results.get('input_size', (0, 0))}"},
            {"Parameter": "Inference Time", "Nilai": f"{metrics.get('inference_time', 0):.2f} ms ({1000/max(0.001, metrics.get('inference_time', 0)):.1f} FPS)"},
            {"Parameter": "Model Path", "Nilai": results.get('model_path', 'Unknown')},
            {"Parameter": "Jumlah Parameter", "Nilai": f"{results.get('param_count', 0):,}"}
        ]
        
        # Tambahkan info device
        device_info = results.get('device', 'Unknown')
        model_info.append({"Parameter": "Device", "Nilai": device_info})
        
        # Display as DataFrame
        model_df = pd.DataFrame(model_info)
        display(model_df)
        
        # Simpan informasi jika diminta
        if save_plots:
            try:
                model_df.to_csv(f"{self.output_dir}/model_summary.csv", index=False)
                self.logger.info(f"✅ Ringkasan disimpan: {self.output_dir}/model_summary.csv")
            except Exception as e:
                self.logger.warning(f"⚠️ Gagal menyimpan ringkasan: {str(e)}")
    
    def _show_sample_detections(self, results, save_plots=False):
        """
        Tampilkan contoh hasil deteksi.
        
        Args:
            results: Dict hasil evaluasi
            save_plots: Simpan gambar ke file
        """
        # Periksa apakah ada sampel deteksi
        sample_detections = results.get('sample_detections', [])
        
        if not sample_detections:
            return
            
        # Tampilkan judul
        display(HTML("<h3>🖼️ Contoh Hasil Deteksi</h3>"))
        
        # Tampilkan sampel (maks 5)
        for i, img in enumerate(sample_detections[:5]):
            plt.figure(figsize=(10, 8))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"Sampel Deteksi {i+1}")
            plt.tight_layout()
            plt.show()
            
            # Simpan gambar jika diminta
            if save_plots:
                try:
                    plt.savefig(f"{self.output_dir}/sample_detection_{i+1}.png", dpi=300, bbox_inches='tight')
                except Exception as e:
                    self.logger.warning(f"⚠️ Gagal menyimpan sampel deteksi: {str(e)}")
    
    def visualize_comparison(
        self,
        results_list: List[Dict[str, Any]],
        model_names: List[str],
        title: str = "Perbandingan Model",
        save_plots: bool = False
    ):
        """
        Visualisasikan perbandingan beberapa model.
        
        Args:
            results_list: List Dict hasil evaluasi dari beberapa model
            model_names: Nama model untuk perbandingan
            title: Judul untuk visualisasi
            save_plots: Simpan plot ke file
        """
        if not results_list or len(results_list) == 0:
            self.logger.warning("⚠️ Tidak ada hasil evaluasi untuk dibandingkan")
            return
            
        # Standarisasi panjang results_list dan model_names
        if len(model_names) < len(results_list):
            # Tambahkan nama generik jika kurang
            model_names.extend([f"Model {i+1}" for i in range(len(model_names), len(results_list))])
        elif len(model_names) > len(results_list):
            # Potong jika terlalu panjang
            model_names = model_names[:len(results_list)]
            
        try:
            # Ekstrak metrik untuk semua model
            comparison_data = []
            
            for i, (results, name) in enumerate(zip(results_list, model_names)):
                metrics = results.get('metrics', {})
                
                # Standarisasi ke persentase
                comparison_data.append({
                    'Model': name,
                    'Accuracy': metrics.get('accuracy', 0) * 100,
                    'Precision': metrics.get('precision', 0) * 100,
                    'Recall': metrics.get('recall', 0) * 100,
                    'F1-Score': metrics.get('f1', 0) * 100,
                    'mAP': metrics.get('mAP', 0) * 100,
                    'Inference Time (ms)': metrics.get('inference_time', 0),
                    'FPS': 1000 / max(0.001, metrics.get('inference_time', 0))
                })
            
            # Buat DataFrame
            comparison_df = pd.DataFrame(comparison_data)
            
            # Tampilkan judul
            display(HTML(f"<h2>{title}</h2>"))
            
            # Tampilkan tabel perbandingan dengan highlight nilai terbaik
            display(HTML("<h3>📊 Tabel Perbandingan</h3>"))
            
            # Highlight nilai terbaik
            styled_df = comparison_df.style.format({
                'Accuracy': '{:.2f}%',
                'Precision': '{:.2f}%',
                'Recall': '{:.2f}%',
                'F1-Score': '{:.2f}%',
                'mAP': '{:.2f}%',
                'Inference Time (ms)': '{:.2f}',
                'FPS': '{:.1f}'
            })
            
            # Highlight nilai terbaik untuk akurasi (max lebih baik)
            for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'mAP', 'FPS']:
                styled_df = styled_df.highlight_max(
                    subset=[col],
                    color='lightgreen'
                )
            
            # Highlight nilai terbaik untuk inference time (min lebih baik)
            styled_df = styled_df.highlight_min(
                subset=['Inference Time (ms)'],
                color='lightgreen'
            )
            
            # Tampilkan tabel
            display(styled_df)
            
            # Plot perbandingan
            self._plot_model_comparison(comparison_df, save_plots)
            
            # Tampilkan analisis dan rekomendasi
            self._show_comparison_analysis(comparison_df)
            
        except Exception as e:
            self.logger.error(f"❌ Error saat visualisasi perbandingan: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _plot_model_comparison(self, comparison_df, save_plots=False):
        """
        Buat plot perbandingan model.
        
        Args:
            comparison_df: DataFrame perbandingan
            save_plots: Simpan plot ke file
        """
        # Metrik untuk perbandingan
        accuracy_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'mAP']
        performance_metrics = ['Inference Time (ms)', 'FPS']
        
        # Setup figure
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Accuracy metrics
        ax1 = axes[0]
        
        # Reshape data untuk seaborn
        accuracy_data = pd.melt(
            comparison_df,
            id_vars=['Model'],
            value_vars=accuracy_metrics,
            var_name='Metric',
            value_name='Value'
        )
        
        # Plot grouped bar chart
        sns.barplot(
            x='Metric',
            y='Value',
            hue='Model',
            data=accuracy_data,
            ax=ax1
        )
        
        ax1.set_title('Perbandingan Metrik Akurasi (%)')
        ax1.set_xlabel('Metrik')
        ax1.set_ylabel('Nilai (%)')
        ax1.set_ylim(0, 105)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        ax1.legend(title='Model')
        
        # Plot 2: Performance metrics
        ax2 = axes[1]
        
        # Bar plot for inference time
        x = np.arange(len(comparison_df['Model']))
        width = 0.35
        
        # Plot inference time
        time_bars = ax2.bar(
            x - width/2,
            comparison_df['Inference Time (ms)'],
            width,
            label='Inference Time (ms)',
            color='#3498db'
        )
        
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Inference Time (ms)', color='#3498db')
        ax2.tick_params(axis='y', labelcolor='#3498db')
        ax2.set_xticks(x)
        ax2.set_xticklabels(comparison_df['Model'])
        ax2.grid(axis='y', linestyle='--', alpha=0.7, color='#3498db')
        
        # Secondary y-axis for FPS
        ax2b = ax2.twinx()
        fps_bars = ax2b.bar(
            x + width/2,
            comparison_df['FPS'],
            width,
            label='FPS',
            color='#e74c3c'
        )
        
        ax2b.set_ylabel('FPS', color='#e74c3c')
        ax2b.tick_params(axis='y', labelcolor='#e74c3c')
        ax2b.grid(False)
        
        # Tambahkan legend
        ax2.legend(handles=[time_bars, fps_bars], loc='upper left')
        
        ax2.set_title('Perbandingan Kinerja Waktu')
        
        # Tambahkan nilai di atas bar
        for bar, time_val in zip(time_bars, comparison_df['Inference Time (ms)']):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.1,
                f"{time_val:.1f}",
                ha='center',
                va='bottom',
                color='#3498db',
                fontweight='bold'
            )
            
        for bar, fps_val in zip(fps_bars, comparison_df['FPS']):
            height = bar.get_height()
            ax2b.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.1,
                f"{fps_val:.1f}",
                ha='center',
                va='bottom',
                color='#e74c3c',
                fontweight='bold'
            )
        
        # Finalisasi layout
        plt.tight_layout()
        
        # Tampilkan plot
        plt.show()
        
        # Simpan plot jika diminta
        if save_plots:
            try:
                fig.savefig(f"{self.output_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
                self.logger.info(f"✅ Plot disimpan: {self.output_dir}/model_comparison.png")
            except Exception as e:
                self.logger.warning(f"⚠️ Gagal menyimpan plot: {str(e)}")
    
    def _show_comparison_analysis(self, comparison_df):
        """
        Tampilkan analisis perbandingan model.
        
        Args:
            comparison_df: DataFrame perbandingan
        """
        # Tampilkan judul
        display(HTML("<h3>🔍 Analisis Perbandingan</h3>"))
        
        # Temukan model terbaik berdasarkan F1-Score
        best_f1_model = comparison_df.loc[comparison_df['F1-Score'].idxmax()]['Model']
        best_accuracy_model = comparison_df.loc[comparison_df['Accuracy'].idxmax()]['Model']
        best_map_model = comparison_df.loc[comparison_df['mAP'].idxmax()]['Model']
        fastest_model = comparison_df.loc[comparison_df['Inference Time (ms)'].idxmin()]['Model']
        
        # Cek apakah semua "terbaik" adalah model yang sama
        all_best_same = (best_f1_model == best_accuracy_model == best_map_model == fastest_model)
        
        # Tampilkan analisis
        if all_best_same:
            analysis = f"""
            <div style="background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin: 10px 0;">
                <p><b>🏆 Model terbaik secara keseluruhan adalah <span style="color: #2980b9;">{best_f1_model}</span></b></p>
                <p>Model ini unggul di semua kategori penting: akurasi, F1-Score, mAP, dan kecepatan inferensi.</p>
            </div>
            """
        else:
            # Analisis lebih detail
            analysis = f"""
            <div style="background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin: 10px 0;">
                <p><b>🎯 Rangkuman Perbandingan:</b></p>
                <ul>
                    <li><b>Model dengan F1-Score terbaik:</b> <span style="color: #2980b9;">{best_f1_model}</span></li>
                    <li><b>Model dengan akurasi tertinggi:</b> <span style="color: #2980b9;">{best_accuracy_model}</span></li>
                    <li><b>Model dengan mAP tertinggi:</b> <span style="color: #2980b9;">{best_map_model}</span></li>
                    <li><b>Model tercepat:</b> <span style="color: #2980b9;">{fastest_model}</span></li>
                </ul>
            """
            
            # Tambahkan rekomendasi berdasarkan kasus penggunaan
            analysis += f"""
                <p><b>🔮 Rekomendasi berdasarkan kasus penggunaan:</b></p>
                <ul>
                    <li><b>Untuk aplikasi real-time:</b> <span style="color: #2980b9;">{fastest_model}</span></li>
                    <li><b>Untuk keseimbangan akurasi dan kecepatan:</b> <span style="color: #2980b9;">{best_f1_model}</span></li>
                    <li><b>Untuk akurasi tertinggi tanpa peduli kecepatan:</b> <span style="color: #2980b9;">{best_map_model}</span></li>
                </ul>
            </div>
            """
        
        # Tampilkan analisis
        display(HTML(analysis))
        
        # Trade-off analysis
        self._show_tradeoff_analysis(comparison_df)
    
    def _show_tradeoff_analysis(self, comparison_df):
        """
        Tampilkan analisis trade-off akurasi vs kecepatan.
        
        Args:
            comparison_df: DataFrame perbandingan
        """
        # Tampilkan judul
        display(HTML("<h3>⚖️ Analisis Trade-off Akurasi vs Kecepatan</h3>"))
        
        # Plot trade-off
        plt.figure(figsize=(10, 6))
        
        # Scatter plot dengan tekstur untuk membedakan model
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        
        # Ensure we have enough markers
        if len(comparison_df) > len(markers):
            markers = markers * (len(comparison_df) // len(markers) + 1)
        
        # Ukuran berdasarkan F1-Score
        sizes = comparison_df['F1-Score'] * 5
        
        # Plot scatter
        for i, (idx, row) in enumerate(comparison_df.iterrows()):
            plt.scatter(
                row['Inference Time (ms)'],
                row['mAP'],
                s=sizes[idx],
                marker=markers[i],
                label=row['Model'],
                alpha=0.7
            )
            
            # Tambahkan label
            plt.annotate(
                row['Model'],
                (row['Inference Time (ms)'], row['mAP']),
                xytext=(5, 5),
                textcoords='offset points'
            )
        
        # Format plot
        plt.xlabel('Inference Time (ms) - Semakin kecil semakin baik →')
        plt.ylabel('mAP (%) - Semakin besar semakin baik →')
        plt.title('Akurasi vs Kecepatan Inferensi')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='Model')
        
        # Add trade-off regions
        plt.axvspan(0, plt.xlim()[1]*0.3, alpha=0.1, color='green', label='Kecepatan Tinggi')
        plt.axvspan(plt.xlim()[1]*0.3, plt.xlim()[1]*0.7, alpha=0.1, color='yellow', label='Seimbang')
        plt.axvspan(plt.xlim()[1]*0.7, plt.xlim()[1], alpha=0.1, color='red', label='Kecepatan Rendah')
        
        plt.axhspan(0, 100*0.3, alpha=0.1, color='red', label='Akurasi Rendah')
        plt.axhspan(100*0.3, 100*0.7, alpha=0.1, color='yellow', label='Akurasi Sedang')
        plt.axhspan(100*0.7, 100, alpha=0.1, color='green', label='Akurasi Tinggi')
        
        # Tampilkan plot
        plt.tight_layout()
        plt.show()
        
        # Berikan kesimpulan
        # Identifikasi model optimal berdasarkan trade-off
        max_accuracy = comparison_df['mAP'].max()
        min_time = comparison_df['Inference Time (ms)'].min()
        
        # Normalisasi untuk score
        normalized_df = comparison_df.copy()
        normalized_df['Norm_mAP'] = comparison_df['mAP'] / max_accuracy
        normalized_df['Norm_Time'] = min_time / comparison_df['Inference Time (ms)']
        
        # Hitung skor keseimbangan
        normalized_df['Balance_Score'] = (normalized_df['Norm_mAP'] + normalized_df['Norm_Time']) / 2
        
        # Temukan model terbaik berdasarkan keseimbangan
        best_balanced = normalized_df.loc[normalized_df['Balance_Score'].idxmax()]['Model']
        
        # Tunjukkan kesimpulan
        conclusion = f"""
        <div style="background-color: #f0f9ed; padding: 15px; border-radius: 5px; margin: 10px 0;">
            <p><b>💡 Kesimpulan:</b></p>
            <p>Berdasarkan analisis trade-off antara akurasi (mAP) dan kecepatan inferensi, model dengan keseimbangan terbaik adalah <span style="color: #27ae60; font-weight: bold;">{best_balanced}</span>.</p>
            <p>Model ini menawarkan kombinasi optimal antara akurasi deteksi dan kecepatan pemrosesan untuk aplikasi SmartCash.</p>
        </div>
        """
        
        # Tampilkan kesimpulan
        display(HTML(conclusion))
    
    def visualize_research_scenario(
        self,
        scenario_results: pd.DataFrame,
        title: str = "Hasil Skenario Penelitian",
        save_plots: bool = False
    ):
        """
        Visualisasikan hasil skenario penelitian.
        
        Args:
            scenario_results: DataFrame hasil skenario
            title: Judul untuk visualisasi
            save_plots: Simpan plot ke file
        """
        if scenario_results.empty:
            self.logger.warning("⚠️ Tidak ada hasil skenario untuk divisualisasikan")
            return
        
        try:
            # Tampilkan judul
            display(HTML(f"<h2>{title}</h2>"))
            
            # Tampilkan tabel hasil
            display(HTML("<h3>📊 Hasil Skenario Penelitian</h3>"))
            
            # Format DataFrame
            styled_df = scenario_results.style.format({
                'Akurasi': '{:.2f}%',
                'Precision': '{:.2f}%',
                'Recall': '{:.2f}%',
                'F1-Score': '{:.2f}%',
                'mAP': '{:.2f}%',
                'Waktu Inferensi': '{:.2f} ms'
            })
            
            # Highlight nilai terbaik
            highlight_cols = ['Akurasi', 'Precision', 'Recall', 'F1-Score', 'mAP']
            styled_df = styled_df.highlight_max(subset=highlight_cols, color='lightgreen')
            styled_df = styled_df.highlight_min(subset=['Waktu Inferensi'], color='lightgreen')
            
            # Tampilkan tabel
            display(styled_df)
            
            # Visualisasikan hasil dengan grafik
            self._plot_scenario_comparison(scenario_results, save_plots)
            
            # Analisis dan kesimpulan
            self._show_scenario_analysis(scenario_results)
            
        except Exception as e:
            self.logger.error(f"❌ Error saat visualisasi skenario penelitian: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _plot_scenario_comparison(self, scenario_results, save_plots=False):
        """
        Plot perbandingan skenario penelitian.
        
        Args:
            scenario_results: DataFrame hasil skenario
            save_plots: Simpan plot ke file
        """
        # Extract scenario names
        scenarios = scenario_results['Skenario'].tolist()
        
        # Setup figure
        fig, axes = plt.subplots(2, 1, figsize=(12, 14))
        
        # Plot 1: Accuracy metrics bar chart
        ax1 = axes[0]
        
        # Reshape data for seaborn
        metrics = ['Akurasi', 'Precision', 'Recall', 'F1-Score', 'mAP']
        plot_data = pd.melt(
            scenario_results,
            id_vars=['Skenario'],
            value_vars=metrics,
            var_name='Metrik',
            value_name='Nilai'
        )
        
        # Grouped bar chart
        sns.barplot(
            x='Skenario',
            y='Nilai',
            hue='Metrik',
            data=plot_data,
            ax=ax1
        )
        
        ax1.set_title('Perbandingan Metrik Akurasi per Skenario (%)')
        ax1.set_xlabel('Skenario')
        ax1.set_ylabel('Nilai (%)')
        ax1.set_ylim(0, 105)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        ax1.legend(title='Metrik')
        
        # Rotate scenario labels for readability
        plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Plot 2: Inference time by scenario
        ax2 = axes[1]
        
        # Bar chart for inference time
        time_data = scenario_results[['Skenario', 'Waktu Inferensi']].copy()
        time_data['FPS'] = 1000 / time_data['Waktu Inferensi']
        
        # Plot inference time
        bars = ax2.bar(
            scenarios,
            time_data['Waktu Inferensi'],
            color='#3498db',
            alpha=0.7
        )
        
        ax2.set_title('Waktu Inferensi per Skenario (ms)')
        ax2.set_xlabel('Skenario')
        ax2.set_ylabel('Waktu Inferensi (ms)', color='#3498db')
        ax2.tick_params(axis='y', labelcolor='#3498db')
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Rotate scenario labels for readability
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Secondary y-axis for FPS
        ax2b = ax2.twinx()
        ax2b.plot(scenarios, time_data['FPS'], 'ro-', linewidth=2)
        ax2b.set_ylabel('FPS', color='#e74c3c')
        ax2b.tick_params(axis='y', labelcolor='#e74c3c')
        
        # Add labels on bars
        for bar, time_val, fps_val in zip(bars, time_data['Waktu Inferensi'], time_data['FPS']):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.5,
                f"{time_val:.1f} ms",
                ha='center',
                va='bottom',
                color='#3498db',
                fontweight='bold'
            )
            ax2b.text(
                bar.get_x() + bar.get_width()/2.,
                fps_val + 0.5,
                f"{fps_val:.1f} FPS",
                ha='center',
                va='bottom',
                color='#e74c3c',
                fontweight='bold'
            )
        
        # Finalisasi layout
        plt.tight_layout()
        
        # Tampilkan plot
        plt.show()
        
        # Simpan plot jika diminta
        if save_plots:
            try:
                fig.savefig(f"{self.output_dir}/scenario_comparison.png", dpi=300, bbox_inches='tight')
                self.logger.info(f"✅ Plot disimpan: {self.output_dir}/scenario_comparison.png")
            except Exception as e:
                self.logger.warning(f"⚠️ Gagal menyimpan plot: {str(e)}")
    
    def _show_scenario_analysis(self, scenario_results):
        """
        Tampilkan analisis skenario penelitian.
        
        Args:
            scenario_results: DataFrame hasil skenario
        """
        # Tampilkan judul
        display(HTML("<h3>🔍 Analisis Skenario Penelitian</h3>"))
        
        # Ekstrak insight dari skenario
        best_scenario_f1 = scenario_results.loc[scenario_results['F1-Score'].idxmax()]['Skenario']
        best_scenario_map = scenario_results.loc[scenario_results['mAP'].idxmax()]['Skenario']
        best_scenario_speed = scenario_results.loc[scenario_results['Waktu Inferensi'].idxmin()]['Skenario']
        
        # Analisis backbone (EfficientNet vs CSPDarknet)
        efficientnet_results = scenario_results[scenario_results['Skenario'].str.contains('EfficientNet')]
        cspdarknet_results = scenario_results[scenario_results['Skenario'].str.contains('CSPDarknet')]
        
        efficientnet_avg_f1 = efficientnet_results['F1-Score'].mean()
        cspdarknet_avg_f1 = cspdarknet_results['F1-Score'].mean()
        
        efficientnet_avg_time = efficientnet_results['Waktu Inferensi'].mean()
        cspdarknet_avg_time = cspdarknet_results['Waktu Inferensi'].mean()
        
        # Analisis kondisi (Posisi vs Pencahayaan)
        position_results = scenario_results[scenario_results['Skenario'].str.contains('Posisi')]
        lighting_results = scenario_results[scenario_results['Skenario'].str.contains('Pencahayaan')]
        
        position_avg_f1 = position_results['F1-Score'].mean()
        lighting_avg_f1 = lighting_results['F1-Score'].mean()
        
        # Persiapkan kesimpulan
        backbone_conclusion = ""
        if efficientnet_avg_f1 > cspdarknet_avg_f1:
            backbone_conclusion = f"""
            <p>EfficientNet-B4 menunjukkan performa yang lebih baik dibandingkan CSPDarknet dengan peningkatan F1-Score sebesar {efficientnet_avg_f1 - cspdarknet_avg_f1:.2f}%.</p>
            """
            if efficientnet_avg_time < cspdarknet_avg_time:
                backbone_conclusion += f"""
                <p>Selain itu, EfficientNet-B4 juga lebih cepat dengan rata-rata waktu inferensi {efficientnet_avg_time:.2f}ms dibandingkan CSPDarknet ({cspdarknet_avg_time:.2f}ms).</p>
                """
            else:
                backbone_conclusion += f"""
                <p>Namun, CSPDarknet sedikit lebih cepat dengan rata-rata waktu inferensi {cspdarknet_avg_time:.2f}ms dibandingkan EfficientNet-B4 ({efficientnet_avg_time:.2f}ms).</p>
                """
        else:
            backbone_conclusion = f"""
            <p>CSPDarknet menunjukkan performa yang lebih baik dibandingkan EfficientNet-B4 dengan peningkatan F1-Score sebesar {cspdarknet_avg_f1 - efficientnet_avg_f1:.2f}%.</p>
            """
            if cspdarknet_avg_time < efficientnet_avg_time:
                backbone_conclusion += f"""
                <p>Selain itu, CSPDarknet juga lebih cepat dengan rata-rata waktu inferensi {cspdarknet_avg_time:.2f}ms dibandingkan EfficientNet-B4 ({efficientnet_avg_time:.2f}ms).</p>
                """
            else:
                backbone_conclusion += f"""
                <p>Namun, EfficientNet-B4 sedikit lebih cepat dengan rata-rata waktu inferensi {efficientnet_avg_time:.2f}ms dibandingkan CSPDarknet ({cspdarknet_avg_time:.2f}ms).</p>
                """
        
        # Kondisi kesimpulan
        condition_conclusion = ""
        if position_avg_f1 > lighting_avg_f1:
            condition_conclusion = f"""
            <p>Model menunjukkan performa lebih baik pada variasi posisi dengan rata-rata F1-Score {position_avg_f1:.2f}% dibandingkan variasi pencahayaan ({lighting_avg_f1:.2f}%).</p>
            <p>Hal ini menunjukkan bahwa model lebih tahan terhadap perubahan posisi uang kertas dibandingkan perubahan kondisi pencahayaan.</p>
            """
        else:
            condition_conclusion = f"""
            <p>Model menunjukkan performa lebih baik pada variasi pencahayaan dengan rata-rata F1-Score {lighting_avg_f1:.2f}% dibandingkan variasi posisi ({position_avg_f1:.2f}%).</p>
            <p>Hal ini menunjukkan bahwa model lebih tahan terhadap perubahan kondisi pencahayaan dibandingkan perubahan posisi uang kertas.</p>
            """
        
        # Format dan tampilkan kesimpulan
        analysis = f"""
        <div style="background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin: 10px 0;">
            <p><b>🏆 Skenario terbaik berdasarkan F1-Score:</b> <span style="color: #2980b9;">{best_scenario_f1}</span></p>
            <p><b>🏆 Skenario terbaik berdasarkan mAP:</b> <span style="color: #2980b9;">{best_scenario_map}</span></p>
            <p><b>🏆 Skenario tercepat:</b> <span style="color: #2980b9;">{best_scenario_speed}</span></p>
            
            <p><b>📊 Perbandingan Backbone:</b></p>
            {backbone_conclusion}
            
            <p><b>📊 Perbandingan Kondisi:</b></p>
            {condition_conclusion}
            
            <p><b>💡 Rekomendasi:</b></p>
            <p>Berdasarkan hasil evaluasi, <span style="color: #2980b9;">{best_scenario_f1}</span> menunjukkan performa terbaik dan direkomendasikan untuk implementasi SmartCash.</p>
        </div>
        """
        
        # Tampilkan analisis
        display(HTML(analysis))

class ResultVisualizer:
    """Visualizer untuk hasil deteksi mata uang dengan tampilan yang informatif."""
    
    def __init__(
        self,
        output_dir: str = "results/detection",
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi visualizer.
        
        Args:
            output_dir: Direktori output untuk menyimpan visualisasi
            logger: Logger untuk logging (opsional)
        """
        self.logger = logger or get_logger("result_visualizer")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup warna untuk setiap denominasi
        self.currency_colors = {
            '001': (255, 0, 0),      # Rp 1.000 (Merah)
            '002': (0, 0, 255),      # Rp 2.000 (Biru)
            '005': (0, 255, 0),      # Rp 5.000 (Hijau)
            '010': (128, 0, 128),    # Rp 10.000 (Ungu)
            '020': (0, 128, 128),    # Rp 20.000 (Teal)
            '050': (128, 128, 0),    # Rp 50.000 (Olive)
            '100': (0, 0, 128),      # Rp 100.000 (Navy)
            # Layer 2 (area nominal)
            'l2_001': (255, 50, 50),
            'l2_002': (50, 50, 255),
            'l2_005': (50, 255, 50),
            'l2_010': (178, 50, 178),
            'l2_020': (50, 178, 178),
            'l2_050': (178, 178, 50),
            'l2_100': (50, 50, 178),
            # Layer 3 (fitur keamanan)
            'l3_sign': (255, 150, 0),
            'l3_text': (150, 255, 0),
            'l3_thread': (0, 150, 255)
        }
    
    def visualize_detections(
        self,
        image: np.ndarray,
        detections: List[Dict],
        show_confidence: bool = True,
        show_value: bool = True,
        filename: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualisasikan hasil deteksi pada gambar.
        
        Args:
            image: Gambar input sebagai array numpy
            detections: List deteksi hasil dari DetectionHandler
            show_confidence: Tampilkan nilai confidence
            show_value: Tampilkan nilai denominasi uang
            filename: Nama file untuk menyimpan (jika None, gunakan timestamp)
            
        Returns:
            Gambar dengan anotasi deteksi
        """
        # Copy gambar agar tidak memodifikasi asli
        vis_image = image.copy()
        
        # Total nilai yang terdeteksi
        total_value = 0
        
        # Kumpulkan denominasi untuk penghitungan
        denominations = []
        
        # Gambar setiap deteksi
        for det in detections:
            # Extract info
            bbox = det['bbox']
            class_name = det['class_name']
            conf = det.get('confidence', 1.0)
            
            # Konversi nama kelas ke nilai denominasi
            value = 0
            if class_name in ['001', 'l2_001']:
                value = 1000
                denomination = "1rb"
            elif class_name in ['002', 'l2_002']:
                value = 2000
                denomination = "2rb"
            elif class_name in ['005', 'l2_005']:
                value = 5000
                denomination = "5rb"
            elif class_name in ['010', 'l2_010']:
                value = 10000
                denomination = "10rb"
            elif class_name in ['020', 'l2_020']:
                value = 20000
                denomination = "20rb"
            elif class_name in ['050', 'l2_050']:
                value = 50000
                denomination = "50rb"
            elif class_name in ['100', 'l2_100']:
                value = 100000
                denomination = "100rb"
            else:
                denomination = class_name
            
            # Jika ini adalah deteksi uang utuh (layer 1), tambahkan ke total
            if class_name in ['001', '002', '005', '010', '020', '050', '100']:
                total_value += value
                denominations.append(denomination)
            
            # Dapatkan warna untuk kelas ini
            color = self.currency_colors.get(class_name, (255, 255, 255))
            
            # Unpack bbox
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
            else:
                # Format alternatif jika bbox dalam format [x_center, y_center, width, height]
                x_center, y_center, width, height = bbox
                x1 = int(x_center - width/2)
                y1 = int(y_center - height/2)
                x2 = int(x_center + width/2)
                y2 = int(y_center + height/2)
            
            # Pastikan koordinat dalam range gambar
            h, w = vis_image.shape[:2]
            x1 = max(0, min(w-1, x1))
            y1 = max(0, min(h-1, y1))
            x2 = max(0, min(w-1, x2))
            y2 = max(0, min(h-1, y2))
            
            # Gambar bounding box
            cv2.rectangle(
                vis_image,
                (x1, y1),
                (x2, y2),
                color,
                2
            )
            
            # Persiapkan label
            if show_value and value > 0:
                label = f"{denomination}"
            else:
                label = f"{class_name}"
                
            if show_confidence:
                label += f" {conf:.2f}"
            
            # Gambar label dengan background untuk keterbacaan
            label_size, baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                1
            )
            
            cv2.rectangle(
                vis_image,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            
            cv2.putText(
                vis_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        # Tampilkan total nilai di bagian bawah
        if total_value > 0:
            total_text = f"Total: Rp {total_value:,}"
            denom_text = f"Denominasi: {', '.join(denominations)}"
            
            # Gambar background untuk teks
            cv2.rectangle(
                vis_image,
                (10, vis_image.shape[0] - 60),
                (10 + max(cv2.getTextSize(total_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0],
                          cv2.getTextSize(denom_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0]) + 10,
                vis_image.shape[0] - 10),
                (0, 0, 0),
                -1
            )
            
            # Tampilkan teks total dan denominasi
            cv2.putText(
                vis_image,
                total_text,
                (15, vis_image.shape[0] - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            cv2.putText(
                vis_image,
                denom_text,
                (15, vis_image.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )
        
        # Simpan hasil visualisasi jika filename disediakan
        if filename:
            output_path = self.output_dir / filename
            cv2.imwrite(str(output_path), vis_image)
            self.logger.info(f"✅ Visualisasi disimpan di: {output_path}")
        
        return vis_image
    
    def plot_detections(
        self,
        images: List[np.ndarray],
        detections_list: List[List[Dict]],
        grid_size: Optional[Tuple[int, int]] = None,
        title: str = "Hasil Deteksi Mata Uang",
        show: bool = True,
        save: bool = False
    ) -> np.ndarray:
        """
        Plot multiple deteksi dalam grid.
        
        Args:
            images: List gambar input
            detections_list: List dari list deteksi untuk setiap gambar
            grid_size: Ukuran grid (rows, cols), jika None akan ditentukan otomatis
            title: Judul untuk plot
            show: Tampilkan plot
            save: Simpan plot ke file
            
        Returns:
            Komposit gambar dengan deteksi
        """
        # Validasi input
        if not images or len(images) == 0:
            self.logger.warning("⚠️ Tidak ada gambar untuk divisualisasikan")
            return None
            
        if len(images) != len(detections_list):
            self.logger.warning("⚠️ Jumlah gambar dan deteksi tidak sama")
            return None
        
        # Tentukan ukuran grid
        n_images = len(images)
        
        if grid_size is None:
            # Hitung grid yang optimal
            cols = min(4, n_images)
            rows = (n_images + cols - 1) // cols
            grid_size = (rows, cols)
        else:
            rows, cols = grid_size
            
        # Validasi grid size
        if rows * cols < n_images:
            self.logger.warning(f"⚠️ Grid size {grid_size} terlalu kecil untuk {n_images} gambar")
            # Adjust grid size
            cols = min(4, n_images)
            rows = (n_images + cols - 1) // cols
            grid_size = (rows, cols)
            self.logger.info(f"🔄 Menggunakan grid size {grid_size}")
        
        # Visualisasikan setiap gambar
        vis_images = []
        
        for img, detections in zip(images, detections_list):
            # Skip jika gambar None
            if img is None:
                continue
                
            # Visualisasikan
            vis_img = self.visualize_detections(
                img,
                detections,
                show_confidence=True,
                show_value=True
            )
            
            vis_images.append(vis_img)
        
        # Resize semua gambar ke ukuran yang sama
        if vis_images:
            # Tentukan ukuran target
            target_height = max(img.shape[0] for img in vis_images)
            target_width = max(img.shape[1] for img in vis_images)
            
            # Resize semua gambar
            vis_images = [
                cv2.resize(img, (target_width, target_height))
                for img in vis_images
            ]
            
            # Buat grid
            grid_img = self._create_grid(vis_images, grid_size, title)
            
            # Tampilkan grid
            if show:
                plt.figure(figsize=(15, 10))
                plt.imshow(cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.title(title)
                plt.tight_layout()
                plt.show()
            
            # Simpan hasil
            if save:
                output_path = self.output_dir / f"{title.lower().replace(' ', '_')}.jpg"
                cv2.imwrite(str(output_path), grid_img)
                self.logger.info(f"✅ Grid deteksi disimpan di: {output_path}")
            
            return grid_img
        
        return None
    
    def _create_grid(
        self,
        images: List[np.ndarray],
        grid_size: Tuple[int, int],
        title: str = ""
    ) -> np.ndarray:
        """
        Buat grid dari gambar.
        
        Args:
            images: List gambar
            grid_size: Ukuran grid (rows, cols)
            title: Judul untuk grid
            
        Returns:
            Komposit gambar
        """
        rows, cols = grid_size
        
        # Ensure all images have the same size
        h, w = images[0].shape[:2]
        
        # Create empty grid
        grid = np.zeros((h * rows + 50, w * cols, 3), dtype=np.uint8)
        
        # Add title
        if title:
            cv2.putText(
                grid,
                title,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )
        
        # Fill grid
        for i, img in enumerate(images):
            if i >= rows * cols:
                break
                
            r = i // cols
            c = i % cols
            
            y_start = r * h + 50
            y_end = (r + 1) * h + 50
            x_start = c * w
            x_end = (c + 1) * w
            
            grid[y_start:y_end, x_start:x_end] = img
        
        return grid

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    normalize: bool = False,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot confusion matrix dengan format yang ditingkatkan.
    
    Args:
        cm: Confusion matrix
        class_names: List nama kelas
        normalize: Normalisasi nilai
        title: Judul plot
        cmap: Colormap
        figsize: Ukuran figure
        
    Returns:
        Matplotlib figure
    """
    # Ensure class_names has correct length
    if len(class_names) != cm.shape[0]:
        # Fallback to generic names
        class_names = [str(i) for i in range(cm.shape[0])]
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)  # Handle division by zero
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Jumlah" if not normalize else "Proporsi", rotation=-90, va="bottom")
    
    # Set labels and title
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    
    # Add ticks
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(class_names)
    
    # Add text annotations in the center of each cell
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if normalize:
                text = f"{cm[i, j]:.2f}"
            else:
                text = f"{cm[i, j]:d}"
                
            ax.text(j, i, text,
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    return fig

def plot_detections(
    image: np.ndarray,
    detections: List[Dict],
    class_colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
    show_conf: bool = True,
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    Plot hasil deteksi pada gambar.
    
    Args:
        image: Gambar input
        detections: List deteksi
        class_colors: Dict warna untuk setiap kelas
        show_conf: Tampilkan confidence score
        figsize: Ukuran figure
        
    Returns:
        Matplotlib figure
    """
    # Default colors if not provided
    if class_colors is None:
        class_colors = {
            '001': (255, 0, 0),      # Rp 1.000 (Merah)
            '002': (0, 0, 255),      # Rp 2.000 (Biru)
            '005': (0, 255, 0),      # Rp 5.000 (Hijau)
            '010': (128, 0, 128),    # Rp 10.000 (Ungu)
            '020': (0, 128, 128),    # Rp 20.000 (Teal)
            '050': (128, 128, 0),    # Rp 50.000 (Olive)
            '100': (0, 0, 128),      # Rp 100.000 (Navy)
        }
    
    # Copy image
    img = image.copy()
    
    # Draw each detection
    for det in detections:
        # Get bbox
        bbox = det['bbox']
        class_id = det.get('class_id', 0)
        class_name = det.get('class_name', str(class_id))
        confidence = det.get('confidence', 1.0)
        
        # Get color
        color = class_colors.get(class_name, (255, 255, 255))
        
        # Draw bbox
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_name}"
        if show_conf:
            label += f" {confidence:.2f}"
            
        # Label with background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(
            img,
            (x1, y1 - label_size[1] - 5),
            (x1 + label_size[0], y1),
            color,
            -1
        )
        
        cv2.putText(
            img,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.tight_layout()
    
    return fig

def plot_training_metrics(
    metrics: Dict[str, List],
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot metrik training.
    
    Args:
        metrics: Dict berisi metrik training
        figsize: Ukuran figure
        
    Returns:
        Matplotlib figure
    """
    # Ensure metrics contains necessary data
    required_keys = ['train_loss', 'val_loss', 'epochs']
    if not all(key in metrics for key in required_keys):
        # Fallback to empty plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Data metrik tidak lengkap", ha='center', va='center')
        return fig
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Plot losses
    ax1.plot(metrics['epochs'], metrics['train_loss'], 'b-', label='Training Loss')
    ax1.plot(metrics['epochs'], metrics['val_loss'], 'r-', label='Validation Loss')
    
    # Add learning rate if available
    if 'learning_rates' in metrics:
        ax2 = ax1.twinx()
        ax2.plot(metrics['epochs'], metrics['learning_rates'], 'g-', label='Learning Rate')
        ax2.set_ylabel('Learning Rate', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        
        # Add legend for all lines
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    else:
        ax1.legend(loc='upper right')
    
    # Set labels and title
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Metrics')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Highlight best epoch if available
    if 'val_loss' in metrics and len(metrics['val_loss']) > 0:
        best_epoch_idx = np.argmin(metrics['val_loss'])
        best_epoch = metrics['epochs'][best_epoch_idx]
        best_loss = metrics['val_loss'][best_epoch_idx]
        
        ax1.scatter([best_epoch], [best_loss], c='red', s=100, zorder=5)
        ax1.annotate(f'Best: {best_loss:.4f}', 
                     (best_epoch, best_loss),
                     xytext=(5, 5), 
                     textcoords='offset points')
    
    fig.tight_layout()
    return fig