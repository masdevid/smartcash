"""
File: smartcash/utils/visualization.py
Author: Alfrida Sabar
Deskripsi: Utilitas untuk visualisasi hasil deteksi, plot metrik, dan rendering grafik untuk SmartCash.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

class ResultVisualizer:
    """Kelas untuk memvisualisasikan hasil deteksi dan metrik evaluasi."""
    
    # Konfigurasi warna untuk visualisasi kelas
    COLORS = {
        '100': (255, 0, 0),     # 100rb - Biru
        '050': (0, 0, 255),     # 50rb - Merah
        '020': (0, 255, 0),     # 20rb - Hijau
        '010': (128, 0, 128),   # 10rb - Ungu
        '005': (0, 128, 128),   # 5rb - Coklat
        '002': (128, 128, 0),   # 2rb - Abu-Abu
        '001': (0, 0, 128),     # 1rb - Merah Tua
        'l2_100': (255, 50, 50),    # Layer 2 (nominal) 100rb
        'l2_050': (50, 50, 255),    # Layer 2 (nominal) 50rb
        'l2_020': (50, 255, 50),    # Layer 2 (nominal) 20rb
        'l2_010': (178, 50, 178),   # Layer 2 (nominal) 10rb
        'l2_005': (50, 178, 178),   # Layer 2 (nominal) 5rb
        'l2_002': (178, 178, 50),   # Layer 2 (nominal) 2rb
        'l2_001': (50, 50, 178),    # Layer 2 (nominal) 1rb
        'l3_sign': (255, 150, 0),   # Layer 3 (security) tanda tangan
        'l3_text': (150, 255, 0),   # Layer 3 (security) teks
        'l3_thread': (0, 150, 255)  # Layer 3 (security) benang
    }
    
    def __init__(self, output_dir: str = "results", logger: Optional[Any] = None):
        """
        Inisialisasi visualizer.
        
        Args:
            output_dir: Direktori untuk menyimpan hasil visualisasi
            logger: Logger untuk logging (opsional)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        
        # Buat direktori untuk berbagai jenis visualisasi
        self.detections_dir = self.output_dir / "detections"
        self.plots_dir = self.output_dir / "plots"
        self.metrics_dir = self.output_dir / "metrics"
        
        for dir_path in [self.detections_dir, self.plots_dir, self.metrics_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def plot_detections(
        self, 
        image: np.ndarray, 
        detections: List[Dict], 
        filename: Optional[str] = None,
        conf_threshold: float = 0.25,
        show_labels: bool = True
    ) -> np.ndarray:
        """
        Plot deteksi pada gambar.
        
        Args:
            image: Gambar input dalam format BGR (OpenCV)
            detections: List deteksi, masing-masing berbentuk dict dengan 'bbox', 'class_id', 'class_name', 'confidence'
            filename: Nama file untuk menyimpan hasil (opsional)
            conf_threshold: Threshold confidence untuk deteksi yang ditampilkan
            show_labels: Apakah menampilkan label kelas
            
        Returns:
            Gambar dengan deteksi yang telah divisualisasikan
        """
        # Buat salinan gambar untuk visualisasi
        vis_img = image.copy()
        
        # Filter deteksi berdasarkan confidence
        filtered_detections = [det for det in detections if det['confidence'] >= conf_threshold]
        
        # Gambar setiap deteksi
        for det in filtered_detections:
            # Ambil informasi deteksi
            bbox = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Dapatkan warna untuk kelas
            color = self.COLORS.get(class_name, (0, 255, 255))  # Default ke kuning jika tidak ditemukan
            
            # Gambar bounding box
            x1, y1, x2, y2 = bbox
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            
            # Tambahkan label jika diminta
            if show_labels:
                label = f"{class_name} {confidence:.2f}"
                
                # Ukuran teks
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                # Gambar background label
                cv2.rectangle(
                    vis_img, 
                    (x1, y1 - text_size[1] - 5), 
                    (x1 + text_size[0], y1), 
                    color, 
                    -1
                )
                
                # Gambar teks label
                cv2.putText(
                    vis_img, 
                    label, 
                    (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (255, 255, 255), 
                    1
                )
        
        # Tambahkan informasi jumlah deteksi
        cv2.putText(
            vis_img,
            f"Deteksi: {len(filtered_detections)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Simpan hasil jika nama file diberikan
        if filename:
            output_path = self.detections_dir / filename
            cv2.imwrite(str(output_path), vis_img)
            
            if self.logger:
                self.logger.info(f"✅ Hasil deteksi disimpan ke {output_path}")
        
        return vis_img
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_names: List[str],
        title: str = "Confusion Matrix",
        filename: Optional[str] = None,
        normalize: bool = True
    ) -> plt.Figure:
        """
        Plot confusion matrix dan simpan hasilnya.
        
        Args:
            confusion_matrix: Matriks konfusi
            class_names: List nama kelas
            title: Judul plot
            filename: Nama file untuk menyimpan hasil (opsional)
            normalize: Apakah menormalisasi nilai dalam matriks
            
        Returns:
            Figure matplotlib dari confusion matrix
        """
        plt.figure(figsize=(10, 8))
        
        # Normalisasi jika diminta
        if normalize:
            confusion_matrix = confusion_matrix.astype(float) / confusion_matrix.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        # Plot heatmap
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            square=True,
            cbar_kws={"shrink": .8}
        )
        
        plt.title(title, size=16)
        plt.ylabel('True Label', size=14)
        plt.xlabel('Predicted Label', size=14)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Simpan gambar jika nama file diberikan
        if filename:
            output_path = self.metrics_dir / filename
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            
            if self.logger:
                self.logger.info(f"📊 Confusion matrix disimpan ke {output_path}")
        
        fig = plt.gcf()
        plt.close()
        
        return fig
    
    def plot_metrics_history(
        self,
        metrics_history: Dict[str, List],
        title: str = "Training Metrics",
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot metrik training/evaluasi.
        
        Args:
            metrics_history: Dictionary berisi list metrik untuk setiap jenis metrik
            title: Judul plot
            filename: Nama file untuk menyimpan hasil (opsional)
            
        Returns:
            Figure matplotlib dari plot metrik
        """
        plt.figure(figsize=(12, 8))
        
        # Plot metrik yang valid
        for metric_name, values in metrics_history.items():
            if isinstance(values, list) and len(values) > 0 and metric_name != 'epochs':
                if 'epoch' in metrics_history:
                    plt.plot(metrics_history['epoch'], values, 'o-', label=metric_name)
                else:
                    plt.plot(range(1, len(values) + 1), values, 'o-', label=metric_name)
        
        plt.title(title, size=16)
        plt.xlabel('Epoch', size=14)
        plt.ylabel('Metrik', size=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Simpan gambar jika nama file diberikan
        if filename:
            output_path = self.plots_dir / filename
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            
            if self.logger:
                self.logger.info(f"📈 Plot metrik disimpan ke {output_path}")
        
        fig = plt.gcf()
        plt.close()
        
        return fig
    
    def display_evaluation_results(
        self,
        results: Dict[str, Any],
        title: str = "Evaluation Results",
        show_confusion_matrix: bool = True,
        show_class_metrics: bool = True
    ) -> None:
        """
        Tampilkan dan simpan hasil evaluasi.
        
        Args:
            results: Dictionary hasil evaluasi
            title: Judul tampilan
            show_confusion_matrix: Apakah menampilkan confusion matrix
            show_class_metrics: Apakah menampilkan metrik per kelas
        """
        # Tampilkan metrik utama
        print(f"\n{title}")
        print("=" * len(title))
        
        metrics = results.get('metrics', {})
        
        print(f"\n📊 Metrik Evaluasi:")
        print(f"• Akurasi: {metrics.get('accuracy', 0) * 100:.2f}%")
        print(f"• Precision: {metrics.get('precision', 0) * 100:.2f}%")
        print(f"• Recall: {metrics.get('recall', 0) * 100:.2f}%")
        print(f"• F1-Score: {metrics.get('f1', 0) * 100:.2f}%")
        print(f"• mAP: {metrics.get('mAP', 0) * 100:.2f}%")
        print(f"• Waktu Inferensi: {metrics.get('inference_time', 0) * 1000:.2f} ms")
        
        # Plot dan tampilkan confusion matrix
        if show_confusion_matrix and 'confusion_matrix' in results:
            cm = results['confusion_matrix']
            class_names = results.get('class_names', [str(i) for i in range(cm.shape[0])])
            
            self.plot_confusion_matrix(
                cm, 
                class_names, 
                title="Confusion Matrix", 
                filename=f"confusion_matrix_{int(time.time())}.png"
            )
            
            # Menampilkan confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names
            )
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.show()
        
        # Tampilkan metrik per kelas
        if show_class_metrics and 'class_metrics' in results:
            class_metrics = results['class_metrics']
            
            print("\n📊 Metrik per Kelas:")
            for class_name, metrics in class_metrics.items():
                print(f"• {class_name}:")
                print(f"  - Precision: {metrics.get('precision', 0) * 100:.2f}%")
                print(f"  - Recall: {metrics.get('recall', 0) * 100:.2f}%")
                print(f"  - F1-Score: {metrics.get('f1', 0) * 100:.2f}%")
    
    def visualize_research_comparison(self, results_df):
        """
        Visualisasikan perbandingan hasil penelitian.
        
        Args:
            results_df: DataFrame hasil penelitian
        """
        if results_df.empty:
            print("⚠️ Tidak ada hasil penelitian untuk divisualisasikan")
            return
        
        # Filter hanya hasil yang sukses
        success_results = results_df[results_df['Status'] == 'Sukses'].copy()
        
        if success_results.empty:
            print("⚠️ Tidak ada skenario yang berhasil dievaluasi")
            return
        
        # Tampilkan tabel hasil
        print("📋 Tabel Hasil Evaluasi:")
        display_df = success_results[['Skenario', 'Backbone', 'Kondisi', 'Akurasi', 'F1-Score', 'mAP', 'Inference Time (ms)']]
        print(display_df.to_string(index=False))
        
        # Visualisasi perbandingan metrik
        plt.figure(figsize=(15, 10))
        
        # Plot perbandingan akurasi
        plt.subplot(2, 2, 1)
        sns.barplot(x='Skenario', y='Akurasi', hue='Backbone', data=success_results)
        plt.title('Perbandingan Akurasi (%)')
        plt.ylabel('Akurasi (%)')
        plt.xticks(rotation=45)
        
        # Plot perbandingan F1-Score
        plt.subplot(2, 2, 2)
        sns.barplot(x='Skenario', y='F1-Score', hue='Backbone', data=success_results)
        plt.title('Perbandingan F1-Score (%)')
        plt.ylabel('F1-Score (%)')
        plt.xticks(rotation=45)
        
        # Plot perbandingan mAP
        plt.subplot(2, 2, 3)
        sns.barplot(x='Skenario', y='mAP', hue='Backbone', data=success_results)
        plt.title('Perbandingan mAP (%)')
        plt.ylabel('mAP (%)')
        plt.xticks(rotation=45)
        
        # Plot perbandingan waktu inferensi
        plt.subplot(2, 2, 4)
        sns.barplot(x='Skenario', y='Inference Time (ms)', hue='Backbone', data=success_results)
        plt.title('Perbandingan Waktu Inferensi (ms)')
        plt.ylabel('Waktu (ms)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "research_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Analisis dan rekomendasi
        print("\n🔍 Analisis & Rekomendasi:")
        
        # Bandingkan backbone
        efficientnet_results = success_results[success_results['Backbone'] == 'efficientnet']
        cspdarknet_results = success_results[success_results['Backbone'] == 'cspdarknet']
        
        if not efficientnet_results.empty and not cspdarknet_results.empty:
            # Hitung rata-rata metrik
            eff_avg_acc = efficientnet_results['Akurasi'].mean()
            csp_avg_acc = cspdarknet_results['Akurasi'].mean()
            
            eff_avg_time = efficientnet_results['Inference Time (ms)'].mean()
            csp_avg_time = cspdarknet_results['Inference Time (ms)'].mean()
            
            print(f"📊 Perbandingan Backbone:")
            print(f"• EfficientNet: Akurasi rata-rata {eff_avg_acc:.2f}%, Inferensi {eff_avg_time:.2f}ms")
            print(f"• CSPDarknet: Akurasi rata-rata {csp_avg_acc:.2f}%, Inferensi {csp_avg_time:.2f}ms")
            
            if eff_avg_acc > csp_avg_acc:
                if eff_avg_time < csp_avg_time:
                    print("\n💡 Rekomendasi: EfficientNet unggul dalam akurasi dan kecepatan")
                else:
                    print(f"\n💡 Rekomendasi: EfficientNet unggul dalam akurasi (+{eff_avg_acc-csp_avg_acc:.2f}%) "
                          f"tetapi lebih lambat (+{eff_avg_time-csp_avg_time:.2f}ms)")
            else:
                if csp_avg_time < eff_avg_time:
                    print("\n💡 Rekomendasi: CSPDarknet unggul dalam akurasi dan kecepatan")
                else:
                    print(f"\n💡 Rekomendasi: CSPDarknet unggul dalam akurasi (+{csp_avg_acc-eff_avg_acc:.2f}%) "
                          f"tetapi lebih lambat (+{csp_avg_time-eff_avg_time:.2f}ms)")


def plot_detections(
    image: np.ndarray, 
    detections: List[Dict], 
    output_path: Optional[str] = None,
    conf_threshold: float = 0.25,
    show_labels: bool = True
) -> np.ndarray:
    """
    Fungsi standalone untuk memvisualisasikan deteksi pada gambar.
    
    Args:
        image: Gambar input dalam format BGR (OpenCV)
        detections: List deteksi, masing-masing berbentuk dict 
        output_path: Path untuk menyimpan hasil (opsional)
        conf_threshold: Threshold confidence untuk deteksi yang ditampilkan
        show_labels: Apakah menampilkan label kelas
        
    Returns:
        Gambar dengan deteksi yang telah divisualisasikan
    """
    visualizer = ResultVisualizer()
    result_image = visualizer.plot_detections(
        image, 
        detections, 
        filename=os.path.basename(output_path) if output_path else None,
        conf_threshold=conf_threshold,
        show_labels=show_labels
    )
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, result_image)
    
    return result_image


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    output_path: Optional[str] = None,
    normalize: bool = True
) -> plt.Figure:
    """
    Fungsi standalone untuk memvisualisasikan confusion matrix.
    
    Args:
        confusion_matrix: Matriks konfusi
        class_names: List nama kelas
        title: Judul plot
        output_path: Path untuk menyimpan hasil (opsional)
        normalize: Apakah menormalisasi nilai dalam matriks
        
    Returns:
        Figure matplotlib dari confusion matrix
    """
    visualizer = ResultVisualizer()
    fig = visualizer.plot_confusion_matrix(
        confusion_matrix,
        class_names,
        title=title,
        filename=os.path.basename(output_path) if output_path else None,
        normalize=normalize
    )
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_training_metrics(
    metrics_history: Dict[str, List],
    title: str = "Training Metrics",
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Fungsi standalone untuk memvisualisasikan metrik training.
    
    Args:
        metrics_history: Dictionary berisi list metrik untuk setiap jenis metrik
        title: Judul plot
        output_path: Path untuk menyimpan hasil (opsional)
        
    Returns:
        Figure matplotlib dari plot metrik
    """
    visualizer = ResultVisualizer()
    fig = visualizer.plot_metrics_history(
        metrics_history,
        title=title,
        filename=os.path.basename(output_path) if output_path else None
    )
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig