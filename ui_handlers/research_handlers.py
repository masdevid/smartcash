"""
File: smartcash/ui_handlers/research_handlers.py
Author: Alfrida Sabar
Deskripsi: Handler untuk UI komponen evaluasi skenario penelitian.
"""

import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ipywidgets as widgets  # Added missing import
from IPython.display import display, clear_output, HTML
from pathlib import Path


def run_research_evaluation(ui_components, research_handler, logger):
    """
    Jalankan evaluasi skenario penelitian.
    
    Args:
        ui_components: Dictionary komponen UI dari create_research_ui()
        research_handler: Instance dari ResearchScenarioHandler
        logger: Logger untuk mencatat aktivitas
        
    Returns:
        DataFrame hasil evaluasi
    """
    # Get selected scenarios
    selected_scenarios = []
    for i, checkbox in enumerate(ui_components['scenario_checkboxes']):
        if checkbox.value:
            selected_scenarios.append(ui_components['scenarios'][i])
    
    if not selected_scenarios:
        logger.warning("⚠️ Tidak ada skenario yang dipilih")
        return pd.DataFrame()
    
    # Cek keberadaan direktori output
    os.makedirs('runs/evaluation', exist_ok=True)
    
    # Siapkan storage untuk hasil
    results_data = []
    
    # Jalankan evaluasi untuk setiap skenario
    for scenario in selected_scenarios:
        try:
            logger.info(f"🔬 Evaluasi skenario: {scenario['name']} - {scenario['description']}")
            
            # Get model path untuk skenario ini
            model_path = f"runs/train/weights/{scenario['backbone']}_{scenario['conditions']}_best.pth"
            
            # Cek keberadaan model
            if not os.path.exists(model_path):
                # Cari model dengan pola serupa jika tidak ada
                potential_models = list(Path('runs/train/weights').glob(f"*{scenario['backbone']}*{scenario['conditions']}*best*.pth"))
                if potential_models:
                    model_path = str(potential_models[0])
                    logger.info(f"🔍 Menggunakan alternatif model path: {model_path}")
                else:
                    logger.warning(f"⚠️ Model tidak ditemukan untuk skenario {scenario['name']}")
                    # Dummy result untuk skenario yang tidak dapat dievaluasi
                    results_data.append({
                        'Skenario': scenario['name'],
                        'Deskripsi': scenario['description'],
                        'Backbone': scenario['backbone'],
                        'Kondisi': scenario['conditions'],
                        'Status': 'Model tidak ditemukan',
                    })
                    continue
            
            # Evaluasi model untuk skenario ini
            evaluation_result = research_handler.evaluate_scenario(
                scenario['name'],
                model_path,
                scenario['test_data']
            )
            
            # Ambil metrik dari hasil evaluasi
            metrics = evaluation_result.get('metrics', {})
            
            # Simpan hasil evaluasi
            results_data.append({
                'Skenario': scenario['name'],
                'Deskripsi': scenario['description'],
                'Backbone': scenario['backbone'],
                'Kondisi': scenario['conditions'],
                'Status': 'Sukses',
                'Akurasi': metrics.get('accuracy', 0) * 100,
                'Precision': metrics.get('precision', 0) * 100,
                'Recall': metrics.get('recall', 0) * 100,
                'F1-Score': metrics.get('f1', 0) * 100,
                'mAP': metrics.get('mAP', 0) * 100,
                'Inference Time (ms)': metrics.get('inference_time', 0)
            })
            
        except Exception as e:
            logger.error(f"❌ Error pada skenario {scenario['name']}: {str(e)}")
            results_data.append({
                'Skenario': scenario['name'],
                'Deskripsi': scenario['description'],
                'Backbone': scenario['backbone'],
                'Kondisi': scenario['conditions'],
                'Status': f'Error: {str(e)}',
            })
    
    # Convert ke DataFrame
    results_df = pd.DataFrame(results_data)
    
    # Simpan hasil untuk referensi future
    results_df.to_csv('runs/evaluation/research_results.csv', index=False)
    
    return results_df

def visualize_research_results(results_df, visualizer=None, logger=None):
    """
    Visualisasikan hasil evaluasi skenario penelitian.
    
    Args:
        results_df: DataFrame hasil evaluasi dari run_research_evaluation()
        visualizer: Instance dari ModelVisualizer atau ResultVisualizer
        logger: Logger untuk mencatat aktivitas
    """
    if results_df.empty:
        if logger:
            logger.warning("⚠️ Tidak ada hasil penelitian untuk divisualisasikan")
        else:
            print("⚠️ Tidak ada hasil penelitian untuk divisualisasikan")
        return
    
    try:
        # Filter hanya skenario yang berhasil
        success_results = results_df[results_df['Status'] == 'Sukses'].copy()
        
        if success_results.empty:
            if logger:
                logger.warning("⚠️ Tidak ada skenario yang berhasil dievaluasi")
            else:
                print("⚠️ Tidak ada skenario yang berhasil dievaluasi")
            display(results_df)
            return
        
        # Cek apakah visualizer tersedia untuk hasil penelitian
        if (visualizer is not None and 
            (hasattr(visualizer, 'visualize_research_comparison') or 
             hasattr(visualizer, 'plot_research_comparison'))):
            # Gunakan visualizer yang tersedia
            if hasattr(visualizer, 'visualize_research_comparison'):
                visualizer.visualize_research_comparison(success_results)
            else:
                visualizer.plot_research_comparison(success_results)
            return
        
        # Fallback visualization jika tidak ada visualizer
        # Tampilkan tabel utama hasil evaluasi
        display(HTML("<h3>📊 Hasil Evaluasi Skenario Penelitian</h3>"))
        
        # Style DataFrame untuk tampilan yang lebih baik
        styled_df = success_results.style.format({
            'Akurasi': '{:.2f}%',
            'Precision': '{:.2f}%',
            'Recall': '{:.2f}%',
            'F1-Score': '{:.2f}%',
            'mAP': '{:.2f}%',
            'Inference Time (ms)': '{:.2f}'
        })
        
        # Highlight skenario terbaik untuk setiap metrik
        styled_df = styled_df.highlight_max(
            subset=['Akurasi', 'Precision', 'Recall', 'F1-Score', 'mAP'],
            color='lightgreen'
        )
        
        # Highlight skenario tercepat (inference time paling rendah)
        styled_df = styled_df.highlight_min(
            subset=['Inference Time (ms)'],
            color='lightgreen'
        )
        
        # Tampilkan tabel dengan format
        display(styled_df)
        
        # Visualisasi perbandingan
        plt.figure(figsize=(15, 10))
        
        # Plot Metrik Akurasi (Heatmap) - Subplot 1
        plt.subplot(2, 2, 1)
        
        # Set metrik untuk heatmap
        metrics_for_heatmap = success_results.set_index('Skenario')[
            ['Akurasi', 'Precision', 'Recall', 'F1-Score', 'mAP']
        ]
        
        sns.heatmap(metrics_for_heatmap, annot=True, fmt='.1f', cmap='YlGnBu', linewidths=.5)
        plt.title('Perbandingan Metrik Akurasi per Skenario (%)')
        
        # Plot Inference Time - Subplot 2
        plt.subplot(2, 2, 2)
        
        # Buat barplot
        sns.barplot(
            x='Skenario',
            y='Inference Time (ms)',
            hue='Backbone',
            data=success_results,
            palette=['#3498db', '#e74c3c']
        )
        plt.title('Perbandingan Waktu Inferensi (ms)')
        plt.ylabel('Waktu (ms)')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Hitung dan Plot FPS - Subplot 3
        plt.subplot(2, 2, 3)
        
        # Tambahkan kolom FPS
        success_results['FPS'] = 1000 / success_results['Inference Time (ms)']
        
        sns.barplot(
            x='Skenario',
            y='FPS',
            hue='Backbone',
            data=success_results,
            palette=['#3498db', '#e74c3c']
        )
        plt.title('Perbandingan FPS (Frame per Second)')
        plt.ylabel('FPS')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot radar chart - Subplot 4
        ax = plt.subplot(2, 2, 4, polar=True)
        
        # Create radar chart
        _create_radar_chart(
            ax=ax,
            metrics=['Akurasi', 'Precision', 'Recall', 'F1-Score', 'mAP'],
            skenarios=success_results['Skenario'].tolist(),
            data=success_results
        )
        plt.title('Radar Chart Metrik Performa', y=1.1)
        
        plt.tight_layout()
        plt.show()
        
        # Tampilkan analisis dan rekomendasi
        display(HTML("<h3>🔍 Analisis & Kesimpulan</h3>"))
        
        # Temukan skenario terbaik untuk setiap metrik
        best_accuracy = success_results.loc[success_results['Akurasi'].idxmax()]
        best_f1 = success_results.loc[success_results['F1-Score'].idxmax()]
        best_map = success_results.loc[success_results['mAP'].idxmax()]
        fastest = success_results.loc[success_results['Inference Time (ms)'].idxmin()]
        
        print(f"🏆 Skenario dengan akurasi tertinggi: {best_accuracy['Skenario']} ({best_accuracy['Akurasi']:.2f}%)")
        print(f"🏆 Skenario dengan F1-Score tertinggi: {best_f1['Skenario']} ({best_f1['F1-Score']:.2f}%)")
        print(f"🏆 Skenario dengan mAP tertinggi: {best_map['Skenario']} ({best_map['mAP']:.2f}%)")
        print(f"⚡ Skenario tercepat: {fastest['Skenario']} ({fastest['Inference Time (ms)']:.2f}ms, {1000/fastest['Inference Time (ms)']:.1f} FPS)")
        
        # Analisis backbone
        efficientnet_results = success_results[success_results['Backbone'] == 'efficientnet']
        cspdarknet_results = success_results[success_results['Backbone'] == 'cspdarknet']
        
        if not efficientnet_results.empty and not cspdarknet_results.empty:
            # Hitung rata-rata metrik per backbone
            efficientnet_avg = efficientnet_results[['Akurasi', 'Precision', 'Recall', 'F1-Score', 'mAP', 'Inference Time (ms)']].mean()
            cspdarknet_avg = cspdarknet_results[['Akurasi', 'Precision', 'Recall', 'F1-Score', 'mAP', 'Inference Time (ms)']].mean()
            
            print("\n📈 Perbandingan Backbone:")
            print(f"• EfficientNet-B4 - Akurasi rata-rata: {efficientnet_avg['Akurasi']:.2f}%, Inference time: {efficientnet_avg['Inference Time (ms)']:.2f}ms")
            print(f"• CSPDarknet - Akurasi rata-rata: {cspdarknet_avg['Akurasi']:.2f}%, Inference time: {cspdarknet_avg['Inference Time (ms)']:.2f}ms")
            
            # Buat rekomendasi
            accuracy_diff = efficientnet_avg['Akurasi'] - cspdarknet_avg['Akurasi']
            speed_diff = cspdarknet_avg['Inference Time (ms)'] - efficientnet_avg['Inference Time (ms)']
            
            if accuracy_diff > 0 and speed_diff > 0:
                print("\n💡 Rekomendasi: EfficientNet-B4 lebih unggul secara keseluruhan, dengan akurasi lebih tinggi dan inferensi lebih cepat.")
            elif accuracy_diff > 0:
                print(f"\n💡 Rekomendasi: EfficientNet-B4 memberikan akurasi lebih tinggi (+{accuracy_diff:.2f}%), namun CSPDarknet lebih cepat (+{-speed_diff:.2f}ms).")
                if accuracy_diff > abs(speed_diff):
                    print("     👉 Untuk aplikasi yang mengutamakan akurasi, gunakan EfficientNet-B4.")
                else:
                    print("     👉 Pilih backbone berdasarkan kebutuhan: akurasi (EfficientNet-B4) atau kecepatan (CSPDarknet).")
            elif speed_diff > 0:
                print(f"\n💡 Rekomendasi: CSPDarknet memberikan akurasi lebih tinggi (+{-accuracy_diff:.2f}%), namun EfficientNet-B4 lebih cepat (+{speed_diff:.2f}ms).")
                if abs(accuracy_diff) > speed_diff:
                    print("     👉 Untuk aplikasi yang mengutamakan akurasi, gunakan CSPDarknet.")
                else:
                    print("     👉 Pilih backbone berdasarkan kebutuhan: akurasi (CSPDarknet) atau kecepatan (EfficientNet-B4).")
            else:
                print("\n💡 Rekomendasi: CSPDarknet lebih unggul secara keseluruhan, dengan akurasi lebih tinggi dan inferensi lebih cepat.")
    
    except Exception as e:
        if logger:
            logger.error(f"❌ Error saat visualisasi hasil penelitian: {str(e)}")
        else:
            print(f"❌ Error saat visualisasi hasil penelitian: {str(e)}")
        import traceback
        traceback.print_exc()

def _create_radar_chart(ax, metrics, skenarios, data):
    """
    Fungsi helper untuk membuat radar chart.
    
    Args:
        ax: Matplotlib axes untuk plot
        metrics: List nama metrik yang akan ditampilkan
        skenarios: List nama skenario
        data: DataFrame dengan data metrics
    """
    # Number of variables
    N = len(metrics)
    
    # Angle for each variable
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Initialize spider plot
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], metrics)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([20, 40, 60, 80, 100], ["20%", "40%", "60%", "80%", "100%"], color="grey", size=8)
    plt.ylim(0, 100)
    
    # Plot each scenario
    for skenario in skenarios:
        row = data[data['Skenario'] == skenario]
        if not row.empty:
            values = row[metrics].values[0].tolist()
            values += values[:1]  # Close the loop
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=skenario)
            ax.fill(angles, values, alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

def on_run_button_clicked(ui_components, research_handler, visualizer, logger):
    """
    Handler untuk tombol evaluasi model.
    
    Args:
        ui_components: Dictionary komponen UI dari create_research_ui()
        research_handler: Instance dari ResearchScenarioHandler
        visualizer: Instance dari ModelVisualizer atau ResultVisualizer (optional)
        logger: Logger untuk mencatat aktivitas
    """
    # Disable button during evaluation
    ui_components['run_button'].disabled = True
    ui_components['run_button'].description = "Evaluasi Berjalan..."
    
    with ui_components['output']:
        clear_output()
        
        print(f"🔍 Menjalankan evaluasi untuk skenario yang dipilih...")
        
        try:
            # Run evaluation
            results = run_research_evaluation(ui_components, research_handler, logger)
            
            # Visualize results
            if not results.empty:
                visualize_research_results(results, visualizer, logger)
                
                # Save results
                current_time = time.strftime("%Y%m%d_%H%M%S")
                os.makedirs('runs/evaluation', exist_ok=True)
                results.to_csv(f'runs/evaluation/results_{current_time}.csv', index=False)
                print(f"✅ Hasil evaluasi disimpan ke runs/evaluation/results_{current_time}.csv")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            # Re-enable button
            ui_components['run_button'].disabled = False
            ui_components['run_button'].description = "Jalankan Evaluasi"

def load_existing_results(ui_components):
    """
    Load hasil evaluasi yang sudah ada.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        DataFrame hasil evaluasi atau None jika tidak ada
    """
    results_file = Path('runs/evaluation/research_results.csv')
    if results_file.exists():
        try:
            results = pd.read_csv(results_file)
            with ui_components['output']:
                print(f"📋 Hasil evaluasi sebelumnya terdeteksi ({len(results)} skenario)")
                display(HTML("<h3>📊 Hasil Evaluasi Sebelumnya</h3>"))
                display(results)
                
                # Add button to visualize existing results
                visualize_button = widgets.Button(
                    description='Visualisasikan Hasil',
                    button_style='info',
                    icon='chart-line'
                )
                
                display(visualize_button)
                
                # Akan diatur handler-nya di setup_research_handlers
                
            return results, visualize_button
        except Exception as e:
            print(f"⚠️ Gagal memuat hasil evaluasi sebelumnya: {str(e)}")
    
    return None, None

def setup_research_handlers(ui_components, research_handler, visualizer=None, logger=None):
    """
    Setup semua event handlers untuk UI evaluasi penelitian.
    
    Args:
        ui_components: Dictionary komponen UI
        research_handler: Instance dari ResearchScenarioHandler
        visualizer: Instance dari ModelVisualizer atau ResultVisualizer (optional)
        logger: Logger untuk mencatat aktivitas
        
    Returns:
        Dictionary updated UI components dengan handlers yang sudah di-attach
    """
    # Setup handler untuk tombol evaluasi
    ui_components['run_button'].on_click(
        lambda b: on_run_button_clicked(ui_components, research_handler, visualizer, logger)
    )
    
    # Load dan setup handler untuk existing results jika tersedia
    existing_results, visualize_button = load_existing_results(ui_components)
    if existing_results is not None and visualize_button is not None:
        visualize_button.on_click(
            lambda b: visualize_research_results(existing_results, visualizer, logger)
        )
        ui_components['existing_results'] = existing_results
        ui_components['visualize_button'] = visualize_button
    
    return ui_components