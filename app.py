#!/usr/bin/env python
"""
File: app.py
Author: Alfrida Sabar
Deskripsi: Entry point utama untuk aplikasi SmartCash dengan antarmuka Streamlit.
           Versi minimalis tanpa Gradio untuk menghindari konflik dependensi.
"""

import os
import sys
import argparse
import logging
import yaml
import streamlit as st
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
import io

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join("logs", f"smartcash_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
    ]
)
logger = logging.getLogger(__name__)

# Lazy imports untuk menghindari circular imports dan mempercepat startup
def import_handler(module_name):
    """Import handler secara lazy."""
    if module_name == "model":
        from smartcash.handlers.model import ModelManager
        return ModelManager()
    elif module_name == "dataset":
        from smartcash.handlers.dataset import DatasetManager
        return DatasetManager()
    elif module_name == "detection":
        from smartcash.handlers.detection import DetectionManager
        return DetectionManager()
    elif module_name == "evaluation":
        from smartcash.handlers.evaluation import EvaluationManager
        return EvaluationManager()
    elif module_name == "preprocessing":
        from smartcash.handlers.preprocessing import PreprocessingManager
        return PreprocessingManager()
    elif module_name == "checkpoint":
        from smartcash.handlers.checkpoint import CheckpointManager
        return CheckpointManager()

def setup_dirs():
    """Buat direktori yang diperlukan jika belum ada."""
    os.makedirs("logs", exist_ok=True)
    os.makedirs("configs", exist_ok=True)
    os.makedirs("runs", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("pretrained", exist_ok=True)

def load_config():
    """Load konfigurasi dari file yaml."""
    try:
        config_path = st.session_state.get("config_path", "configs/base_config.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        st.error(f"Error loading config file: {e}")
        return {}

def is_colab():
    """Deteksi apakah berjalan di Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False

# Fungsi utilitas untuk UI
def get_image_list(folder_path):
    """Dapatkan daftar gambar dalam folder."""
    image_exts = [".jpg", ".jpeg", ".png", ".bmp"]
    if not os.path.exists(folder_path):
        return []
    return [f for f in os.listdir(folder_path) 
            if os.path.isfile(os.path.join(folder_path, f)) and 
            any(f.lower().endswith(ext) for ext in image_exts)]

def visualize_detection(img, detections, class_names=None, conf_threshold=0.25):
    """Visualisasikan hasil deteksi pada gambar."""
    # Clone gambar untuk menghindari modifikasi input
    if isinstance(img, np.ndarray):
        vis_img = img.copy()
    else:
        vis_img = np.array(img)
    
    # Convert to BGR untuk OpenCV jika perlu
    if len(vis_img.shape) == 3 and vis_img.shape[2] == 3:
        if vis_img.dtype != np.uint8:
            vis_img = (vis_img * 255).astype(np.uint8)
    
    height, width = vis_img.shape[:2]
    
    # Warna untuk setiap kelas
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), 
              (0, 255, 255), (255, 0, 255), (128, 128, 0)]
    
    for det in detections:
        if len(det) >= 6:  # [x1, y1, x2, y2, conf, class_id]
            x1, y1, x2, y2, conf, class_id = det[:6]
            
            if conf < conf_threshold:
                continue
                
            # Skala ke ukuran gambar jika perlu
            if x1 < 1 and x2 < 1 and y1 < 1 and y2 < 1:
                x1 = int(x1 * width)
                y1 = int(y1 * height)
                x2 = int(x2 * width)
                y2 = int(y2 * height)
            
            # Gambar bounding box
            color = colors[int(class_id) % len(colors)]
            cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Label kelas
            label = f"{class_names[int(class_id)] if class_names else class_id}: {conf:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_img, (int(x1), int(y1) - label_height - 5), 
                         (int(x1) + label_width, int(y1)), color, -1)
            cv2.putText(vis_img, label, (int(x1), int(y1) - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Convert BGR ke RGB untuk Streamlit
    if len(vis_img.shape) == 3 and vis_img.shape[2] == 3:
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    
    return vis_img

# Fungsi utama dan halaman
def main():
    """Fungsi utama aplikasi."""
    setup_dirs()
    
    st.set_page_config(
        page_title="SmartCash: Deteksi Nilai Mata Uang Rupiah",
        page_icon="💵",
        layout="wide",
    )
    
    st.title("🔍 SmartCash: Deteksi Nilai Mata Uang Rupiah")
    st.subheader("Sistem deteksi nilai mata uang Rupiah menggunakan YOLOv5 dengan berbagai backbone")
    
    # Sidebar untuk navigasi
    st.sidebar.title("Navigasi")
    pages = [
        "Home", 
        "Dataset & Preprocessing", 
        "Model Manager",
        "Training", 
        "Evaluasi", 
        "Deteksi", 
        "Tentang"
    ]
    selected_page = st.sidebar.selectbox("Pilih halaman", pages)
    
    # Konfigurasi
    if "config" not in st.session_state:
        st.session_state.config = load_config()
    
    # Tampilkan halaman yang dipilih
    if selected_page == "Home":
        show_home_page()
    elif selected_page == "Dataset & Preprocessing":
        show_dataset_page()
    elif selected_page == "Model Manager": 
        show_model_page()
    elif selected_page == "Training":
        show_training_page()
    elif selected_page == "Evaluasi":
        show_evaluation_page()
    elif selected_page == "Deteksi":
        show_detection_page()
    elif selected_page == "Tentang":
        show_about_page()

def show_home_page():
    """Tampilkan halaman utama."""
    st.markdown("""
    ## Selamat Datang di SmartCash
    
    SmartCash adalah sistem deteksi nilai mata uang Rupiah yang menggunakan model YOLOv5 
    dengan integrasi backbone EfficientNet-B4. Sistem ini dioptimasi untuk akurasi tinggi
    dalam berbagai kondisi pengambilan gambar.
    
    ### Fitur Utama
    - ✅ Deteksi mata uang Rupiah dengan akurasi tinggi
    - ✅ Dukungan backbone berganti (EfficientNet, CSPDarknet)
    - ✅ Pengelolaan dataset dan augmentasi
    - ✅ Training multi-layer detection
    - ✅ Evaluasi model dan perbandingan backbone
    - ✅ Pipeline preprocessing yang komprehensif
    
    ### Quick Start
    1. Buka halaman **Dataset & Preprocessing** untuk mengelola dataset
    2. Gunakan **Model Manager** untuk pengelolaan model
    3. Lakukan **Training** untuk melatih model baru
    4. Jalankan **Evaluasi** untuk menguji performa model
    5. Gunakan **Deteksi** untuk mendeteksi mata uang pada gambar
    """)
    
    # Status sistem
    st.subheader("📊 Status Sistem")
    
    # Layout kolom
    col1, col2 = st.columns(2)
    
    with col1:
        # Dataset info
        st.info("### Dataset")
        dataset_dir = os.path.join("data", "train")
        if os.path.exists(dataset_dir):
            image_count = len(get_image_list(os.path.join(dataset_dir, "images")))
            st.write(f"Training Images: {image_count}")
        else:
            st.write("Dataset belum diunduh")
            
        # Model info
        st.info("### Model")
        model_dir = os.path.join("runs", "train", "weights")
        if os.path.exists(model_dir):
            if os.path.exists(os.path.join(model_dir, "best.pt")):
                st.write("Model terlatih tersedia ✓")
            else:
                st.write("Model belum dilatih")
        else:
            st.write("Model belum dilatih")
    
    with col2:
        # Environment info
        st.info("### Environment")
        if is_colab():
            st.write("Running di Google Colab ☁️")
        else:
            st.write("Running di Local Environment 💻")
            
        # GPU info
        if torch.cuda.is_available():
            st.write(f"GPU: {torch.cuda.get_device_name(0)} ✓")
            st.write(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            st.write("GPU tidak tersedia ✗")

def show_dataset_page():
    """Tampilkan halaman dataset dan preprocessing."""
    st.header("📁 Dataset & Preprocessing")
    
    # Tab untuk berbagai fungsi
    tab1, tab2, tab3 = st.tabs(["📥 Download Dataset", "🔍 Validasi & Info", "🔄 Augmentasi"])
    
    with tab1:
        st.subheader("Download Dataset")
        
        # Sumber dataset
        source = st.radio("Sumber dataset", ["Roboflow", "Local"])
        
        if source == "Roboflow":
            # Roboflow info
            col1, col2 = st.columns(2)
            with col1:
                workspace = st.text_input("Workspace", "smartcash-wo2us")
                project = st.text_input("Project", "rupiah-emisi-2022")
            with col2:
                version = st.text_input("Version", "3")
                api_key = st.text_input("API Key", "", type="password")
            
            if st.button("📥 Download Dataset", key="download_dataset"):
                if api_key:
                    with st.spinner("Downloading dataset dari Roboflow..."):
                        try:
                            # Lazy load DatasetManager
                            dataset_manager = import_handler("dataset")
                            # Update config
                            config = st.session_state.config
                            config['data']['roboflow']['api_key'] = api_key
                            config['data']['roboflow']['workspace'] = workspace
                            config['data']['roboflow']['project'] = project
                            config['data']['roboflow']['version'] = version
                            
                            # Download dataset
                            dataset_manager.download_dataset()
                            st.success("✅ Dataset berhasil diunduh!")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                else:
                    st.warning("⚠️ API key diperlukan")
        else:
            # Local upload
            st.info("Upload dataset lokal (ZIP atau folder)")
            uploaded_file = st.file_uploader("Upload ZIP dataset", type=["zip"])
            
            if uploaded_file is not None:
                target_dir = st.text_input("Direktori ekstrak", "data")
                if st.button("📤 Proses Dataset", key="process_dataset"):
                    with st.spinner("Memproses dataset..."):
                        try:
                            import zipfile
                            with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                                zip_ref.extractall(target_dir)
                            st.success(f"✅ Dataset berhasil diekstrak ke {target_dir}!")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
    
    with tab2:
        st.subheader("Info Dataset & Validasi")
        
        # Split selector
        split = st.selectbox("Split dataset", ["train", "valid", "test"])
        data_dir = os.path.join("data", split)
        
        if os.path.exists(data_dir):
            if st.button("🔍 Refresh Info", key="refresh_info"):
                with st.spinner("Menganalisis dataset..."):
                    try:
                        # Lazy load PreprocessingManager
                        preprocessing_manager = import_handler("preprocessing")
                        stats = preprocessing_manager.analyze_dataset(split=split)
                        
                        # Save stats to session state
                        st.session_state.dataset_stats = stats
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            
            # Show stats if available
            if "dataset_stats" in st.session_state:
                stats = st.session_state.dataset_stats
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info("### Dataset Statistics")
                    st.write(f"Total images: {stats.get('num_images', 0)}")
                    st.write(f"Total annotations: {stats.get('num_annotations', 0)}")
                    st.write(f"Classes: {', '.join(stats.get('classes', []))}")
                    
                with col2:
                    st.info("### Class Distribution")
                    
                    # Create bar chart for class distribution
                    if 'class_counts' in stats:
                        class_counts = stats['class_counts']
                        df = pd.DataFrame({
                            'Class': list(class_counts.keys()),
                            'Count': list(class_counts.values())
                        })
                        st.bar_chart(df.set_index('Class'))
            
            # Validasi Dataset
            st.subheader("Validasi Dataset")
            fix_issues = st.checkbox("Fix issues otomatis", value=False)
            move_invalid = st.checkbox("Pindahkan file tidak valid", value=False)
            
            if st.button("🛠️ Validasi Dataset", key="validate_dataset"):
                with st.spinner("Validasi dataset..."):
                    try:
                        # Lazy load PreprocessingManager
                        preprocessing_manager = import_handler("preprocessing")
                        results = preprocessing_manager.validate_dataset(
                            split=split,
                            fix_issues=fix_issues,
                            move_invalid=move_invalid
                        )
                        
                        # Show validation results
                        st.success("✅ Validasi selesai!")
                        st.json(results)
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        else:
            st.warning(f"⚠️ Dataset {split} tidak ditemukan. Silahkan download dataset terlebih dahulu.")
    
    with tab3:
        st.subheader("Augmentasi Dataset")
        
        # Split hanya train
        aug_split = "train"
        aug_dir = os.path.join("data", aug_split)
        
        if os.path.exists(aug_dir):
            # Augmentation settings
            st.info("### Pengaturan Augmentasi")
            
            col1, col2 = st.columns(2)
            with col1:
                aug_types = st.multiselect(
                    "Tipe augmentasi", 
                    ["combined", "lighting", "position", "noise"], 
                    default=["combined"]
                )
                num_variations = st.slider("Jumlah variasi per gambar", 1, 10, 3)
            
            with col2:
                output_prefix = st.text_input("Prefix output", "aug")
                validate_results = st.checkbox("Validasi hasil augmentasi", value=True)
            
            if st.button("🔄 Augmentasi Dataset", key="augment_dataset"):
                if not aug_types:
                    st.warning("⚠️ Pilih minimal satu tipe augmentasi")
                else:
                    with st.spinner("Melakukan augmentasi dataset..."):
                        try:
                            # Lazy load PreprocessingManager
                            preprocessing_manager = import_handler("preprocessing")
                            results = preprocessing_manager.augment_dataset(
                                split=aug_split,
                                augmentation_types=aug_types,
                                num_variations=num_variations,
                                output_prefix=output_prefix,
                                validate_results=validate_results
                            )
                            
                            # Show augmentation results
                            st.success("✅ Augmentasi selesai!")
                            st.json(results)
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
            
            # Clean augmented images
            st.subheader("Bersihkan Hasil Augmentasi")
            if st.button("🧹 Bersihkan", key="clean_augmentation"):
                with st.spinner("Membersihkan hasil augmentasi..."):
                    try:
                        # Lazy load PreprocessingManager
                        preprocessing_manager = import_handler("preprocessing")
                        preprocessing_manager.clean_augmented_images(split=aug_split, prefix=output_prefix)
                        st.success("✅ Pembersihan selesai!")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        else:
            st.warning(f"⚠️ Dataset {aug_split} tidak ditemukan. Silahkan download dataset terlebih dahulu.")

def show_model_page():
    """Tampilkan halaman model manager."""
    st.header("🛠️ Model Manager")
    
    # Tab untuk berbagai fungsi
    tab1, tab2, tab3 = st.tabs(["⚙️ Setup Model", "💾 Checkpoint Manager", "📤 Export Model"])
    
    with tab1:
        st.subheader("Inisialisasi Model")
        
        # Model settings
        col1, col2 = st.columns(2)
        
        with col1:
            backbone = st.selectbox(
                "Backbone", 
                ["efficientnet_b4", "cspdarknet", "efficientnet_b0", "efficientnet_b3"]
            )
            pretrained = st.checkbox("Gunakan pretrained weights", value=True)
        
        with col2:
            num_classes = st.number_input("Jumlah kelas", 1, 20, 7)
            layers = st.multiselect(
                "Detection layers", 
                ["banknote", "nominal", "security"], 
                default=["banknote"]
            )
        
        # Inisialisasi Model
        if st.button("⚙️ Inisialisasi Model", key="init_model"):
            with st.spinner("Menginisialisasi model..."):
                try:
                    # Lazy load ModelManager
                    model_manager = import_handler("model")
                    
                    # Update config
                    config = st.session_state.config
                    config['model']['backbone'] = backbone
                    config['model']['pretrained'] = pretrained
                    config['layers'] = layers
                    
                    # Initialize model
                    model = model_manager.create_model(
                        backbone_type=backbone,
                        pretrained=pretrained,
                        detection_layers=layers
                    )
                    
                    # Save to session state
                    st.session_state.model = model
                    st.success(f"✅ Model berhasil diinisialisasi dengan backbone {backbone}!")
                    
                    # Get model summary
                    summary = model_manager.get_model_summary(model)
                    st.code(summary)
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # Visualisasi model
        if "model" in st.session_state and st.button("📊 Visualisasi Model", key="viz_model"):
            with st.spinner("Membuat visualisasi model..."):
                try:
                    # Lazy load ModelManager
                    model_manager = import_handler("model")
                    
                    # Visualize
                    viz_path = model_manager.visualize_model(st.session_state.model)
                    st.image(viz_path)
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    with tab2:
        st.subheader("Checkpoint Manager")
        
        # List checkpoints
        if st.button("📋 List Checkpoints", key="list_checkpoints"):
            with st.spinner("Mencari checkpoints..."):
                try:
                    # Lazy load CheckpointManager
                    checkpoint_manager = import_handler("checkpoint")
                    checkpoints = checkpoint_manager.list_checkpoints()
                    
                    if checkpoints:
                        st.success(f"✅ Ditemukan {len(checkpoints)} checkpoint!")
                        
                        # Group by type
                        for checkpoint_type, checkpoint_list in checkpoints.items():
                            st.info(f"### {checkpoint_type.title()} Checkpoints")
                            for cp in checkpoint_list:
                                st.write(f"📄 {cp}")
                    else:
                        st.warning("⚠️ Tidak ada checkpoint yang ditemukan")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # Load checkpoint
        st.subheader("Load Checkpoint")
        checkpoint_path = st.text_input("Path checkpoint", "runs/train/weights/best.pt")
        
        if os.path.exists(checkpoint_path) and st.button("💾 Load Checkpoint", key="load_checkpoint"):
            with st.spinner("Loading checkpoint..."):
                try:
                    # Lazy load ModelManager
                    model_manager = import_handler("model")
                    
                    # Load model
                    model = model_manager.load_from_checkpoint(checkpoint_path)
                    
                    # Save to session state
                    st.session_state.model = model
                    st.success(f"✅ Model berhasil dimuat dari {checkpoint_path}!")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # Cleanup checkpoints
        st.subheader("Cleanup Checkpoints")
        max_checkpoints = st.number_input("Max checkpoints per kategori", 1, 10, 3)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            keep_best = st.checkbox("Pertahankan best", value=True)
        with col2:
            keep_latest = st.checkbox("Pertahankan latest", value=True)
        with col3:
            max_epochs = st.number_input("Max epoch checkpoints", 0, 50, 5)
        
        if st.button("🧹 Cleanup Checkpoints", key="cleanup_checkpoints"):
            with st.spinner("Membersihkan checkpoints..."):
                try:
                    # Lazy load CheckpointManager
                    checkpoint_manager = import_handler("checkpoint")
                    result = checkpoint_manager.cleanup_checkpoints(
                        max_checkpoints=max_checkpoints,
                        keep_best=keep_best,
                        keep_latest=keep_latest,
                        max_epochs=max_epochs
                    )
                    st.success(f"✅ Pembersihan checkpoints selesai! {result['removed_count']} file dihapus.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    with tab3:
        st.subheader("Export Model")
        
        # Export settings
        col1, col2 = st.columns(2)
        
        with col1:
            model_path = st.text_input("Path model", "runs/train/weights/best.pt")
            export_format = st.selectbox(
                "Format export", 
                ["torchscript", "onnx", "coreml", "saved_model"],
                index=0
            )
        
        with col2:
            dynamic = st.checkbox("Dynamic input shape", value=False)
            half_precision = st.checkbox("Half precision (FP16)", value=True)
            
            img_size = st.number_input("Input size", 320, 1280, 640, step=32)
        
        if os.path.exists(model_path) and st.button("📤 Export Model", key="export_model"):
            with st.spinner(f"Mengexport model ke format {export_format}..."):
                try:
                    # Lazy load ModelManager
                    model_manager = import_handler("model")
                    
                    # Export model
                    output_path = model_manager.export_model(
                        model_path=model_path,
                        format=export_format,
                        dynamic=dynamic,
                        half_precision=half_precision,
                        img_size=(img_size, img_size)
                    )
                    
                    st.success(f"✅ Model berhasil diexport ke {output_path}!")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # Optimize untuk inferensi
        st.subheader("Optimasi Model")
        if os.path.exists(model_path) and st.button("⚡ Optimasi untuk Inferensi", key="optimize_model"):
            with st.spinner("Mengoptimasi model..."):
                try:
                    # Lazy load ModelManager
                    model_manager = import_handler("model")
                    
                    # Optimize model
                    opt_path = model_manager.optimize_for_inference(
                        model_path=model_path,
                        half_precision=half_precision,
                        img_size=(img_size, img_size)
                    )
                    
                    st.success(f"✅ Model berhasil dioptimasi ke {opt_path}!")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

def show_training_page():
    """Tampilkan halaman training."""
    st.header("🏋️ Training")
    
    # Training settings
    st.subheader("Pengaturan Training")
    
    col1, col2 = st.columns(2)
    
    with col1:
        epochs = st.number_input("Epochs", 1, 300, 50)
        batch_size = st.number_input("Batch size", 1, 128, 16)
        img_size = st.number_input("Image size", 320, 1280, 640, step=32)
    
    with col2:
        optimizer = st.selectbox("Optimizer", ["Adam", "AdamW", "SGD"])
        lr = st.number_input("Learning rate", 0.0001, 0.1, 0.01, format="%.4f")
        patience = st.number_input("Early stopping patience", 0, 50, 10)
    
    # Advanced settings
    with st.expander("Advanced Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            weight_decay = st.number_input("Weight decay", 0.0, 0.1, 0.0005, format="%.5f")
            momentum = st.number_input("Momentum (untuk SGD)", 0.0, 1.0, 0.937, format="%.3f")
            scheduler = st.selectbox("Scheduler", ["cosine", "linear", "step"])
        
        with col2:
            save_period = st.number_input("Save period (epochs)", 1, 20, 5)
            resume = st.checkbox("Resume dari checkpoint", value=False)
            resume_path = st.text_input("Resume path", "runs/train/weights/last.pt")
    
    # Start training
    if st.button("🚀 Mulai Training", key="start_training"):
        # Update config
        config = st.session_state.config
        config['training']['epochs'] = epochs
        config['training']['batch_size'] = batch_size
        config['training']['img_size'] = [img_size, img_size]
        config['training']['optimizer'] = optimizer
        config['training']['lr0'] = lr
        config['training']['early_stopping_patience'] = patience
        config['training']['weight_decay'] = weight_decay
        config['training']['momentum'] = momentum
        config['training']['scheduler'] = scheduler
        config['training']['save_period'] = save_period
        
        with st.spinner(f"Training model selama {epochs} epochs..."):
            try:
                # Buat progress bar custom
                progress_placeholder = st.empty()
                progress_bar = st.progress(0)
                metrics_placeholder = st.empty()
                
                # Plot placeholder untuk metrik
                plot_placeholder = st.empty()
                
                # Callback untuk update status
                metrics_history = {'train_loss': [], 'val_loss': [], 'mAP': []}
                
                def progress_callback(epoch, epoch_metrics, epoch_num, total_epochs):
                    """Callback untuk update progress bar dan metrik."""
                    # Update progress
                    progress = (epoch_num + 1) / total_epochs
                    progress_bar.progress(progress)
                    progress_placeholder.text(f"Epoch {epoch_num+1}/{total_epochs}")
                    
                    # Update metrics history
                    for k, v in epoch_metrics.items():
                        if k not in metrics_history:
                            metrics_history[k] = []
                        metrics_history[k].append(v)
                    
                    # Format metrics
                    metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in epoch_metrics.items() if isinstance(v, (int, float))])
                    metrics_placeholder.info(f"Metrik: {metrics_str}")
                    
                    # Plot loss curves
                    if len(metrics_history['train_loss']) > 1:
                        fig, ax = plt.subplots(figsize=(10, 4))
                        epochs_range = list(range(1, len(metrics_history['train_loss']) + 1))
                        
                        ax.plot(epochs_range, metrics_history['train_loss'], 'b-', label='Train Loss')
                        if 'val_loss' in metrics_history and metrics_history['val_loss']:
                            ax.plot(epochs_range, metrics_history['val_loss'], 'r-', label='Val Loss')
                        
                        ax.set_title('Training Progress')
                        ax.set_xlabel('Epoch')
                        ax.set_ylabel('Loss')
                        ax.legend()
                        
                        # Convert plot to image
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png')
                        buf.seek(0)
                        
                        # Display the plot
                        plot_placeholder.image(buf)
                        plt.close(fig)
                
                # Lazy load ModelManager
                model_manager = import_handler("model")
                
                # Create or load model
                if "model" not in st.session_state or st.session_state.model is None:
                    if resume and os.path.exists(resume_path):
                        model = model_manager.load_from_checkpoint(resume_path)
                        st.session_state.model = model
                    else:
                        # Create model from config
                        model = model_manager.create_model(
                            backbone_type=config['model']['backbone'],
                            pretrained=config['model']['pretrained']
                        )
                        st.session_state.model = model
                else:
                    model = st.session_state.model
                
                # Train model
                results = model_manager.train_model(
                    model=model,
                    batch_size=batch_size,
                    epochs=epochs,
                    img_size=img_size,
                    optimizer_name=optimizer,
                    lr=lr,
                    patience=patience,
                    callback=progress_callback,
                    resume=resume,
                    resume_path=resume_path if resume else None
                )
                
                # Show training results
                st.success("✅ Training selesai!")
                st.json(results)
                
                # Display best model path
                if 'best_checkpoint_path' in results:
                    st.info(f"📂 Best model tersimpan di: {results['best_checkpoint_path']}")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

def show_evaluation_page():
    """Tampilkan halaman evaluasi."""
    st.header("📊 Evaluasi Model")
    
    # Model path
    model_path = st.text_input("Path model", "runs/train/weights/best.pt")
    
    # Evaluation settings
    col1, col2 = st.columns(2)
    
    with col1:
        split = st.selectbox("Split dataset", ["test", "valid"])
        conf_thresh = st.slider("Confidence threshold", 0.1, 0.9, 0.25, 0.05)
    
    with col2:
        iou_thresh = st.slider("IoU threshold", 0.1, 0.9, 0.45, 0.05)
        batch_size = st.number_input("Batch size", 1, 64, 16)
    
    # Run evaluation
    if os.path.exists(model_path) and st.button("📊 Evaluasi Model", key="eval_model"):
        with st.spinner("Mengevaluasi model..."):
            try:
                # Lazy load EvaluationManager
                eval_manager = import_handler("evaluation")
                
                # Evaluate model
                results = eval_manager.evaluate_model(
                    model_path=model_path,
                    data_split=split,
                    conf_threshold=conf_thresh,
                    iou_threshold=iou_thresh,
                    batch_size=batch_size
                )
                
                # Show results
                st.success("✅ Evaluasi selesai!")
                
                # Display key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("mAP", f"{results.get('mAP', 0):.4f}")
                with col2:
                    st.metric("Precision", f"{results.get('precision', 0):.4f}")
                with col3:
                    st.metric("Recall", f"{results.get('recall', 0):.4f}")
                with col4:
                    st.metric("F1", f"{results.get('f1', 0):.4f}")
                
                # Display per-class metrics if available
                if 'class_metrics' in results:
                    st.subheader("Per-class Metrics")
                    
                    # Convert to DataFrame for display
                    class_metrics = []
                    for cls_name, metrics in results['class_metrics'].items():
                        metrics['class'] = cls_name
                        class_metrics.append(metrics)
                    
                    if class_metrics:
                        df = pd.DataFrame(class_metrics)
                        # Reorder columns to put class first
                        cols = ['class'] + [col for col in df.columns if col != 'class']
                        st.dataframe(df[cols])
                
                # Display confusion matrix if available
                if 'confusion_matrix' in results:
                    st.subheader("Confusion Matrix")
                    cm = results['confusion_matrix']
                    
                    # Convert to DataFrame for display
                    if isinstance(cm, list) or isinstance(cm, np.ndarray):
                        classes = results.get('classes', [f"Class {i}" for i in range(len(cm))])
                        cm_df = pd.DataFrame(cm, index=classes, columns=classes)
                        
                        # Plot confusion matrix
                        fig, ax = plt.subplots(figsize=(10, 8))
                        im = ax.imshow(cm_df, cmap='Blues')
                        
                        # Show all ticks and label them
                        ax.set_xticks(np.arange(len(classes)))
                        ax.set_yticks(np.arange(len(classes)))
                        ax.set_xticklabels(classes)
                        ax.set_yticklabels(classes)
                        
                        # Rotate x tick labels
                        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                        
                        # Loop over data dimensions and create text annotations
                        for i in range(len(classes)):
                            for j in range(len(classes)):
                                text = ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
                        
                        ax.set_title("Confusion Matrix")
                        fig.tight_layout()
                        
                        # Convert plot to image
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png')
                        buf.seek(0)
                        
                        # Display the plot
                        st.image(buf)
                        plt.close(fig)
                
                # Display all results
                with st.expander("Complete Results"):
                    st.json(results)
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Compare models
    st.subheader("Perbandingan Model")
    
    # Model paths
    model1 = st.text_input("Model 1", "runs/train/weights/best.pt")
    model2 = st.text_input("Model 2", "")
    
    if model2:
        if (os.path.exists(model1) and os.path.exists(model2) and 
            st.button("🔍 Bandingkan Model", key="compare_models")):
            with st.spinner("Membandingkan model..."):
                try:
                    # Lazy load EvaluationManager
                    eval_manager = import_handler("evaluation")
                    
                    # Compare models
                    comparison = eval_manager.compare_models(
                        model_paths=[model1, model2],
                        data_split=split,
                        conf_threshold=conf_thresh,
                        iou_threshold=iou_thresh
                    )
                    
                    # Show comparison
                    st.success("✅ Perbandingan selesai!")
                    
                    # Display key metrics comparison
                    st.subheader("Perbandingan Metrik")
                    
                    # Convert to DataFrame for display
                    model_names = [os.path.basename(p) for p in [model1, model2]]
                    metrics = ['mAP', 'precision', 'recall', 'f1', 'inference_time']
                    
                    comparison_data = {}
                    for i, model_name in enumerate(model_names):
                        comparison_data[model_name] = {
                            metric: comparison[i].get(metric, 0) 
                            for metric in metrics if metric in comparison[i]
                        }
                    
                    df = pd.DataFrame(comparison_data)
                    st.dataframe(df)
                    
                    # Plot comparison bar chart
                    st.subheader("Comparison Chart")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Extract metrics for plotting
                    plot_metrics = ['mAP', 'precision', 'recall', 'f1']
                    
                    # Set up bar positions
                    x = np.arange(len(plot_metrics))
                    width = 0.35
                    
                    # Get values for each model
                    m1_values = [comparison_data[model_names[0]].get(m, 0) for m in plot_metrics]
                    
                    rects1 = ax.bar(x - width/2, m1_values, width, label=model_names[0])
                    
                    if len(model_names) > 1:
                        m2_values = [comparison_data[model_names[1]].get(m, 0) for m in plot_metrics]
                        rects2 = ax.bar(x + width/2, m2_values, width, label=model_names[1])
                    
                    # Add labels and title
                    ax.set_ylabel('Scores')
                    ax.set_title('Model Comparison')
                    ax.set_xticks(x)
                    ax.set_xticklabels(plot_metrics)
                    ax.legend()
                    
                    # Convert plot to image
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png')
                    buf.seek(0)
                    
                    # Display the plot
                    st.image(buf)
                    plt.close(fig)
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

def show_detection_page():
    """Tampilkan halaman deteksi."""
    st.header("🔎 Deteksi Mata Uang")
    
    # Model path
    model_path = st.text_input("Path model", "runs/train/weights/best.pt")
    
    if not os.path.exists(model_path):
        st.warning(f"⚠️ Model tidak ditemukan di {model_path}")
    
    # Detection settings
    col1, col2 = st.columns(2)
    
    with col1:
        conf_thresh = st.slider("Confidence threshold", 0.1, 0.9, 0.25, 0.05)
        visualize = st.checkbox("Visualisasi hasil", value=True)
    
    with col2:
        iou_thresh = st.slider("IoU threshold", 0.1, 0.9, 0.45, 0.05)
        save_results = st.checkbox("Simpan hasil", value=True)
    
    # Upload image or use example
    st.subheader("Unggah Gambar")
    
    source_type = st.radio("Sumber gambar", ["Upload", "Contoh", "Kamera"])
    
    if source_type == "Upload":
        uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            with st.spinner("Memproses gambar..."):
                # Save uploaded file
                img_path = os.path.join("temp", uploaded_file.name)
                os.makedirs("temp", exist_ok=True)
                
                with open(img_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Show uploaded image
                img = Image.open(img_path)
                st.image(img, caption="Gambar diunggah", use_column_width=True)
                
                # Run detection button
                if os.path.exists(model_path) and st.button("🔍 Deteksi", key="detect_uploaded"):
                    try:
                        # Lazy load DetectionManager
                        detection_manager = import_handler("detection")
                        
                        # Run detection
                        results = detection_manager.detect(
                            source=img_path,
                            model_path=model_path,
                            conf_threshold=conf_thresh,
                            iou_threshold=iou_thresh,
                            visualize=visualize,
                            save_results=save_results
                        )
                        
                        # Show results
                        st.success("✅ Deteksi selesai!")
                        
                        # Display visualization
                        if visualize and 'visualization_path' in results:
                            st.image(results['visualization_path'], caption="Hasil deteksi", use_column_width=True)
                        
                        # Show detections
                        st.subheader("Deteksi")
                        
                        if 'detections' in results:
                            detections = results['detections']
                            
                            if len(detections) > 0:
                                # Create DataFrame for display
                                det_data = []
                                for det in detections:
                                    det_data.append({
                                        'Kelas': det.get('class_name', ''),
                                        'Confidence': f"{det.get('confidence', 0):.2f}",
                                        'Koordinat': f"{det.get('bbox_xyxy', [])}",
                                    })
                                
                                if det_data:
                                    st.dataframe(pd.DataFrame(det_data))
                            else:
                                st.info("Tidak ada objek terdeteksi")
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    
    elif source_type == "Contoh":
        # Example images folder
        example_dir = os.path.join("data", "test", "images")
        
        if os.path.exists(example_dir):
            examples = get_image_list(example_dir)
            
            if examples:
                selected_example = st.selectbox("Pilih gambar contoh", examples)
                example_path = os.path.join(example_dir, selected_example)
                
                # Show selected example
                img = Image.open(example_path)
                st.image(img, caption="Gambar contoh", use_column_width=True)
                
                # Run detection button
                if os.path.exists(model_path) and st.button("🔍 Deteksi", key="detect_example"):
                    try:
                        # Lazy load DetectionManager
                        detection_manager = import_handler("detection")
                        
                        # Run detection
                        results = detection_manager.detect(
                            source=example_path,
                            model_path=model_path,
                            conf_threshold=conf_thresh,
                            iou_threshold=iou_thresh,
                            visualize=visualize,
                            save_results=save_results
                        )
                        
                        # Show results
                        st.success("✅ Deteksi selesai!")
                        
                        # Display visualization
                        if visualize and 'visualization_path' in results:
                            st.image(results['visualization_path'], caption="Hasil deteksi", use_column_width=True)
                        
                        # Show detections
                        st.subheader("Deteksi")
                        
                        if 'detections' in results:
                            detections = results['detections']
                            
                            if len(detections) > 0:
                                # Create DataFrame for display
                                det_data = []
                                for det in detections:
                                    det_data.append({
                                        'Kelas': det.get('class_name', ''),
                                        'Confidence': f"{det.get('confidence', 0):.2f}",
                                        'Koordinat': f"{det.get('bbox_xyxy', [])}",
                                    })
                                
                                if det_data:
                                    st.dataframe(pd.DataFrame(det_data))
                            else:
                                st.info("Tidak ada objek terdeteksi")
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("⚠️ Tidak ada gambar contoh. Silahkan download dataset terlebih dahulu.")
        else:
            st.warning(f"⚠️ Direktori contoh {example_dir} tidak ditemukan.")
    
    elif source_type == "Kamera":
        # Capture from camera
        img_file = st.camera_input("Ambil gambar")
        
        if img_file is not None:
            with st.spinner("Memproses gambar..."):
                # Save captured image
                img_path = os.path.join("temp", "camera_capture.jpg")
                os.makedirs("temp", exist_ok=True)
                
                with open(img_path, "wb") as f:
                    f.write(img_file.getbuffer())
                
                # Run detection button
                if os.path.exists(model_path) and st.button("🔍 Deteksi", key="detect_camera"):
                    try:
                        # Lazy load DetectionManager
                        detection_manager = import_handler("detection")
                        
                        # Run detection
                        results = detection_manager.detect(
                            source=img_path,
                            model_path=model_path,
                            conf_threshold=conf_thresh,
                            iou_threshold=iou_thresh,
                            visualize=visualize,
                            save_results=save_results
                        )
                        
                        # Show results
                        st.success("✅ Deteksi selesai!")
                        
                        # Display visualization
                        if visualize and 'visualization_path' in results:
                            st.image(results['visualization_path'], caption="Hasil deteksi", use_column_width=True)
                        
                        # Show detections
                        st.subheader("Deteksi")
                        
                        if 'detections' in results:
                            detections = results['detections']
                            
                            if len(detections) > 0:
                                # Create DataFrame for display
                                det_data = []
                                for det in detections:
                                    det_data.append({
                                        'Kelas': det.get('class_name', ''),
                                        'Confidence': f"{det.get('confidence', 0):.2f}",
                                        'Koordinat': f"{det.get('bbox_xyxy', [])}",
                                    })
                                
                                if det_data:
                                    st.dataframe(pd.DataFrame(det_data))
                            else:
                                st.info("Tidak ada objek terdeteksi")
                    
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

def show_about_page():
    """Tampilkan halaman tentang aplikasi."""
    st.header("ℹ️ Tentang SmartCash")
    
    st.markdown("""
    ## SmartCash: Deteksi Nilai Mata Uang Rupiah
    
    SmartCash adalah sistem deteksi nilai mata uang Rupiah yang menggunakan YOLOv5 dengan berbagai backbone, 
    termasuk EfficientNet. Sistem ini dioptimasi untuk akurasi tinggi dalam berbagai kondisi pengambilan gambar.
    
    ### Fitur Utama
    - **Deteksi Multi-layer**: Deteksi mata uang penuh, area nominal, dan fitur keamanan
    - **Backbone Fleksibel**: Dukungan untuk EfficientNet-B4 dan CSPDarknet
    - **Dataset Manager**: Pengelolaan dataset multilayer
    - **Augmentasi**: Augmentasi gambar dengan berbagai teknik
    - **Evaluasi**: Evaluasi model dan perbandingan backbone
    
    ### Arsitektur
    - **YOLOv5**: Basis arsitektur deteksi
    - **EfficientNet-B4**: Backbone utama untuk ekstraksi fitur
    - **FPN-PAN**: Feature processing untuk fitur multiscale
    
    ### Struktur Project
    Project diorganisasi dengan pola desain modular:
    - **Handlers**: Pengelolaan operasi kompleks (dataset, model, detection, dll)
    - **Models**: Definisi model dan arsitektur
    - **Utils**: Utilitas untuk berbagai fungsi
    - **UI Components**: Komponen antarmuka pengguna
    
    ### Lisensi / License
    Project ini dilisensikan di bawah [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
    
    &copy; 2024 Alfrida Sabar
    """)
    
    # System info
    st.subheader("📊 Info Sistem")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("### Environment")
        st.write(f"Python version: {sys.version.split(' ')[0]}")
        st.write(f"OS: {os.name.upper()}")
        if is_colab():
            st.write("Running di Google Colab ☁️")
        else:
            st.write("Running di Local Environment 💻")
    
    with col2:
        st.info("### Hardware")
        if torch.cuda.is_available():
            st.write(f"GPU: {torch.cuda.get_device_name(0)} ✓")
            st.write(f"CUDA Version: {torch.version.cuda}")
            props = torch.cuda.get_device_properties(0)
            st.write(f"Memory: {props.total_memory / 1e9:.2f} GB")
            st.write(f"Compute Capability: {props.major}.{props.minor}")
        else:
            st.write("GPU tidak tersedia ✗")
            st.write("Running on CPU")

if __name__ == "__main__":
    main()