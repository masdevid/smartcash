# Model API Refactoring Plan

Refactor `smartcash/model/**` from scratch that can have comprehensive progress tracker bridge, summary reports and metrics tracker with ui module.
Training and Evaluation will have seperate UI module. Training will focus on model building and training. Evaluation phase will focus on research scenarios evaluation using prebuilt model.

**Minimal Folder Structure:**
```
smartcash/model/
├── __init__.py
├── api/
├── core/
├── utils/
└── configs/
    └── model_config.yaml
```
**Rules:**
- It should compatible with `Progress Tracker API Documentation.md`
- Create new `model_config.yaml` and `evaluation_config.yaml` and all code related to model domain use this config as default if not provided. Essential config only please related to model configuration rather UI/UX configurations. 
- always auto creating missing directories
- do not generate code more than 400 lines, split it into multiple files if necessary
- When building the model, provide error trace to the user with progress tracker if error occurred. UI module will suppress the error trace.
**Backbone Options:**
- baseline: cspdarknet for yolov5s
- default: efficiennet_b4 for yolov5s
**Feature Optimization:**
- default: False
- Feature selections must be configurable through api params
**Data loading:**
- model: use pretrained model (yolov5s.pt) available in `/data/pretrained` for cspdarknet. 
  * If pretrained model not available, use pretrained model in `ultralytics` for yolov5s and `timm` for efficiennet_b4
- dataset: use dataset available in `/data/preprocessed/{splits}`
  * use `train` and `valid` splits to train model
  * use `test` split to evaluate model with custom research scenario augmentation
**Callbacks:**
- Progress Callback: Progress Tracker Bridge for updating progress tracker
- Report Callback: Summary Reports for logging in UI
- Metrics Callback: Metrics Tracker for updating metrics cards, plot chart and confussion matrics
**Progress Tracker:**
- Overall Pipeline progress tracking
- Per pipeline's step progress tracking
- Per sub steps progress tracking (if any)
**Metrics:**
- mAP (train and evaluation phase)
- loss (train phase only)
- accuracy (evaluation phase only)
- precission (evaluation phase only)
- f1 score (evaluation phase only)
- inference times (evaluation phase only)
**Checkpoints:**
- Always save best checkpoints with format: `best_{model_name}_{model_backbone}_{single|multi}_{MMDDYYYY}.pt`
- Evaluation phase will have seperate UI module, so make sure to have configurable checkpoint selection for testing different evaluation scenarios
- Provide api to list available checkpoints for evaluation phase
- Those are defaults metrics settings. It's configurable from api params wether some metrics used in train/evaluation phase or both
**Evaluation Scenarios**
- Deteksi nilai uang dengan model YOLOv5 Default (CSPDarknet) sebagai baseline backbone dengan variasi posisi pengambilan gambar
- Deteksi nilai uang dengan model YOLOv5 Default (CSPDarknet) sebagai baseline backbone dengan variasi pencahayaan
- Deteksi nilai uang dengan model YOLOv5 dengan arsitektur EfficientNet-B4 sebagai backbone dengan variasi posisi pengambilan gambar
- Deteksi nilai uang dengan model YOLOv5 dengan arsitektur EfficientNet-B4 sebagai backbone dengan variasi pencahayaan
**Analysis:**
- Currency Denomination detection metrics analysis (main analysis)
  * Use banknote layer as main layer detection
  * Use nominal layer as confidence boost when banknote layer is undetected
  * Use security layer as validation layer wether object is a valid money or just general images or numbers objects. At least detected 1 class of security layer to validate object as valid money
- Per class object detection metrics analysis
- Per layer object detection metrics analysis


# 🚀 SmartCash Model Development Plan

## **📋 Fase Development Overview**

Refactor `smartcash/model/**` menjadi comprehensive model API dengan progress tracker integration, summary reports, dan metrics tracker yang compatible dengan UI module.

### **🎯 Development Objectives**
- Model API yang modular dan scalable
- Progress tracking integration dengan UI
- Backbone selection (CSPDarknet baseline vs EfficientNet-B4 enhanced)  
- Training dan Evaluation sebagai module terpisah
- Research scenarios evaluation support
- Comprehensive metrics dan analysis

---

## **Fase 1: Core Model API & Configuration** ✅ **SELESAI**
**Prioritas:** Critical foundation untuk semua fase selanjutnya

### **Deliverables:**
- ✅ Core model API dengan progress tracker integration
- ✅ Configuration management (model_config.yaml) 
- ✅ Backbone factory (CSPDarknet vs EfficientNet-B4)
- ✅ Device management utilities
- ✅ Checkpoint management system
- ✅ Progress bridge untuk UI integration

### **Components:**
- `SmartCashModelAPI` - Main API entry point
- `ModelBuilder` - Modular model construction
- `BackboneFactory` - CSPDarknet/EfficientNet-B4 selection
- `CheckpointManager` - Automatic checkpoint naming & management
- `ModelProgressBridge` - UI progress tracker integration
- `DeviceUtils` - CUDA optimization & management

---

## **Fase 2: Training Pipeline** 🎯 **NEXT SESSION**
**Prioritas:** High - Training service dengan comprehensive tracking

### **Planned Deliverables:**
- ✅ `TrainingService` - Main training loop dengan UI integration
- ✅ `DataLoaderFactory` - Dataset loading dari `/data/preprocessed`
- ✅ `MetricsTracker` - mAP, loss collection selama training
- ✅ `OptimizerFactory` - Adam/SGD dengan scheduler support
- ✅ `TrainingProgressBridge` - Detailed training progress tracking
- ✅ `LossManager` - Multi-layer loss calculation

### **Key Features:**
- Automatic checkpoint saving dengan format: `best_{model}_{backbone}_{mode}_{date}.pt`
- Real-time metrics tracking (mAP, loss)
- Progress tracking: overall → epoch → batch level
- Early stopping dengan patience configuration
- Mixed precision training support
- Training resume dari checkpoint

---

## **Fase 3: Evaluation Pipeline** 🔲 **FUTURE SESSION**
**Prioritas:** Medium - Research scenarios evaluation

### **Planned Deliverables:**
- ✅ `EvaluationService` - Separate evaluation module
- ✅ `ScenarioManager` - Research scenarios implementation
- ✅ `EvaluationMetrics` - Accuracy, precision, F1, inference time
- ✅ `CheckpointSelector` - UI untuk pilih checkpoint evaluation
- ✅ `TestAugmentation` - Custom research scenario augmentation

### **Research Scenarios:**
1. **Posisi Pengambilan Gambar:**
   - CSPDarknet baseline dengan variasi posisi
   - EfficientNet-B4 dengan variasi posisi
2. **Variasi Pencahayaan:**
   - CSPDarknet baseline dengan variasi lighting
   - EfficientNet-B4 dengan variasi lighting

### **Evaluation Metrics:**
- mAP (evaluation phase)
- Accuracy (evaluation phase only)
- Precision (evaluation phase only)  
- F1 Score (evaluation phase only)
- Inference Times (evaluation phase only)

---

## **Fase 4: Analysis & Reporting** 🔲 **FINAL SESSION**
**Prioritas:** Medium - Comprehensive analysis dan visualization

### **Planned Deliverables:**
- 🔲 `CurrencyAnalyzer` - Main currency denomination analysis
- 🔲 `LayerAnalyzer` - Multi-layer detection analysis  
- 🔲 `ClassAnalyzer` - Per-class metrics analysis
- 🔲 `ReportGenerator` - Summary reports generation
- 🔲 `VisualizationManager` - Charts, confusion matrix, plots

### **Analysis Types:**
1. **Currency Denomination Analysis (Primary):**
   - Banknote layer sebagai main detection
   - Nominal layer sebagai confidence boost
   - Security layer sebagai validation layer
2. **Per-Class Object Detection Analysis**
3. **Per-Layer Object Detection Analysis**

---

## **🏗️ Technical Architecture**

### **Progress Tracker Integration:**
- **Overall Pipeline:** 0-100% untuk entire operation
- **Per Pipeline Step:** Sub-progress untuk each major phase  
- **Sub-steps:** Granular progress dalam each step

### **Callbacks Support:**
- **Progress Callback:** Update UI progress tracker
- **Report Callback:** Summary logging ke UI
- **Metrics Callback:** Real-time metrics update (cards, charts, confusion matrix)

### **Data Flow:**
```
Raw Data → Preprocessed → Model Training → Checkpoints → Evaluation → Analysis → Reports
    ↓           ↓              ↓             ↓            ↓           ↓         ↓
/data/raw → /data/preprocessed → Training API → /data/checkpoints → Eval API → Analysis → UI
```

### **Checkpoint Strategy:**
- **Best Checkpoints:** `best_{model_name}_{backbone}_{single|multi}_{MMDDYYYY}.pt`
- **Regular Checkpoints:** Configurable save interval
- **Evaluation Selection:** UI untuk choose checkpoint for testing scenarios

---

## **🎯 Success Criteria**

### **Fase 1 (Completed):** ✅
- [x] Model dapat dibangun dengan backbone selection
- [x] Progress tracking berfungsi dengan UI
- [x] Checkpoint management otomatis
- [x] Device optimization aktif
- [x] Configuration management working

### **Fase 2 Goals:**
- [x] Training loop berjalan dengan progress tracking
- [x] Metrics collection real-time (mAP, loss)
- [x] Checkpoint auto-saving dengan format yang benar
- [x] Early stopping berfungsi
- [x] Resume training dari checkpoint

### **Fase 3 Goals:**  
- [x] Evaluation scenarios dapat dijalankan
- [x] Multiple checkpoint selection untuk testing
- [x] Research scenarios metrics collection
- [x] Comparison analysis antar backbone

### **Fase 4 Goals:**
- [ ] Comprehensive analysis reports
- [ ] Multi-layer detection analysis
- [ ] Currency denomination analysis
- [ ] Visualization dan charts generation

---

## **📦 Dependencies & Requirements**

### **Core Dependencies:**
- `torch` >= 1.13.0 (CUDA support)
- `timm` (EfficientNet-B4 backbone)
- `ultralytics` (YOLOv5 reference)
- `yaml` (configuration management)

### **Optional Dependencies:**
- `tensorboard` (training visualization)
- `wandb` (experiment tracking)
- `matplotlib` (plotting)
- `seaborn` (advanced plots)

### **Data Requirements:**
- Preprocessed dataset di `/data/preprocessed/{train,valid,test}`
- Pretrained weights di `/data/pretrained` (fallback: download otomatis)
- Checkpoints directory di `/data/checkpoints`
