Berikut adalah ringkasan tahapan pengenalan objek menggunakan EfficientNet-B4 sebagai backbone dalam arsitektur YOLOv5, beserta diagram Mermaidnya:

### Ringkasan Tahapan

1. **Tahap Input Gambar**:
   - Gambar mentah (uang kertas rupiah) dimasukkan ke dalam sistem deteksi.
   - Gambar mengalami preprocessing (resize, normalisasi, augmentasi) untuk memastikan ukurannya sesuai dengan input yang diharapkan oleh YOLOv5 dan EfficientNet-B4.

2. **Backbone EfficientNet-B4**:
   - EfficientNet-B4 menggantikan backbone YOLOv5 (CSPDarknet) untuk mengekstraksi fitur dari gambar input.
   - EfficientNet-B4 mengekstrak fitur melalui serangkaian lapisan konvolusi (MBConv) yang lebih efisien.
   - Fitur yang diekstrak mencakup pola-pola penting pada uang kertas rupiah, seperti angka nominal, tekstur uang, gambar, dan simbol watermark atau keamanan.
   - Compound Scaling membuat fitur yang diekstrak lebih detail dan multi-level, membantu mendeteksi objek kecil dan variasi pencahayaan dengan lebih baik.
   - Hasil ekstraksi fitur adalah feature map yang dikirimkan ke Neck YOLOv5.

3. **Neck YOLOv5**:
   - Neck (FPN atau PANet) menggabungkan fitur multi-level yang dihasilkan EfficientNet-B4.
   - Ini memungkinkan model menangani objek dengan berbagai ukuran dan resolusi.

4. **Prediksi di Head YOLOv5**:
   - Head YOLOv5 memproses fitur dari Neck untuk memprediksi:
     - Bounding Box (posisi dan ukuran objek).
     - Confidence Score (tingkat keyakinan deteksi).
     - Kelas Objek (misalnya nominal uang: "50.000" atau "100.000").

### Diagram Mermaid

```mermaid
graph TD
    A[Input Gambar] --> B[Preprocessing]
    B --> C["EfficientNet-B4 (Backbone)"]
    C --> D["Feature Map"]
    D --> E["Neck YOLOv5"]
    E --> F["Head YOLOv5"]
    F --> G["Output (Bounding Box + Confidence Score + Class Label)"]
```

Diagram ini menggambarkan alur kerja dari input gambar hingga output prediksi objek menggunakan arsitektur YOLOv5 dengan EfficientNet-B4 sebagai backbone.
Berikut diagram lebih detilnya:

```mermaid
flowchart TD
    subgraph Input["Input Processing"]
        A[RGB Image] --> B[Preprocessing]
        B --> |"Resize 640x640"| C[Normalized Image]
        C --> |"Image Augmentation"| D[Enhanced Image]
    end

    subgraph Backbone["EfficientNet-B4 Backbone"]
        E["timm.create_model
        efficientnet_b4"] --> F["Feature Extraction
        (features_only=True)"]
        F --> G["Multi-scale Features
        out_indices=(2,3,4)"]
        G --> H["Feature Channels
        P3, P4, P5"]
    end

    subgraph Neck["Feature Processing"]
        I["Feature Pyramid
        Network"] --> J["Lateral
        Connections"]
        J --> K["Top-down
        Pathway"]
        K --> L["Path Aggregation
        Network"]
    end

    subgraph Head["Detection Head"]
        M["Multi-scale
        Detection"] --> N["Bounding Box
        Regression"]
        M --> O["Confidence
        Score"]
        M --> P["Class
        Prediction"]
    end

    Input --> Backbone
    Backbone --> Neck
    Neck --> Head
```