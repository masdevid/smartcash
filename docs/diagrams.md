# SmartCash Denomination Detector 💵🔍

The SmartCash Denomination Detector workflow utilizes a combination of EfficientNet-B4 for feature extraction and YOLOv5 for object detection. This hybrid approach aims to improve the accuracy and efficiency of detecting currency denominations by leveraging advanced neural network architectures.

```mermaid
flowchart TD
    A["Input Layer 🖼️"] --> |RGB Image| B["Preprocessing"]
    B --> |Resize 640x640| C["Normalization"]
    
    subgraph "EfficientNet-B4 Backbone 🧠"
    C --> D["Feature Extraction"]
    D --> |Compound Scaling| E["Multi-Level Feature Maps"]
    end
    
    subgraph "Feature Processing 🔍"
    E --> F["Feature Pyramid Network (FPN)"]
    F --> G["Path Aggregation Network (PAN)"]
    end
    
    subgraph "YOLOv5 Head 🎯"
    G --> H["Bounding Box Prediction"]
    H --> I["Class Label Detection"]
    I --> J["Confidence Score Calculation"]
    end
    
    J --> K["Output Layer 📊"]
    K --> |Detection Results| L["Bounding Box Location"]
    K --> |Denomination| M["Currency Value"]
    K --> |Precision| N["Detection Confidence"]
    
    style A fill:#f9d,stroke:#333,stroke-width:2px
    style K fill:#bbf,stroke:#333,stroke-width:2px
```
## Reference Workflow 📚

### YOLOv5 Workflow 🚀
```mermaid
flowchart TD
    A[Input Image] --> B[Preprocessing]
    B --> |Resize to 640x640| C[Normalization]
    C --> |Scale Pixel Values 0-1| D[Backbone: EfficientNet-B4]
    
    D --> |Extract Features| E[Neck: Feature Pyramid Network]
    E --> |Generate Multi-Scale Features| F[Detection Heads]
    
    F --> G[Predict Bounding Boxes]
    G --> |Evaluate Each Grid Cell| H[Prediction]
    H --> |1. Coordinates x,y,w,h| I[2. Confidence Score]
    H --> |3. Object Class| I
    
    I --> J[Apply Non-Maximum Suppression]
    J --> K[Produce Final Detection Output]
    
    K --> L{Detection Scenarios}
    L -->|Normal Conditions| M[Accurate Detection]
    L -->|Low Light Conditions| N[Challenging Detection]
    L -->|Overlapping Objects| O[Complex Localization]
    
    subgraph Challenges
    M --> P[Achieve High Precision]
    N --> Q[Utilize Adaptive Feature Extraction]
    O --> R[Implement Smart Filtering]
    end
    
    K --> |Output Bounding Boxes| S[Bounding Boxes]
    K --> |Output Confidence Scores| T[Class Labels]
    
    style A fill:#f9d,stroke:#333,stroke-width:2px
    style K fill:#bbf,stroke:#333,stroke-width:2px
    style L fill:#ffd,stroke:#333,stroke-width:2px
    style Challenges fill:#dfd,stroke:#333,stroke-width:1px
```

### EfficientNet-B4 Workflow 🧠
```mermaid
flowchart TD
    A[Input Image] --> B[Preprocessing]
    B --> |Resize 380x380px| C[Normalization 0-1]
    
    C --> D[Input Layer]
    D --> E[Stem Layer]
    E --> F[Feature Extraction: Backbone]
    
    subgraph Backbone Features
    F --> |MBConv Blocks| G[Depthwise Separable Convolution]
    G --> H[Squeeze-and-Excitation Blocks]
    H --> I[Swish Activation]
    I --> J[Batch Normalization]
    end
    
    J --> K[Feature Aggregation: Neck]
    K --> L[Classification Head]
    
    L --> M[Global Average Pooling]
    M --> N[Fully Connected Layer]
    N --> O[Softmax Activation]
    
    O --> P{Class Prediction}
    P --> |Highest Probability| Q[Final Class Label]
    
    subgraph Postprocessing
    Q --> R[Thresholding]
    R --> S[Label Mapping]
    end
    
    style A fill:#f9d,stroke:#333,stroke-width:2px
    style P fill:#bbf,stroke:#333,stroke-width:2px
    style Backbone\ Features fill:#dfd,stroke:#333,stroke-width:1px
    style Postprocessing fill:#ffd,stroke:#333,stroke-width:1px
```

### EfficientNet-B4 YOLOv5 Approaches 🔄
```mermaid
flowchart TD
    subgraph "Approach A: EfficientNet-B4 as Additional Classification"
    A1["Input Image"] --> A2["YOLOv5 Detection"]
    A2 --> A3["Crop Bounding Box"]
    A3 --> A4["Normalize RoI 380x380"]
    A4 --> A5["EfficientNet-B4 Classification"]
    A5 --> A6["Detailed Class Prediction"]
    end

    subgraph "Approach B: EfficientNet-B4 as Backbone"
    B1["Input Image"] --> B2["Preprocess"]
    B2 --> B3["EfficientNet-B4 Feature Extraction"]
    B3 --> B4["YOLOv5 Neck (FPN/PANet)"]
    B4 --> B5["YOLOv5 Head"]
    B5 --> B6["Object Detection Output"]
    end

    style A1 fill:#f9d,stroke:#333,stroke-width:2px
    style B1 fill:#f9d,stroke:#333,stroke-width:2px
    style A6 fill:#000,stroke:#fff,stroke-width:2px
    style B6 fill:#000,stroke:#fff,stroke-width:2px
```

### Testing Scenarios 🧪
```mermaid
flowchart TD
    A[SmartCash Detector Testing Framework] --> B[Baseline Model: YOLOv5 CSPDarknet]
    A --> C[Proposed Model: YOLOv5 + EfficientNet-B4]
    
    B --> D[Scenario 1: Position Variations]
    B --> E[Scenario 2: Lighting Conditions]
    
    C --> F[Scenario 3: Position Variations]
    C --> G[Scenario 4: Lighting Conditions]
    
    D --> H[Experimental Setup]
    E --> H
    F --> H
    G --> H
    
    H --> I[Training Set: 70%]
    H --> J[Validation Set: 15%]
    H --> K[Testing Set: 15%]
    
    D --> L[Evaluation Metrics]
    E --> L
    F --> L
    G --> L
    
    L --> M[Precision]
    L --> N[Recall]
    L --> O[F1-Score]
    L --> P[mAP]
    
    classDef darkBg fill:#f9d,stroke:#333,stroke-width:2px,color:#ffffff;
    classDef lightBg fill:#66b3ff,stroke:#333,stroke-width:2px,color:#000000;
    classDef mediumBg fill:#99ff99,stroke:#333,stroke-width:2px,color:#000000;
    classDef neutralBg fill:#ffcc99,stroke:#333,stroke-width:2px,color:#000000;
    classDef metricBg fill:#ff9999,stroke:#333,stroke-width:2px,color:#000000;
    
    class A darkBg;
    class B,F lightBg;
    class C mediumBg;
    class H neutralBg;
    class L metricBg;
    class I,J,K neutralBg;
    class M,N,O,P metricBg;
```