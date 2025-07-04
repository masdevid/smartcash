# Summary

The advancement of technology has significantly impacted various aspects of life, including financial management and transactions. One notable technology is the automation system for currency detection, which is crucial in applications such as automated cash machines, assistive devices for the visually impaired, and digital financial management systems. This system enables quick and accurate currency recognition, enhancing efficiency and convenience.

Artificial Intelligence (AI) has revolutionized object detection systems, including currency value detection. AI allows computers to understand complex patterns, such as subtle differences in numbers, colors, or designs on banknotes. It can adapt to various conditions, such as low light, different angles, or worn-out banknotes, ensuring high accuracy and reliability.

The YOLO (You Only Look Once) algorithm is widely recommended for real-time object detection due to its high accuracy and speed. It processes the entire image in a single pass to detect all objects simultaneously. However, YOLOv5 faces challenges in detecting small details on banknotes, such as nominal numbers, watermarks, or small security symbols, especially under poor lighting or when banknotes are folded or torn.

A Hybrid Neural Network Architecture, combining YOLOv5 with EfficientNet-B4, can address these challenges. YOLOv5 quickly detects the location of objects, while EfficientNet-B4 provides detailed classification of objects, such as nominal numbers and security symbols. EfficientNet is designed for high efficiency in image recognition tasks, optimizing performance without excessive computational load.

The research aims to develop and optimize a currency detection system for Indonesian rupiah banknotes using YOLOv5 and EfficientNet-B4. This system is expected to accurately recognize banknotes even in challenging conditions, such as low lighting or damaged banknotes. Future applications include assistive devices for the visually impaired, automated cash machines, ATMs, money changers, and counterfeit detection systems.

# Research Objectives

1. **Implementation of YOLOv5 Algorithm**:
   - To implement the YOLOv5 algorithm for detecting the value of Indonesian rupiah banknotes.

2. **Accuracy of YOLOv5**:
   - To determine the accuracy of the YOLOv5 algorithm in detecting the value of Indonesian rupiah banknotes.

3. **Combination of YOLOv5 and EfficientNet-B4**:
   - To implement a combined approach using YOLOv5 and EfficientNet-B4 for detecting the value of Indonesian rupiah banknotes.

4. **Accuracy of Combined Approach**:
   - To evaluate the accuracy of the combined YOLOv5 and EfficientNet-B4 approach in detecting the value of Indonesian rupiah banknotes.

5. **Optimization and Performance**:
   - To understand the performance improvements achieved by combining YOLOv5 with EfficientNet-B4 in detecting the value of banknotes.