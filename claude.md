# Project Documentation: Spatio-Temporal Violence Predictor

## 1. Executive Summary
This project proposes a lightweight, multi-modal architecture designed for edge-cloud computing environments. The primary objective is to move beyond reactive surveillance by predicting violent incidents before they escalate, utilizing a unique fusion of spatial, temporal, visual, and audio data streams.

## 2. High-Level Architectural Flow
The system processes parallel streams of video and audio through dedicated subsystems before fusing them for a final predictive output:
1. **Visual Ingestion:** Video frames are analyzed for spatial dynamics (movements) and specific threat objects (weapons).
2. **Acoustic Ingestion:** Synchronized audio is processed to detect aggressive acoustic anomalies.
3. **Multi-Modal Fusion:** Features from visual behavior, weapon detection, and audio cues are concatenated.
4. **Early Warning Prediction:** The fused data is evaluated to generate a probability score for future violence, incorporating a predictive lead-time metric.

---

## 3. Detailed Module Breakdown

### Module 1: The Spatio-Temporal Violence Predictor
Designed to be lightweight for edge deployment, this module avoids computationally heavy 3D CNNs.
* **Spatial Feature Extraction:** Incoming frames are passed through a lightweight 2D CNN (e.g., MobileNetV3) to extract base spatial features.
* **Temporal Attention:** A Temporal Attention Module (TAM) replaces standard pooling, weighting the importance of specific frames over a 16-frame rolling window.
* **Behavioral Embedding:** The output is a feature vector representing the physical dynamics of the scene (e.g., rapid, erratic movements indicative of a struggle).

### Module 2: Audio Stream Analysis
Audio serves as a critical secondary validation metric to reduce false positives inherent in video-only uncontrolled environments.
* **Preprocessing:** Raw audio streams are converted into Mel-spectrograms.
* **Sequence Modeling:** Spectrograms are fed into a Long Short-Term Memory (LSTM) network.
* **Audio Embedding:** The LSTM outputs a feature vector representing aggressive acoustic anomalies, such as raised voices, screams, or sudden loud noises.

### Module 3: Parallel Multi-Class Weapon Detector
Operating in parallel with Module 1, this component broadens threat detection beyond standard firearm-centric models.
* **Detection Engine:** Utilizes a lightweight, real-time object detector like YOLOv9-tiny.
* **Class Definitions:** Trained on custom datasets to identify a spectrum of threats: firearms, knives, and blunt objects.
* **Confusion Classes:** Training data explicitly includes a "confusion object" class (e.g., smartphones, wallets, water bottles) to teach the model to reject non-threats, significantly minimizing false alarms.

### Module 4: Multi-Modal Fusion and Early Warning Engine
The core prediction engine that transitions the system from detection to forecasting.
* **Attention-Based Fusion:** Embeddings from the Video Stream (Module 1), Audio Stream (Module 2), and confidence scores/bounding boxes from the Weapon Detector (Module 3) are concatenated.
* **Prediction Mechanism:** The fused vector ($F_{fusion}$) is passed through a final predictive network layer to output a future probability using the following function:
$$P(V_{t+\Delta t}) = \sigma(W \cdot F_{fusion} + b)$$
* **Lead-Time Metric:** $P(V)$ represents the probability of violence occurring, while $\Delta t$ represents the lead time (e.g., predicting an incident 3 to 5 seconds *before* a physical strike occurs).

---

## 4. Recommended Implementation Stack
* **Deep Learning Framework:** PyTorch (optimal for custom attention mechanisms and complex multi-modal fusion).
* **Weapon Detection Ecosystem:** Ultralytics (YOLOv9/YOLOv10).
* **Edge Optimization & Deployment:** TensorRT or ONNX Runtime. Required to quantize models (FP16 or INT8 precision) ensuring smooth execution on edge devices like the NVIDIA Jetson Nano or Orin Nano.

---

## 5. Core Novelty and Contributions
This framework directly addresses critical gaps in current video surveillance literature through four key innovations:
1. **Predictive Lead-Time Metric ($\Delta t$):** Shifts focus from reactive detection (during/after an incident) to forecasting violence seconds before physical conflict.
2. **Audio-Visual Cross-Attention Fusion:** Fuses spatial-temporal visual features with Mel-spectrogram audio representations to provide a multi-modal secondary validation layer, filtering out visual-only false alarms.
3. **Extended Weapon Taxonomy & Confusion Class Mitigation:** Expands detection to knives and blunt objects while explicitly teaching the network to reject false positive threats via "confusion object" training.
4. **Lightweight Edge-Optimized Spatio-Temporal Pipeline:** Replaces expensive 3D-CNNs with a combination of MobileNetV3 and a custom Temporal Attention Module (TAM) for real-time inference on edge hardware.