# AegisGuard: Real-Time Multi-Modal Violence Escalation Predictor & Early Warning System

## 1. Overview
AegisGuard is an advanced, lightweight multi-modal surveillance AI framework designed for edge-cloud computing environments. Moving beyond reactive violence detection (flagging incidents during or after occurrence), this architecture **forecasts violent escalation** seconds before physical conflict manifests, incorporating a dynamic **Predictive Lead-Time Metric ($\Delta t$)**.

---

## 2. Multi-Modal Architectural Modules

```
 ┌────────────────────────┐    ┌────────────────────────┐    ┌────────────────────────┐
 │   Module 1: Video      │    │   Module 2: Audio      │    │   Module 3: Weapon     │
 │ 16-frame 2D-CNN        │    │ Mel-Spectrogram (64)   │    │ YOLO + Confusion       │
 │ (MobileNetV3)          │    │ + CNN-BiLSTM           │    │ Object Rejection       │
 └──────────┬─────────────┘    └───────────┬────────────┘    └───────────┬────────────┘
            │ Spatial: f_t                 │ Acoustic: h_a               │ Threat: w_threat
            ▼                              ▼                             ▼
 ┌────────────────────────┐    ┌────────────────────────┐    ┌────────────────────────┐
 │ Temporal Attention TAM │    │ Acoustic Anomaly /     │    │ Spatial Density &      │
 │ Attention Vector: v_t  │    │ Fallback Embed: a_emb  │    │ Confidence Vector      │
 └──────────┬─────────────┘    └───────────┬────────────┘    └───────────┬────────────┘
            │                              │                             │
            └──────────────────────┬───────┴─────────────────────────────┘
                                   │
                                   ▼
              ┌──────────────────────────────────────────────┐
              │  Module 4: Gated Cross-Attention Fusion      │
              │  F_fusion = CrossAttention(v_t, a_emb, w_th) │
              └────────────────────┬─────────────────────────┘
                                   │
                                   ▼
              ┌──────────────────────────────────────────────┐
              │  Early Warning & Lead-Time Prediction Engine  │
              │  P(V_{t+Δt}) = σ(W · F_fusion + b)           │
              │  Escalation Velocity: S(t) = dP/dt           │
              │  Lead Time: Δt_lead = (τ_crit - P) / S(t)    │
              │  Composite Threat: H(t)                      │
              └──────────────────────────────────────────────┘
```

### Module 1: Lightweight Spatio-Temporal Video Predictor
- Avoids computationally heavy 3D-CNNs for real-time edge execution.
- **Spatial Feature Extraction**: Incoming frames $\mathbf{I}_t \in \mathbb{R}^{3 \times 112 \times 112}$ are processed via a lightweight 2D-CNN (MobileNetV3-Small / ResNet18):
  $$\mathbf{f}_t = \text{Backbone}(\mathbf{I}_t) \in \mathbb{R}^{512} \quad (t = 1, \dots, 16)$$
- **Temporal Attention Module (TAM)**:
  $$\mathbf{h}_t = \text{BiGRU}(\mathbf{f}_t, \mathbf{h}_{t-1})$$
  $$\alpha_t = \frac{\exp(\mathbf{w}_a^\top \tanh(\mathbf{W}_h \mathbf{h}_t + \mathbf{b}_h))}{\sum_{k=1}^{16} \exp(\mathbf{w}_a^\top \tanh(\mathbf{W}_h \mathbf{h}_k + \mathbf{b}_h))}$$
  $$\mathbf{v}_{temporal} = \sum_{t=1}^{16} \alpha_t \mathbf{h}_t \in \mathbb{R}^{512}$$

### Module 2: Audio Stream Anomaly Encoder
- **Log-Mel Spectrogram**: $N_{mels} = 64, T_{audio} = 130, SR = 22,050 \text{ Hz}$.
- **CNN-BiLSTM Sequence Modeling**:
  $$\mathbf{a}_{emb} = \text{BiLSTM}(\text{Conv2D}(\mathbf{S}_{mel})) \in \mathbb{R}^{128}$$
- **Learned Missing-Modality Fallback**: Seamlessly handles silent feeds or cameras without microphones:
  $$\tilde{\mathbf{a}} = m_{audio} \cdot \mathbf{a}_{emb} + (1 - m_{audio}) \cdot \mathbf{e}_{fallback}$$

### Module 3: Parallel Weapon & Threat Vector Engine
- Extended Taxonomy: Firearms (Rifles, Pistols), Edged Weapons (Knives), Blunt Objects.
- **Confusion Object Rejection**: Explicitly filters out non-threats (smartphones, wallets, water bottles) to eliminate false alarms.
- **Structured Threat Vector $\mathbf{w}_{threat} \in \mathbb{R}^6$**:
  $$\mathbf{w}_{threat} = [c_{firearm}, c_{knife}, c_{blunt}, c_{max}, D_{spatial}, N_{threat}]$$
  $$\mathbf{z}_w = \text{MLP}(\mathbf{w}_{threat}) \in \mathbb{R}^{64}$$

### Module 4: Gated Cross-Modal Fusion
- Dynamic Modality Gating (GMU) + Cross-Attention:
  $$\mathbf{z}_v = \text{LN}(\mathbf{W}_v \mathbf{v}_{temporal}), \quad \mathbf{z}_a = \text{LN}(\mathbf{W}_a \tilde{\mathbf{a}}), \quad \mathbf{z}_w = \text{LN}(\mathbf{W}_w \mathbf{z}_w)$$
  $$\mathbf{g} = \text{Softmax}(\mathbf{W}_g [\mathbf{z}_v, \mathbf{z}_a, \mathbf{z}_w])$$
  $$\mathbf{F}_{fusion} = \text{LN}(\mathbf{g}_1 \mathbf{z}_v + \mathbf{g}_2 \mathbf{z}_a + \mathbf{g}_3 \mathbf{z}_w + \text{CrossAttention}(\mathbf{z}_v, [\mathbf{z}_a; \mathbf{z}_w]))$$
- **Predictive Early Warning Probability Formulation**:
  $$P(V_{t+\Delta t}) = \sigma(\mathbf{W}_{pred} \cdot \mathbf{F}_{fusion} + b_{pred})$$

### Module 5: Dynamic Real-Time Trajectory & Lead-Time Analyzer
- **Escalation Velocity (Slope)**:
  $$\mathcal{S}(t) = \frac{P(t) - P(t-k)}{k \cdot \Delta t_{frame}}$$
- **Predictive Lead-Time Metric ($\Delta t_{lead}$ in seconds)**:
  $$\Delta t_{lead} = \begin{cases} \frac{\tau_{crit} - P(t)}{\max(\mathcal{S}(t), \epsilon)}, & \text{if } \mathcal{S}(t) > 0 \text{ and } P(t) < \tau_{crit} \\ 0.0, & \text{if } P(t) \ge \tau_{crit} \\ \infty, & \text{otherwise} \end{cases}$$
- **Multi-Modal Hazard Index**:
  $$\mathcal{H}(t) = 0.45 \cdot P(V_{t+\Delta t}) + 0.25 \cdot \text{ReLU}(\mathcal{S}(t)) + 0.20 \cdot c_{max}^{weapon} + 0.10 \cdot \text{AudioEnergy}(t)$$

---

## 3. Directory Structure

```
violence-app/
├── app.py                     # FastAPI server with predictive endpoints
├── requirements.txt           # Dependency specifications
├── Dockerfile                 # Production Docker image with ffmpeg
├── weights/                   # Neural network weights
│   ├── fusion_best.pt         # Multi-modal fusion checkpoint
│   └── weapon_best.pt         # Fine-tuned YOLOv8 weapon model
├── src/
│   ├── models/                # Core Multi-Modal Submodules
│   │   ├── spatial_temporal.py# Module 1: MobileNetV3 + TAM
│   │   ├── audio_stream.py    # Module 2: Mel + CNN-BiLSTM Audio
│   │   ├── weapon_stream.py   # Module 3: YOLO + Confusion Mitigation
│   │   ├── multimodal_fusion.py # Module 4: Gated Cross-Modal Network
│   │   └── early_warning.py   # Module 5: Lead-Time & Trajectory Engine
│   ├── pipeline.py            # End-to-end unified inference pipeline
│   ├── model.py               # Video model architecture
│   └── model_fusion.py        # Fusion model loader
├── static/
│   └── index.html             # Real-time Glassmorphism Surveillance Dashboard
└── test_multimodal.py         # Automated test suite (5/5 modules)
```

---

## 4. Running the Application

### Local Setup & Execution
```bash
# Install dependencies
pip install -r requirements.txt

# Start FastAPI Server with auto-reload
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000` in your web browser.

### API Endpoints
- `POST /predict_video`: Upload video file -> returns full multi-modal diagnostics, trajectory array, lead-time $\Delta t$, weapon detections, and download URL for annotated telemetry video.
- `POST /predict_clip`: Accepts 16 rolling webcam frames (JSON base64) + audio chunk -> returns real-time early warning metrics.
- `POST /predict_frame`: Fast single-frame weapon detection with confusion object rejection.
- `GET /health`: Health and hardware capability status.
