#  Prototipo Cautivo: Edge AI Geological Scanning System
> **NASA Space Apps Challenge 2026** | Category: *Identify Earth Locations that Analog the Permanent Moon Base Locations and Mars*  
> Developed by **UTD Robotics** (Universidad Tecnológica de Durango, México)

---

## 📌 Project Overview

**Prototipo Cautivo** is an autonomous, stationary geological scanning cube designed for real-time identification, measurement, and classification of planetary mineral analogs. Built to operate in bandwidth-constrained space exploration environments, the system performs 100% **offline edge computing** to analyze hand samples in terrestrial analog sites (such as volcanic and arid fields in Durango, Mexico) prior to lunar or Martian deployment.

The platform integrates deep-learning vision inference on an **NVIDIA Jetson Orin Nano Super (8GB)** with multi-sensor spatial telemetry powered by an **ESP32-S3** microcontroller.

---

## 🏗️ Hardware Architecture & Sensor Fusion

| Component | Technical Specifications | Function |
| --- | --- | --- |
| **Main Compute Node** | NVIDIA Jetson Orin Nano Super (8GB RAM, CUDA 13.0, PyTorch 2.14) | Real-time YOLOv8 object detection & telemetry packaging |
| **Sensor Controller** | ESP32-S3 (N16R8 Dual-Core, PlatformIO / C++) | Distance measurement & spatial orientation processing |
| **Vision Input** | Logitech 1080p Webcam (`/dev/video0`, MJPEG @ 60 FPS) | High-speed image acquisition inside the scanning chamber |
| **Distance Sensors** | 4x VL53L0X ToF (Time-of-Flight) via TCA9548A I2C Multiplexer | 3D physical dimensioning & sample clearance checking |
| **IMU Orientation** | MPU6050 6-DOF Gyroscope / Accelerometer | Scanning bed tilt angle & stability monitoring |
| **Power Distribution** | 4S2P 18650 Pack + BMS 4S 40A (XL4016 & XL4015 Buck Regulators) | Autonomous embedded power management |

---

## 🔬 Multi-Class Geological Dataset

The vision model is trained to classify 6 target mineral classes essential for **In-Situ Resource Utilization (ISRU)** and planetary surface characterization:

1. `lunar_basalt` — Dark, fine-grained mare basalt analogs.
2. `anorthosite` — Calcium-rich plagioclase highlands analogs.
3. `lunar_breccia` — Impact-fragmented composite rocks.
4. `obsidian` — Highly reflective volcanic glass.
5. `pumice_scoria` — Highly porous vesicular volcanic ejecta.
6. `hematite` — Iron oxide mineral indicator for water/oxidation processes.

---

## ⚡ Performance & Edge AI Pipeline

- **Inference Engine:** YOLOv8 Nano (`yolov8n.pt`) fine-tuned for low latency.
- **Frame Rate:** Stable 60 FPS processing directly on Jetson Orin Nano Tensor Cores.
- **Bandwidth Reduction (Deep Space Protocol):** Instead of streaming heavy raw video (1080p), the system extracts metadata (Class ID, Confidence, Bounding Box, Physical Volume, Tilt) into lightweight JSON payloads. Crop images of anomalies ("weird rocks") are encoded in Base64 for selective ground transmission.

---

## 📂 Repository Structure

```text
prototipo-cautivo-vision/
├── README.md                   # Project documentation
├── requirements.txt             # Python environment dependencies
├── data.yaml                   # YOLOv8 multi-class dataset configuration
├── vision_webcam.py            # Main real-time GPU inference script (60 FPS)
├── reentrenar_rocas.py          # YOLOv8 fine-tuning & training script (50 Epochs)
├── verificar_dataset.py         # Dataset class balance & integrity validator
└── docs/
    └── hardware_topology.png   # Architecture block diagram
