# 🚗 Driver Cognitive Load & Distraction Tracker
### Client Prototype Demonstration Codebase

Welcome to the **Driver Cognitive Load & Distraction Tracker** prototype repository. This production-ready system leverages state-of-the-art computer vision to monitor driver attention, detect multiple cognitive distractions, and trigger multi-stage safety warnings.

This repository is pre-configured with fully portable paths and optimized models, allowing you to run demonstrations instantly on any Windows machine.

---

## 🚀 Client Quick-Start Guide (Run in 60 Seconds)

To run the interactive desktop application or live webcam tracking on your machine, follow these three simple steps:

### Step 1: Set Up Your Python Environment
Ensure you have Python installed (version **3.10 to 3.13** is recommended). Open your PowerShell or Command Prompt and run:

```powershell
# 1. Clone or download this repository and navigate into the folder
cd Distract

# 2. Create a clean virtual environment
python -m venv .venv

# 3. Activate the virtual environment
.venv\Scripts\activate
```

### Step 2: Install Optimized Dependencies
This project is configured to run smoothly on standard laptops and office computers without requiring expensive GPU hardware. Run the following to install the required packages (pre-configured for Windows CPU environments to avoid massive 2GB+ CUDA downloads):

```powershell
# Upgrade package installer
python -m pip install --upgrade pip

# Install CPU-optimized PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install Core & GUI dependencies
pip install -r requirements.txt
```

### Step 3: Launch the Applications

#### Option A: Run the Interactive Dashboard GUI (Recommended)
This launches a professional dark-themed desktop application. You can load driving video clips, watch real-time background analysis, view detailed analytics (Total Frames, safety scores), and click the event logs to jump to specific distraction timestamps:
```powershell
python gui_app.py
```

#### Option B: Run Real-Time Webcam Detection
This starts a live camera feed using your webcam to track your attention level in real-time, displaying a dynamic HUD and emitting safety alerts:
```powershell
python detect_live.py --source 0
```
*   *Note: Press **`q`** on your keyboard to cleanly close the webcam feed.*

---

## 🌟 Core System Capabilities

1. **Interactive PyQt5 Desktop App (`gui_app.py`)**:
   *   **Video Upload**: Upload driving clips (supports `.mp4`, `.avi`, `.mov` up to 3 minutes).
   *   **Live Progress**: Features a progress bar and background thread execution to keep the UI smooth and responsive.
   *   **Safety Scorecard**: Evaluates the driving session and generates a safety score out of 100 based on distraction frequency.
   *   **Interactive Event Log**: Lists every detected cognitive loading event. Clicking any item instantly seeks the video to that exact violation timestamp.
2. **Real-Time Webcam HUD (`detect_live.py`)**:
   *   **Rolling Attention Gauge**: Computes a continuous driver attention index (0% - 100%) using an Exponential Moving Average (EMA).
   *   **Vocal Alerts & Screen Flashing**: Emits pitch-modulated alerts and displays a crimson flashing warning banner if the driver is distracted for more than 1.5 seconds.
3. **10-Class Cognitive Load Classifier**:
   *   Detects exact driving behaviors mapped into three risk tiers:
       *   🟢 **Safe**: Normal Driving
       *   🟡 **Warning**: Looking Left, Looking Right, Adjusting Dashboard/Radio, Eating, Drinking
       *   🔴 **Critical**: Hands Off Steering, Mobile Texting, Mobile Talking, Drowsiness

---

## 📊 Model Performance & Accuracy

The custom-trained YOLOv8n model has been thoroughly verified on separate validation and test splits:

| Metric | Validation (val) Split | Test (test) Split |
| :--- | :---: | :---: |
| **mAP@50 (Overall Accuracy)** | **82.7%** | **85.9%** |
| **mAP@50-95 (Precision Over Thresholds)** | **69.9%** | **73.4%** |
| **Model Precision** | **87.9%** | **90.7%** |
| **Model Recall** | **81.0%** | **90.7%** |

*All performance plots, including confusion matrices and Precision-Recall curves, are archived for review in the `runs/val/val_run/` and `runs/test/test_run/` directories.*

---

## 🛠️ Help & Troubleshooting Guide

| Issue / Question | Diagnostic | Solution |
| :--- | :--- | :--- |
| **Webcam does not open** | The camera index might be different on your hardware. | Run `python detect_live.py --source 1` (try index 1, 2, etc. if you have external USB cameras). |
| **No audio alerts playing** | Auditory beeps are supported on Windows OS via the native system sound API. | Ensure your Windows system volume is turned up. On macOS/Linux, the warning banner flashes on-screen as a visual fallback. |
| **PyTorch DLL Load Failures** | Windows-specific environment conflicts with PyQt5 DLL paths. | **Resolved in code**: All scripts have been built with special import logic (`import torch` executed first) to prevent this issue. |
| **How to run on a recorded video?** | You want to process a raw driving file without launching the GUI. | Run: `python process_video.py --input your_video.mp4 --output output.mp4 --conf 0.25` |

---

## 📂 Deliverable Directory Contents

*   `gui_app.py` - Core interactive desktop analytics app.
*   `detect_live.py` - Live webcam attention tracker.
*   `process_video.py` - Offline video batch processor.
*   `validate_test.py` - Automated accuracy validation script.
*   `best.pt` - Trained YOLOv8n network weights (~6.2 MB).
*   `data.yaml` - Fully portable dataset configurations (relative paths).
*   `requirements.txt` - Python environment requirements.
*   `training_colab.ipynb` / `train.py` - Scripts for model retraining and local test training.
