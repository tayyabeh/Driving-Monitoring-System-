# 🚗 Driver Cognitive Load & Distraction Tracker

A state-of-the-art computer vision pipeline built using **YOLOv8**, **OpenCV**, and **PyQt5** to monitor driver attention, detect cognitive distractions, and trigger real-time multi-stage warnings. This repository is packaged as a ready-to-run prototype for client demonstration.

---

## 🌟 Key Features

1. **Dual Run Modes**:
   - **Real-Time Webcam HUD**: Low-latency live video monitoring via webcam or connected dashboard camera.
   - **Offline Video Processor**: High-throughput batch inference of recorded driving sessions, exporting high-fidelity HUD-annotated MP4 video.
2. **PyQt5 Desktop Application**:
   - Clean, dark-themed dashboard GUI.
   - Background thread processing with live preview updates and progress tracking.
   - Post-analysis playback with interactive Seek and Play/Pause features.
   - Sidebar displaying driving session diagnostics: **Total Frames**, **Distraction Duration**, and overall **Safety Score**.
   - Clickable **Event Log** that jumps the video player directly to the timestamp of detected violations.
3. **Advanced Attention Engine**:
   - Dynamically tracks driver focus using a **Rolling Driver Attention Index (0% - 100%)** backed by an Exponential Moving Average (EMA).
   - Distractions cause rapid decay based on class severity (e.g., drowsiness decays faster than adjusting the radio), while returning eyes to the road restores the score.
4. **Adaptive Warning & Notification System**:
   - Multi-tier alert hierarchy: **Normal (Safe)**, **Distracted (Warning)**, and **Critical Warning (Danger)**.
   - Pitch-modulated auditory feedback (system beeps) triggers if the driver is distracted for more than **1.5 seconds** or the attention score falls below **50%**.
   - Flashing crimson HUD overlay alerts the driver: `WARNING: BRING HANDS TO WHEEL & EYES ON ROAD!`.
5. **10-Class Cognitive Load Model**:
   - Fine-tuned on a comprehensive driving dataset to track exact cognitive load profiles:
     * **Safe (Green)**: `Normal_Driving`
     * **Warning (Orange)**: `Looking_Left`, `Looking_right`, `Adjusting_Radio_Dashboard`, `Eating`, `Drinking`
     * **Danger (Red)**: `Hands_off_steering`, `Mobile_Talking`, `Mobile_Texting`, `Drowsiness`

---

## 📊 Model Performance & Metrics

The custom YOLOv8n model was trained for **50 epochs** on a Google Colab GPU environment. Detailed validation and test splits metrics are summarized below:

### Core Evaluation Metrics
| Split | mAP@50 | mAP@50-95 | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Validation (val)** | **0.8270** | **0.6990** | 0.8790 | 0.8100 |
| **Test (test)** | **0.8590** | **0.7340** | 0.9070 | 0.9070 |

*Plots and evaluation charts (confusion matrices, PR curves, and F1 curves) are saved inside the `runs/val/val_run/` and `runs/test/test_run/` directories.*

### Technical Observations
- **Strong Performers**: The model exhibits exceptional accuracy in detecting highly defined postures: `Drowsiness` (AP50: **0.95+**), `Drinking`, `Eating`, and `Normal_Driving`.
- **Edge Classes**: Subtle facial shifts like `Looking_Left` (AP50: **0.495**) and `Looking_right` (AP50: **0.597**) have lower precision. This is typical when dashcam angles are near-parallel, but is compensated for in real-world deployment by our rolling temporal attention index which filters out instantaneous false-positives.

---

## 📂 Project Directory Structure

```
d:/projects/Distract/
├── best.pt                       # Fine-tuned YOLOv8n weights (~6.2 MB)
├── data.yaml                      # Dataset configuration file (relative paths)
├── detect_live.py                 # Real-time webcam monitoring script + Audio Alerts + HUD
├── process_video.py               # Batch video processing pipeline script
├── gui_app.py                     # PyQt5 Desktop Application (dashboard + video player)
├── validate_test.py               # Comprehensive validation & test splits evaluator
├── train.py                       # Local CPU-safe training framework
├── training_colab.ipynb           # Google Colab GPU training notebook
├── requirements.txt               # Unified pip dependencies manifest
├── .gitignore                     # Git ignore rules for datasets, cached files, and large videos
└── README.md                      # Professional technical documentation
```

---

## ⚙️ Installation & Setup Guide

### 1. Clone the Repository
Clone this repository to your local development machine:
```bash
git clone <your-private-repo-url>
cd Distract
```

### 2. Set Up Virtual Environment & Packages
We highly recommend using Python **3.10 to 3.13**. 

To prevent Windows-specific PyTorch DLL issues and avoid downloading massive CUDA binaries if you intend to run on CPU-only machines, follow these installation commands:

```powershell
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch (CPU-Optimized Build to prevent huge downloads)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Run the Applications

### A. Live Webcam Detection (`detect_live.py`)
Runs real-time inference on your camera stream. Since standard webcams have different angles compared to driving dataset dashcams, we use a calibrated low-confidence detection threshold:
```bash
python detect_live.py --source 0 --conf 0.05 --imgsz 640
```
- **Controls**: Press **`q`** to cleanly close the live monitoring window.

### B. Interactive Desktop GUI (`gui_app.py`)
Launches the full dashboard application. Perfect for client presentations and walk-throughs:
```bash
python gui_app.py
```
- **Usage**:
  1. Click **Upload Video** to select a driving clip (e.g., up to 3 minutes).
  2. Click **Start Processing**. Watch the pipeline analyze the video with a live progress preview.
  3. Once finished, click **Play** or use the **slider** to navigate the processed video.
  4. Click on any distraction event in the **Event Log** to jump directly to that violation!

### C. Batch Video Processing (`process_video.py`)
Process a pre-recorded driving video and save it with the custom HUD overlay:
```bash
python process_video.py --input test1.mp4 --output output_test1.mp4 --conf 0.25 --limit-frames 300
```
- Use `--limit-frames 0` to process the entire video file.

### D. Model Metrics Evaluator (`validate_test.py`)
Run this script to reproduce the metrics table and generate plots (confusion matrix, PR curve) on your splits:
```bash
python validate_test.py
```

### E. Model Retraining (`train.py` & Colab)
- **Local Test**: Run `python train.py --epochs 3` for a rapid execution test.
- **GPU Retraining**: Upload `training_colab.ipynb` to Google Colab, mount your drive, connect the dataset path, and run the notebook to train for 50 epochs.

---

## 🔒 Professional Client Delivery (GitHub Push Guide)

If you are preparing to deliver this codebase to a client privately on GitHub, execute the following commands:

```powershell
# 1. Initialize git repository
git init

# 2. Add files (large videos and datasets are automatically filtered by .gitignore)
git add .

# 3. Create initial commit
git commit -m "Initial commit: Ready-to-run Driver Cognitive Load & Distraction System"

# 4. Create a private repository on GitHub, then link it
git remote add origin https://github.com/YOUR_USERNAME/YOUR_PRIVATE_REPO.git
git branch -M main

# 5. Push code
git push -u origin main
```
