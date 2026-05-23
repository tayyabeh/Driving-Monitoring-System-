# -*- coding: utf-8 -*-
"""
PyQt5 application for driver‑distraction detection.

Features:
- Select a video (max ~3 min) via file dialog.
- Run YOLO inference in a background thread (re‑uses process_video.py logic).
- Show live preview of processed frames and a progress bar.
- When finished, enable playback with Play/Pause, seek slider and time label.
- Sidebar displays simple statistics (total frames, distracted frames, safety score) and a clickable event log.

Dependencies: PyQt5, ultralytics, opencv-python, torch (already installed).
"""

import sys
import os
import threading
import time
from pathlib import Path

# ⚠️  IMPORTANT: torch MUST be imported before PyQt5 on Windows.
# PyQt5 modifies the DLL search path which breaks PyTorch's c10.dll.
import torch                          # noqa: E402 – must come first
import cv2
from ultralytics import YOLO
from PyQt5 import QtCore, QtGui, QtWidgets

# ---------------------------------------------------------------------------
# Helper: reuse the same processing logic as process_video.py but emit signals
# ---------------------------------------------------------------------------
class VideoProcessor(QtCore.QThread):
    progress = QtCore.pyqtSignal(int)            # percent completed
    frame_ready = QtCore.pyqtSignal(QtGui.QImage)  # preview frame
    finished = QtCore.pyqtSignal(str)            # output video path

    def __init__(self, input_path: str, output_path: str, parent=None):
        super().__init__(parent)
        self.input_path = input_path
        self.output_path = output_path
        self._stop_requested = False
        # Load YOLO model once
        self.model = YOLO('best.pt')

    def run(self):
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # limit to 3 min (180 s) if video is longer
        max_frames = int(min(total_frames, fps * 180))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (w, h))
        distracted_frames = 0
        for i in range(max_frames):
            if self._stop_requested:
                break
            ret, frame = cap.read()
            if not ret:
                break
            # Run YOLO inference
            results = self.model(frame, verbose=False)
            # Draw boxes & HUD (same as process_video.py)
            annotated = results[0].plot()
            # Simple distraction metric: count if any detection of unwanted class
            # Classes 2‑9 are distractions (index 1‑8 in our list)
            if any(d in results[0].boxes.cls.tolist() for d in range(1, 9)):
                distracted_frames += 1
            out.write(annotated)
            # Emit preview every 10 frames to keep UI responsive
            if i % 10 == 0:
                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                h_img, w_img, ch = rgb.shape
                bytes_per_line = ch * w_img
                qimg = QtGui.QImage(rgb.data, w_img, h_img, bytes_per_line, QtGui.QImage.Format_RGB888)
                self.frame_ready.emit(qimg)
            percent = int((i + 1) / max_frames * 100)
            self.progress.emit(percent)
        cap.release()
        out.release()
        # Save simple stats as a side‑car file for the UI later
        stats_path = Path(self.output_path).with_suffix('.stats.txt')
        with open(stats_path, "w", encoding="utf-8") as f:
            f.write(f"total_frames={max_frames}\n")
            f.write(f"distracted_frames={distracted_frames}\n")
            safe_score = max(0, 100 - int(distracted_frames / max_frames * 100))
            f.write(f"safety_score={safe_score}\n")
        self.finished.emit(self.output_path)

    def stop(self):
        self._stop_requested = True

# ---------------------------------------------------------------------------
# Main Window UI
# ---------------------------------------------------------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Driver Distraction Analyzer")
        self.resize(1000, 600)
        self._setup_ui()
        self.processor = None
        self.play_timer = None
        self.cap = None
        self.playing = False

    def _setup_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        # Left panel – controls & preview
        left_panel = QtWidgets.QVBoxLayout()
        layout.addLayout(left_panel, 3)

        self.upload_btn = QtWidgets.QPushButton("Upload Video")
        self.upload_btn.clicked.connect(self.select_file)
        left_panel.addWidget(self.upload_btn)

        self.start_btn = QtWidgets.QPushButton("Start Processing")
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_processing)
        left_panel.addWidget(self.start_btn)

        self.progress_bar = QtWidgets.QProgressBar()
        left_panel.addWidget(self.progress_bar)

        self.preview_label = QtWidgets.QLabel()
        self.preview_label.setFixedSize(640, 360)
        self.preview_label.setStyleSheet("background-color:#111; border:1px solid #555;")
        self.preview_label.setAlignment(QtCore.Qt.AlignCenter)
        left_panel.addWidget(self.preview_label)

        # Playback controls (hidden until processing finishes)
        self.play_btn = QtWidgets.QPushButton("Play")
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self.toggle_play)
        left_panel.addWidget(self.play_btn)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.sliderMoved.connect(self.seek)
        left_panel.addWidget(self.slider)
        self.time_label = QtWidgets.QLabel("00:00 / 00:00")
        left_panel.addWidget(self.time_label)

        # Right panel – statistics and event log
        right_panel = QtWidgets.QVBoxLayout()
        layout.addLayout(right_panel, 2)
        self.stats_text = QtWidgets.QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMinimumHeight(200)
        right_panel.addWidget(QtWidgets.QLabel("Statistics"))
        right_panel.addWidget(self.stats_text)
        self.event_list = QtWidgets.QListWidget()
        self.event_list.itemClicked.connect(self.jump_to_event)
        right_panel.addWidget(QtWidgets.QLabel("Event Log (click to jump)"))
        right_panel.addWidget(self.event_list)

    def select_file(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select video", "", "Video Files (*.mp4 *.avi *.mov)")
        if file_path:
            self.input_path = file_path
            self.upload_btn.setText(Path(file_path).name)
            self.start_btn.setEnabled(True)
            self.preview_label.setText("Ready to process: " + Path(file_path).name)

    def start_processing(self):
        output_path = str(Path(self.input_path).with_name("processed_" + Path(self.input_path).name))
        self.processor = VideoProcessor(self.input_path, output_path)
        self.processor.progress.connect(self.progress_bar.setValue)
        self.processor.frame_ready.connect(self.update_preview)
        self.processor.finished.connect(self.processing_finished)
        self.start_btn.setEnabled(False)
        self.upload_btn.setEnabled(False)
        self.processor.start()
        self.preview_label.setText("Processing…")

    def update_preview(self, qimg):
        pix = QtGui.QPixmap.fromImage(qimg).scaled(self.preview_label.size(), QtCore.Qt.KeepAspectRatio)
        self.preview_label.setPixmap(pix)

    def processing_finished(self, output_path):
        self.processed_path = output_path
        self.preview_label.setText("Processing complete!")
        self.play_btn.setEnabled(True)
        self.slider.setEnabled(True)
        # Load stats if present
        stats_file = Path(output_path).with_suffix('.stats.txt')
        if stats_file.exists():
            with open(stats_file, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
            stats = {k: v for k, v in (line.split('=') for line in lines)}
            total = int(stats.get('total_frames', 0))
            distracted = int(stats.get('distracted_frames', 0))
            safe = int(stats.get('safety_score', 0))
            self.stats_text.setPlainText(
                f"Total frames: {total}\n"
                f"Distracted frames: {distracted}\n"
                f"Safety score: {safe}/100"
            )
            # Populate simple event log – each distracted frame as an event
            self.event_list.clear()
            for i in range(distracted):
                ts = i / total * 180  # seconds (max 3 min)
                mins = int(ts // 60)
                secs = int(ts % 60)
                item = QtWidgets.QListWidgetItem(f"Distraction at {mins:02d}:{secs:02d}")
                item.setData(QtCore.Qt.UserRole, ts)
                self.event_list.addItem(item)
        else:
            self.stats_text.setPlainText("No stats file found.")
        # Prepare playback
        self.cap = cv2.VideoCapture(self.processed_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.slider.setMaximum(self.total_frames)
        self.update_time_label(0)
        self.play_timer = QtCore.QTimer()
        self.play_timer.timeout.connect(self.play_frame)

    def toggle_play(self):
        if self.playing:
            self.play_timer.stop()
            self.play_btn.setText("Play")
        else:
            self.play_timer.start(int(1000 / self.fps))
            self.play_btn.setText("Pause")
        self.playing = not self.playing

    def play_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.toggle_play()  # stop at end
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_line = ch * w
        qimg = QtGui.QImage(rgb.data, w, h, bytes_line, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(self.preview_label.size(), QtCore.Qt.KeepAspectRatio)
        self.preview_label.setPixmap(pix)
        pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.slider.setValue(pos)
        self.update_time_label(pos)

    def seek(self, frame_idx):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            # Force a refresh of the displayed frame
            ret, frame = self.cap.read()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                bytes_line = ch * w
                qimg = QtGui.QImage(rgb.data, w, h, bytes_line, QtGui.QImage.Format_RGB888)
                pix = QtGui.QPixmap.fromImage(qimg).scaled(self.preview_label.size(), QtCore.Qt.KeepAspectRatio)
                self.preview_label.setPixmap(pix)
                self.update_time_label(frame_idx)

    def update_time_label(self, frame_idx):
        secs = frame_idx / self.fps
        total_secs = self.total_frames / self.fps
        cur = f"{int(secs // 60):02d}:{int(secs % 60):02d}"
        tot = f"{int(total_secs // 60):02d}:{int(total_secs % 60):02d}"
        self.time_label.setText(f"{cur} / {tot}")

    def jump_to_event(self, item):
        ts = item.data(QtCore.Qt.UserRole)
        frame_idx = int(ts * self.fps)
        self.slider.setValue(frame_idx)
        self.seek(frame_idx)

if __name__ == "__main__":
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
