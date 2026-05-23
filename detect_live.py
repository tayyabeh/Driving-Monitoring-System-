import torch          # must be imported first on Windows to avoid DLL conflict
import cv2
import time
import os
import argparse
from ultralytics import YOLO

# Try importing winsound for Windows audio alerts, fallback to console alerts if not available
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

# Class names mapping from data.yaml
CLASS_NAMES = [
    'Adjusting_Radio_Dashboard', 
    'Drinking', 
    'Drowsiness', 
    'Eating', 
    'Hands off steering', 
    'Looking_Left', 
    'Looking_right', 
    'Mobile Talking', 
    'Mobile Texting', 
    'Normal_Driving'
]

# Risk levels: 0 = Safe (Green), 1 = Warning/Distracted (Orange), 2 = High Risk/Danger (Red)
CLASS_RISK = {
    'Normal_Driving': 0,
    'Looking_Left': 1,
    'Looking_right': 1,
    'Adjusting_Radio_Dashboard': 1,
    'Eating': 1,
    'Drinking': 1,
    'Hands off steering': 2,
    'Mobile Talking': 2,
    'Mobile Texting': 2,
    'Drowsiness': 2
}

# Color palette (BGR for OpenCV)
COLORS = {
    0: (0, 220, 0),     # Safe - Vivid Green
    1: (0, 140, 255),   # Warning - Orange
    2: (0, 0, 255),     # Danger - Red
    'hud_bg': (20, 20, 20),      # Dark charcoal for HUD background
    'text': (240, 240, 240),     # Off-white text
    'alert_bg': (0, 0, 180)      # Crimson background for alarms
}

def play_alert_sound(frequency=1800, duration=150):
    """Play warning beep sound using winsound (Windows only)"""
    if HAS_WINSOUND:
        try:
            winsound.Beep(frequency, duration)
        except Exception:
            pass

def main():
    parser = argparse.ArgumentParser(description="Live Driver Distraction Detection HUD")
    parser.add_argument("--source", type=str, default="0", help="Camera index (e.g. 0) or path to video file")
    parser.add_argument("--weights", type=str, default="best.pt", help="Path to YOLO model weights")
    parser.add_argument("--conf", type=float, default=0.05, help="Confidence threshold (set low for webcam sensitivity)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    args = parser.parse_args()

    # Determine which weights to load
    weights_path = args.weights
    is_test_mode = False

    if not os.path.exists(weights_path):
        # Check alternative local training output path
        local_train_weights = os.path.join("driver_distraction", "yolov8n_local", "weights", "best.pt")
        if os.path.exists(local_train_weights):
            weights_path = local_train_weights
        else:
            # Fallback to pretrained yolov8n
            print(f"Weights file '{weights_path}' not found.")
            print("Falling back to pretrained 'yolov8n.pt' for test/camera verification...")
            weights_path = "yolov8n.pt"
            is_test_mode = True

    print(f"Loading YOLO model from: {weights_path}...")
    model = YOLO(weights_path)

    # Initialize video capture
    # If source is digit, parse as int
    source = args.source
    if source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source: {args.source}")
        return

    # Set video properties for responsiveness
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # State variables
    attention_score = 100.0   # Rolling driver attention index (0 - 100)
    last_frame_time = time.time()
    distraction_start_time = None
    active_distractions = []
    
    # Sound throttle (prevent frozen UI due to synchronous beep calls)
    last_beep_time = 0
    beep_cooldown = 0.5 # seconds

    print("Driver Distraction System Initialized. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video stream ended or failed to read frame.")
            break

        current_time = time.time()
        dt = current_time - last_frame_time
        last_frame_time = current_time
        fps = 1.0 / dt if dt > 0 else 0.0

        # Run inference
        results = model.predict(frame, conf=args.conf, imgsz=args.imgsz, iou=0.45, verbose=False)
        
        # Parse detections
        detections = []
        highest_risk = 0
        current_frame_classes = []

        if results and len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                # Class ID and confidence
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Check if model has standard 10 classes or default YOLO COCO classes
                if is_test_mode or cls_id >= len(CLASS_NAMES):
                    # In test mode, we map coco classes to something simple, or just display whatever
                    class_name = model.names[cls_id]
                    risk = 1 if class_name != "person" else 0
                else:
                    class_name = CLASS_NAMES[cls_id]
                    risk = CLASS_RISK[class_name]

                # Bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append((x1, y1, x2, y2, class_name, conf, risk))
                current_frame_classes.append(class_name)
                
                if risk > highest_risk:
                    highest_risk = risk

        # Filter out distractions
        if is_test_mode:
            # Simple simulation logic in test mode: if person detected -> safe; if no person -> warning
            is_distracted = "person" not in [d[4] for d in detections] and len(detections) > 0
            active_distractions = ["Distracted (Test Mode)"] if is_distracted else []
            severity = 1 if is_distracted else 0
        else:
            # Real dataset logic
            active_distractions = [cls for cls in current_frame_classes if CLASS_RISK.get(cls, 0) > 0]
            severity = highest_risk

        # Update Driver Attention Score (EMA-based)
        if len(active_distractions) > 0:
            # Reduce attention score based on severity
            decay_rate = 25.0 if severity == 2 else 12.0
            attention_score = max(0.0, attention_score - decay_rate * dt)
            
            # Start distraction timer
            if distraction_start_time is None:
                distraction_start_time = current_time
        else:
            # Recover attention score
            attention_score = min(100.0, attention_score + 8.0 * dt)
            distraction_start_time = None

        # Determine distraction duration
        distracted_duration = 0.0
        if distraction_start_time is not None:
            distracted_duration = current_time - distraction_start_time

        # Render custom bounding boxes
        for x1, y1, x2, y2, name, conf, risk in detections:
            color = COLORS[risk]
            # Draw sleek box corners
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw semi-transparent label tag
            label = f"{name} {conf:.2f}"
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y1_label = max(y1, label_size[1] + 10)
            cv2.rectangle(frame, (x1, y1_label - label_size[1] - 5), (x1 + label_size[0] + 10, y1_label + base_line - 5), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1_label - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw HUD Panel
        h, w, _ = frame.shape
        hud_w = 320
        # Draw translucent background panel on the right side
        hud_overlay = frame.copy()
        cv2.rectangle(hud_overlay, (w - hud_w, 0), (w, h), COLORS['hud_bg'], -1)
        cv2.addWeighted(hud_overlay, 0.7, frame, 0.3, 0, frame)

        # Title
        cv2.putText(frame, "DRIVER HUD SYSTEM", (w - hud_w + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.line(frame, (w - hud_w + 20, 55), (w - 20, 55), (80, 80, 80), 1)

        # Attention Gauge
        cv2.putText(frame, "ATTENTION LEVEL:", (w - hud_w + 20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text'], 1, cv2.LINE_AA)
        
        # Color of attention gauge bar
        gauge_color = COLORS[0]
        if attention_score < 40:
            gauge_color = COLORS[2]
        elif attention_score < 70:
            gauge_color = COLORS[1]

        # Draw gauge bar
        bar_x_start = w - hud_w + 20
        bar_width = 280
        cv2.rectangle(frame, (bar_x_start, 110), (bar_x_start + bar_width, 125), (40, 40, 40), -1)
        filled_width = int(bar_width * (attention_score / 100.0))
        cv2.rectangle(frame, (bar_x_start, 110), (bar_x_start + filled_width, 125), gauge_color, -1)
        cv2.putText(frame, f"{attention_score:.1f}%", (bar_x_start + bar_width - 55, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, gauge_color, 2, cv2.LINE_AA)

        # Driver State
        status_text = "NORMAL"
        status_color = COLORS[0]
        if len(active_distractions) > 0:
            if severity == 2:
                status_text = "CRITICAL WARNING"
                status_color = COLORS[2]
            else:
                status_text = "DISTRACTED"
                status_color = COLORS[1]

        cv2.putText(frame, "CURRENT STATUS:", (w - hud_w + 20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text'], 1, cv2.LINE_AA)
        cv2.putText(frame, status_text, (w - hud_w + 20, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv2.LINE_AA)

        # Distraction Details
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - hud_w + 20, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text'], 1, cv2.LINE_AA)
        
        if distracted_duration > 0:
            cv2.putText(frame, f"Distract Duration: {distracted_duration:.1f}s", (w - hud_w + 20, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1, cv2.LINE_AA)

        # Active Violations List
        cv2.putText(frame, "DETECTED COGNITIVE LOADS:", (w - hud_w + 20, 335), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text'], 1, cv2.LINE_AA)
        y_offset = 365
        if len(active_distractions) == 0:
            cv2.putText(frame, "- None (Attentive)", (w - hud_w + 20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)
        else:
            for dist in list(set(active_distractions)):
                dist_risk = CLASS_RISK.get(dist, 1)
                cv2.putText(frame, f"- {dist.replace('_', ' ')}", (w - hud_w + 20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[dist_risk], 1, cv2.LINE_AA)
                y_offset += 25

        # Alert banner triggers if distracted for more than 1.5 seconds or attention index falls below 50%
        is_alarm_active = (distracted_duration > 1.5 and severity >= 1) or (attention_score < 50.0)
        
        if is_alarm_active:
            # Frequencies: High pitch danger (2200Hz) vs medium warning (1500Hz)
            beep_freq = 2200 if (severity == 2 or attention_score < 40) else 1500
            
            # Sound alarm in separate check to keep cv2 loop running smoothly
            if current_time - last_beep_time > beep_cooldown:
                play_alert_sound(frequency=beep_freq, duration=150)
                last_beep_time = current_time

            # Draw flashing warning box around screen
            overlay_alarm = frame.copy()
            cv2.rectangle(overlay_alarm, (0, 0), (w, h), COLORS['alert_bg'], 8)
            
            # Alert Text banner
            cv2.rectangle(overlay_alarm, (0, h - 80), (w, h), COLORS['alert_bg'], -1)
            cv2.putText(overlay_alarm, "WARNING: BRING HANDS TO WHEEL & EYES ON ROAD!", (40, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3, cv2.LINE_AA)
            
            # Flash effect based on milliseconds
            if int(current_time * 5) % 2 == 0:
                cv2.addWeighted(overlay_alarm, 0.5, frame, 0.5, 0, frame)

        # Notify if running test mode
        if is_test_mode:
            cv2.rectangle(frame, (10, 10), (520, 45), (0, 120, 255), -1)
            cv2.putText(frame, "TEST MODE (No Custom Weights): Using Default YOLOv8n", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Display the live window
        cv2.imshow("Driver Distraction & Attention Tracker HUD", frame)

        # Stop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("System shut down cleanly.")

if __name__ == "__main__":
    main()
