import cv2
import time
import os
import argparse
from ultralytics import YOLO

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
    0: (0, 220, 0),     # Safe - Green
    1: (0, 140, 255),   # Warning - Orange
    2: (0, 0, 255),     # Danger - Red
    'hud_bg': (20, 20, 20),      # Dark charcoal for HUD background
    'text': (240, 240, 240),     # Off-white
    'alert_bg': (0, 0, 180)      # Crimson background for alarms
}

def main():
    parser = argparse.ArgumentParser(description="Process video with Driver Distraction Detection YOLO Model")
    parser.add_argument("--input", type=str, default="test1.mp4", help="Input video file path")
    parser.add_argument("--output", type=str, default="output_test1.mp4", help="Output video file path")
    parser.add_argument("--weights", type=str, default="best.pt", help="YOLO model weights path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--limit-frames", type=int, default=300, help="Limit number of frames to process (0 for all)")
    args = parser.parse_args()

    if not os.path.exists(args.weights):
        print(f"Error: Weights file '{args.weights}' not found. Please train first.")
        return

    if not os.path.exists(args.input):
        print(f"Error: Input video '{args.input}' not found.")
        return

    print(f"Loading YOLO model from: {args.weights}...")
    model = YOLO(args.weights)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: Could not open input video: {args.input}")
        return

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps == 0 or fps is None:
        fps = 30.0

    print(f"Input Video: {args.input}")
    print(f"Resolution: {width}x{height} | FPS: {fps} | Total Frames: {total_frames}")

    # Set up video writer (using mp4v codec for mp4 format)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    # State variables
    attention_score = 100.0
    distraction_start_frame = None
    frame_count = 0
    dt = 1.0 / fps

    limit = args.limit_frames
    max_frames = limit if (limit > 0 and limit < total_frames) else total_frames
    
    print(f"Processing {max_frames} frames... Saving output to {args.output}")
    
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= max_frames:
            break

        frame_count += 1
        
        # Print progress
        if frame_count % 50 == 0:
            elapsed = time.time() - start_time
            speed = frame_count / elapsed if elapsed > 0 else 0
            print(f"Processed {frame_count}/{max_frames} frames ({frame_count/max_frames*100:.1f}%) | Current Speed: {speed:.1f} FPS")

        # Run inference
        results = model.predict(frame, conf=args.conf, verbose=False)
        
        detections = []
        highest_risk = 0
        current_frame_classes = []

        if results and len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls_id < len(CLASS_NAMES):
                    class_name = CLASS_NAMES[cls_id]
                    risk = CLASS_RISK[class_name]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append((x1, y1, x2, y2, class_name, conf, risk))
                    current_frame_classes.append(class_name)
                    if risk > highest_risk:
                        highest_risk = risk

        # Distraction logic
        active_distractions = [cls for cls in current_frame_classes if CLASS_RISK.get(cls, 0) > 0]
        severity = highest_risk

        if len(active_distractions) > 0:
            decay_rate = 25.0 if severity == 2 else 12.0
            attention_score = max(0.0, attention_score - decay_rate * dt)
            if distraction_start_frame is None:
                distraction_start_frame = frame_count
        else:
            attention_score = min(100.0, attention_score + 8.0 * dt)
            distraction_start_frame = None

        distracted_duration = 0.0
        if distraction_start_frame is not None:
            distracted_duration = (frame_count - distraction_start_frame) * dt

        # Draw custom bounding boxes
        for x1, y1, x2, y2, name, conf, risk in detections:
            color = COLORS[risk]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{name} {conf:.2f}"
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y1_label = max(y1, label_size[1] + 10)
            cv2.rectangle(frame, (x1, y1_label - label_size[1] - 5), (x1 + label_size[0] + 10, y1_label + base_line - 5), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1_label - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw HUD Panel
        h, w, _ = frame.shape
        hud_w = int(w * 0.25) if w > 800 else 240
        hud_overlay = frame.copy()
        cv2.rectangle(hud_overlay, (w - hud_w, 0), (w, h), COLORS['hud_bg'], -1)
        cv2.addWeighted(hud_overlay, 0.7, frame, 0.3, 0, frame)

        # Title
        cv2.putText(frame, "DRIVER HUD SYSTEM", (w - hud_w + 15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.line(frame, (w - hud_w + 15, 55), (w - 15, 55), (80, 80, 80), 1)

        # Attention Gauge
        cv2.putText(frame, "ATTENTION LEVEL:", (w - hud_w + 15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS['text'], 1, cv2.LINE_AA)
        
        gauge_color = COLORS[0]
        if attention_score < 40:
            gauge_color = COLORS[2]
        elif attention_score < 70:
            gauge_color = COLORS[1]

        bar_x_start = w - hud_w + 15
        bar_width = hud_w - 30
        cv2.rectangle(frame, (bar_x_start, 110), (bar_x_start + bar_width, 125), (40, 40, 40), -1)
        filled_width = int(bar_width * (attention_score / 100.0))
        cv2.rectangle(frame, (bar_x_start, 110), (bar_x_start + filled_width, 125), gauge_color, -1)
        cv2.putText(frame, f"{attention_score:.1f}%", (bar_x_start + bar_width - 55, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, gauge_color, 2, cv2.LINE_AA)

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

        cv2.putText(frame, "CURRENT STATUS:", (w - hud_w + 15, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS['text'], 1, cv2.LINE_AA)
        cv2.putText(frame, status_text, (w - hud_w + 15, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA)

        # Distraction Details
        if distracted_duration > 0:
            cv2.putText(frame, f"Distract Duration: {distracted_duration:.1f}s", (w - hud_w + 15, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1, cv2.LINE_AA)

        # Active Violations List
        cv2.putText(frame, "DETECTED COGNITIVE LOADS:", (w - hud_w + 15, 295), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS['text'], 1, cv2.LINE_AA)
        y_offset = 325
        if len(active_distractions) == 0:
            cv2.putText(frame, "- None (Attentive)", (w - hud_w + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)
        else:
            for dist in list(set(active_distractions)):
                dist_risk = CLASS_RISK.get(dist, 1)
                cv2.putText(frame, f"- {dist.replace('_', ' ')}", (w - hud_w + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[dist_risk], 1, cv2.LINE_AA)
                y_offset += 25

        # Alert banner
        is_alarm_active = (distracted_duration > 1.5 and severity >= 1) or (attention_score < 50.0)
        
        if is_alarm_active:
            overlay_alarm = frame.copy()
            cv2.rectangle(overlay_alarm, (0, 0), (w, h), COLORS['alert_bg'], 8)
            cv2.rectangle(overlay_alarm, (0, h - 80), (w, h), COLORS['alert_bg'], -1)
            cv2.putText(overlay_alarm, "WARNING: BRING HANDS TO WHEEL & EYES ON ROAD!", (30, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Flash effect
            if int(frame_count / 5) % 2 == 0:
                cv2.addWeighted(overlay_alarm, 0.5, frame, 0.5, 0, frame)

        # Write frame to output video
        out.write(frame)

    # Cleanup
    cap.release()
    out.release()
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nProcessing finished!")
    print(f"Successfully processed {frame_count} frames in {total_time:.1f} seconds ({frame_count / total_time:.1f} FPS).")
    print(f"Output saved to: {args.output}")

if __name__ == "__main__":
    main()
