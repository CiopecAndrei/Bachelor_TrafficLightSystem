from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import time

# Load model
model = YOLO("yolo11n_ncnn_model", task="detect")

# Config
TASK = "detect"
COUNT = True
SHOW_TRACKS = False

# Shared state
track_history = defaultdict(lambda: [])
seen_ids_per_class = defaultdict(set)

def is_inside_roi(x, y, w, h, frame_width, frame_height):
    # Updated ROI parameters: move up, shift right, increase width (reduce side margins)
    roi_margin_x_left = int(frame_width * 0.10)    # shift right
    roi_margin_x_right = int(frame_width * 0.10)   # increase width
    roi_margin_y_top = int(frame_height * 0.35)    # move up slightly
    roi_margin_y_bottom = int(frame_height * 0.25) # same bottom margin

    x_center = int(x)
    y_center = int(y)

    return (roi_margin_x_left < x_center < frame_width - roi_margin_x_right and
            roi_margin_y_top < y_center < frame_height - roi_margin_y_bottom)

def process_and_save_frame():
    # Open video capture (use 0 for webcam, or provide path to video/image)
    cap = cv2.VideoCapture(6)  # Adjust to your camera index or video file path
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    success, frame = cap.read()
    if not success:
        print("Error: Could not read frame.")
        cap.release()
        return

    start_time = time.time()
    class_counts = defaultdict(int)

    # Perform detection
    results = model.predict(frame, conf=0.5)
    annotated_frame = results[0].plot()

    if results[0].boxes and results[0].boxes.cls is not None:
        boxes = results[0].boxes.xywh.cpu()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        names = results[0].names

        if COUNT:
            for box, cls_id in zip(boxes, class_ids):
                x, y, w, h = box
                class_name = names[cls_id]

                if class_name == "car":
                    if is_inside_roi(x, y, w, h, annotated_frame.shape[1], annotated_frame.shape[0]):
                        class_counts[class_name] += 1

    # Draw updated ROI rectangle
    frame_h, frame_w = annotated_frame.shape[:2]
    roi_margin_x_left = int(frame_w * 0.10)
    roi_margin_x_right = int(frame_w * 0.10)
    roi_margin_y_top = int(frame_h * 0.35)
    roi_margin_y_bottom = int(frame_h * 0.25)

    cv2.rectangle(
        annotated_frame,
        (roi_margin_x_left, roi_margin_y_top),
        (frame_w - roi_margin_x_right, frame_h - roi_margin_y_bottom),
        (255, 0, 0), 2
    )

    # Count display
    if COUNT:
        x0, y0 = 10, annotated_frame.shape[0] - 80
        for i, (cls_name, total) in enumerate(class_counts.items()):
            label = f"{cls_name}: {total}"
            y = y0 + i * 25
            cv2.putText(annotated_frame, label, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # FPS display (single frame, so approximate)
    fps = 1 / (time.time() - start_time)
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Save the annotated frame
    output_path = "detection_output.png"
    cv2.imwrite(output_path, annotated_frame)
    print(f"Image saved to: {output_path}")

    cap.release()

if __name__ == '__main__':
    process_and_save_frame()