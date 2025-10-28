from ultralytics import YOLO
import cv2
import time
import json
from datetime import datetime

model = YOLO("yolo11n_ncnn_model", task="detect")

CAMERA_PATHS = {
    "North": "/dev/v4l/by-path/platform-fd500000.pcie-pci-0000:01:00.0-usb-0:1.1.1:1.0-video-index0",
    "East": "/dev/v4l/by-path/platform-fd500000.pcie-pci-0000:01:00.0-usb-0:1.1.2:1.0-video-index0",
    "South": "/dev/v4l/by-path/platform-fd500000.pcie-pci-0000:01:00.0-usb-0:1.1.3:1.0-video-index0",
    "West": "/dev/v4l/by-path/platform-fd500000.pcie-pci-0000:01:00.0-usb-0:1.1.4:1.0-video-index0"
}

SWITCH_INTERVAL = 3  # seconds

car_counts = {name: 0 for name in CAMERA_PATHS}

def is_inside_roi(x, y, w, h, frame_width, frame_height):
    roi_margin_x_left = int(frame_width * 0.01)
    roi_margin_x_right = int(frame_width * 0.10)
    roi_margin_y_top = int(frame_height * 0.35)
    roi_margin_y_bottom = int(frame_height * 0.25)

    x_center = int(x)
    y_center = int(y)

    return (roi_margin_x_left < x_center < frame_width - roi_margin_x_right and
            roi_margin_y_top < y_center < frame_height - roi_margin_y_bottom)

def write_log_file():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("log.txt", "w") as log_file:
        for name, count in car_counts.items():
            log_file.write(f"Camera {name}: {count} cars - {timestamp}\n")

def update_camera_status(error_name=None):
    status_file = "camera_status.json"
    try:
        with open(status_file, "r") as f:
            status = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        status = {"errors": [], "last_error": None}
    
    if "errors" not in status:
        status["errors"] = []
    if "last_error" not in status:
        status["last_error"] = None

    # Verifică starea tuturor camerelor de fiecare dată
    all_working = True
    current_error = None
    for name in CAMERA_PATHS:
        cap = cv2.VideoCapture(CAMERA_PATHS[name])
        if not cap.isOpened():
            all_working = False
            current_error = name
        cap.release()

    if not all_working and current_error:
        if current_error not in status["errors"]:
            status["errors"].append(current_error)
        status["last_error"] = current_error
    elif all_working:
        status["errors"] = []
        status["last_error"] = None

    status["blink"] = bool(status["errors"])
    status["message"] = f"Camera with error: {status['last_error']}" if status["last_error"] else "All cameras are working."
    status["error"] = bool(status["last_error"])

    with open(status_file, "w") as f:
        json.dump(status, f)

def monitor_cameras():
    while True:
        for name, path in CAMERA_PATHS.items():
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                print(f"Camera {name} failed to open.")
                update_camera_status(name)
                continue

            success, frame = cap.read()
            cap.release()
            if not success:
                print(f"Camera {name} failed to read.")
                continue

            results = model.predict(frame, conf=0.5)
            count = 0

            if results[0].boxes and results[0].boxes.cls is not None:
                boxes = results[0].boxes.xywh.cpu()
                class_ids = results[0].boxes.cls.int().cpu().tolist()
                names_list = results[0].names

                for box, cls_id in zip(boxes, class_ids):
                    x, y, w, h = box
                    class_name = names_list[cls_id]
                    if class_name == "car":
                        if is_inside_roi(x, y, w, h, frame.shape[1], frame.shape[0]):
                            count += 1

            car_counts[name] = count
            write_log_file()
            print(f"Updated Camera {name}: {count} cars")
            update_camera_status()

            time.sleep(SWITCH_INTERVAL)

if __name__ == '__main__':
    monitor_cameras()