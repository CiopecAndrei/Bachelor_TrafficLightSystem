import RPi.GPIO as GPIO
import json
import time
import threading
import joblib

TRAFFIC_LIGHTS = {
    "West": {"red": 2, "yellow": 3, "green": 4},
    "South": {"red": 5, "yellow": 6, "green": 7},
    "East": {"red": 8, "yellow": 9, "green": 10},
    "North": {"red": 11, "yellow": 12, "green": 13},
}

MODEL_PATH = "/home/ciopec/yolo_try2/decision_tree_model.pkl"
LIGHT_STATE_FILE = "light_state.json"
LOG_FILE = "log.txt"
BLINK_FILE = "blink_mode.json"
TIMES_FILE = "traffic_times.json"
CAMERA_STATUS_FILE = "camera_status.json"  # Adăugat pentru a verifica erorile

# Timing constants (seconds)
YELLOW_TIME = 4.0
BLINK_COUNT = 3
BLINK_ON_TIME = 0.5
BLINK_OFF_TIME = 0.5

model = joblib.load(MODEL_PATH)
current_duration = 3
last_green_dirs = None

def get_blink_state():
    try:
        with open(BLINK_FILE, "r") as f:
            return json.load(f).get("blink", False)
    except:
        return False

def get_camera_blink_state():
    try:
        with open(CAMERA_STATUS_FILE, "r") as f:
            return json.load(f).get("blink", False)
    except:
        return False

def setup():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    for pins in TRAFFIC_LIGHTS.values():
        for pin in [pins["red"], pins["yellow"], pins["green"]]:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)

def set_light(direction, color):
    pins = TRAFFIC_LIGHTS[direction]
    for c in ["red", "yellow", "green"]:
        GPIO.output(pins[c], GPIO.LOW)
    GPIO.output(pins[color], GPIO.HIGH)
    update_state(direction, color)
    print(f"Set {direction} to {color}")
    time.sleep(0.1)

def update_state(name, color):
    try:
        with open(LIGHT_STATE_FILE, "r") as f:
            state = json.load(f)
    except:
        state = {}
    state[name] = color
    with open(LIGHT_STATE_FILE, "w") as f:
        json.dump(state, f)

def all_red(directions):
    for d in directions:
        set_light(d, "red")

def green_solid(directions):
    for d in directions:
        set_light(d, "green")

def yellow_on(directions):
    for d in directions:
        set_light(d, "yellow")

def green_off(directions):
    for d in directions:
        pins = TRAFFIC_LIGHTS[d]
        GPIO.output(pins["green"], GPIO.LOW)
        update_state(d, "off")
        print(f"Green off for {d}")
        time.sleep(0.1)

def blink_thread():
    while True:
        manual_blink = get_blink_state()  # Blinking manual
        camera_blink = get_camera_blink_state()  # Blinking din erori
        if manual_blink or camera_blink:
            for tl in TRAFFIC_LIGHTS.values():
                GPIO.output(tl["red"], GPIO.LOW)
                GPIO.output(tl["green"], GPIO.LOW)
                GPIO.output(tl["yellow"], GPIO.HIGH)
            time.sleep(BLINK_ON_TIME)
            for tl in TRAFFIC_LIGHTS.values():
                GPIO.output(tl["yellow"], GPIO.LOW)
            time.sleep(BLINK_OFF_TIME)
        else:
            time.sleep(0.1)

def predict_and_update():
    global current_duration, last_green_dirs

    while True:
        if get_blink_state() or get_camera_blink_state():  # Verifică ambele stări pentru a suspenda predicția
            print("Blink mode active, skipping prediction")
            time.sleep(0.5)
            continue

        try:
            with open(LOG_FILE, "r") as f:
                lines = f.readlines()

            counts = {'North': 0, 'East': 0, 'South': 0, 'West': 0}
            for line in lines:
                if line.startswith("Camera"):
                    parts = line.strip().split()
                    direction = parts[1].strip(":")
                    count = int(parts[2])
                    if direction in counts:
                        counts[direction] = count

            X = [[counts['South'], counts['North'], counts['West'], counts['East']]]
            bucket = model.predict(X)[0]
            bucket_to_seconds = {0: 3, 1: 6, 2: 9}
            total_green_time = bucket_to_seconds.get(bucket, 3)

            sn_total = counts['South'] + counts['North']
            ew_total = counts['East'] + counts['West']

            if sn_total >= ew_total:
                green_dirs = ['South', 'North']
                red_dirs = ['East', 'West']
            else:
                green_dirs = ['East', 'West']
                red_dirs = ['South', 'North']

            print(f"Counts: {counts}, Bucket: {bucket}, Green time: {total_green_time}s, Green: {green_dirs}, Red: {red_dirs}")

            traffic_times = {
                "total_green_time": total_green_time,
                "green_dirs": green_dirs
            }
            with open(TIMES_FILE, "w") as f:
                json.dump(traffic_times, f)

            if last_green_dirs == green_dirs:
                print(f"Same green directions {green_dirs}, solid green for {total_green_time}s")
                green_solid(green_dirs)
                all_red(red_dirs)
                time.sleep(total_green_time)
            else:
                if last_green_dirs:
                    print(f"Blinking green on old green directions {last_green_dirs} {BLINK_COUNT} times")
                    for _ in range(BLINK_COUNT):
                        green_off(last_green_dirs)
                        time.sleep(BLINK_OFF_TIME)
                        green_solid(last_green_dirs)
                        time.sleep(BLINK_ON_TIME)

                    print(f"Yellow on old green directions {last_green_dirs} for {YELLOW_TIME}s")
                    yellow_on(last_green_dirs)
                    all_red(red_dirs)
                    time.sleep(YELLOW_TIME)

                    print(f"Setting old green directions {last_green_dirs} to red")
                    all_red(last_green_dirs)

                    time.sleep(0.5)

                all_red(red_dirs)
                print(f"New green directions {green_dirs}, solid green for {total_green_time}s")
                green_solid(green_dirs)
                time.sleep(total_green_time)

                last_green_dirs = green_dirs

        except Exception as e:
            print(f"Prediction error: {e}")
            time.sleep(3)

def run():
    setup()
    threading.Thread(target=blink_thread, daemon=True).start()
    predict_and_update()

if __name__ == "__main__":
    run()