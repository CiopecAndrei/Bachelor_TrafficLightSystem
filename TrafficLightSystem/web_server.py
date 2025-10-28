from flask import Flask, render_template, redirect, jsonify, request, session, flash
import json
import os

app = Flask(__name__)
app.secret_key = "your-secret-key-here"  # Schimbă asta cu o cheie secretă sigură!

STATE_FILE = "light_state.json"
BLINK_FILE = "blink_mode.json"
LOG_FILE = "log.txt"
TIMES_FILE = "traffic_times.json"
USERS_FILE = "users.json"
CAMERA_STATUS_FILE = "camera_status.json"

def get_blink_state():
    try:
        with open(BLINK_FILE, "r") as f:
            return json.load(f).get("blink", False)
    except:
        return False

def set_blink_state(state: bool):
    with open(BLINK_FILE, "w") as f:
        json.dump({"blink": state}, f)

def read_car_counts():
    car_counts = {}
    try:
        with open(LOG_FILE, "r") as f:
            for line in f:
                if line.startswith("Camera"):
                    parts = line.strip().split()
                    if len(parts) >= 3 and parts[2].isdigit():
                        direction = parts[1].strip(":")
                        count = int(parts[2])
                        car_counts[direction] = count
    except:
        pass
    return car_counts

def read_traffic_times():
    times = {}
    try:
        with open(TIMES_FILE, "r") as f:
            data = json.load(f)
            total_green_time = data.get("total_green_time", 3)
            green_dirs = data.get("green_dirs", [])
            for direction in TRAFFIC_LIGHTS.keys():
                times[direction] = total_green_time if direction in green_dirs else 0
    except:
        for direction in TRAFFIC_LIGHTS.keys():
            times[direction] = 0
    return times

def get_users():
    try:
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    except:
        return {}

def get_camera_status():
    try:
        with open(CAMERA_STATUS_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"errors": [], "blink": False, "message": "All cameras are working.", "error": False}

TRAFFIC_LIGHTS = {
    "West": {"red": 2, "yellow": 3, "green": 4},
    "South": {"red": 5, "yellow": 6, "green": 7},
    "East": {"red": 8, "yellow": 9, "green": 10},
    "North": {"red": 12, "yellow": 11, "green": 13},
}

@app.route("/")
def index():
    if not session.get("logged_in"):
        return redirect("/login")
    try:
        with open(STATE_FILE, "r") as f:
            states = json.load(f)
    except:
        states = {}
    car_counts = read_car_counts()
    times = read_traffic_times()
    camera_status = get_camera_status()
    manual_blink = get_blink_state()
    camera_blink = camera_status.get("blink", False)
    return render_template("index.html", lights=states, blinking=manual_blink or camera_blink, 
                          car_counts=car_counts, times=times, camera_error=camera_status.get("message"))

@app.route("/toggle")
def toggle():
    if not session.get("logged_in"):
        return redirect("/login")
    current_blink = not get_blink_state()
    set_blink_state(current_blink)
    return redirect("/")

@app.route("/update")
def update():
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized"}), 401
    try:
        with open(STATE_FILE, "r") as f:
            states = json.load(f)
    except:
        states = {}
    car_counts = read_car_counts()
    times = read_traffic_times()
    camera_status = get_camera_status()
    manual_blink = get_blink_state()
    camera_blink = camera_status.get("blink", False)
    return jsonify({
        "lights": states,
        "car_counts": car_counts,
        "times": times,
        "blinking": manual_blink or camera_blink,
        "camera_error": camera_status.get("message")
    })

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        users = get_users()
        if users.get(username) == password:
            session["logged_in"] = True
            return redirect("/")
        else:
            flash("Invalid credentials", "error")
            return redirect("/login")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("logged_in", None)
    return redirect("/login")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)