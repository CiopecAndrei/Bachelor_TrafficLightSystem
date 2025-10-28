import subprocess

# Starts the 3 components in parallel
counter = subprocess.Popen(["python3", "yolo11_export.py"])
gpio = subprocess.Popen(["python3", "Test_semafor.py"])
flask = subprocess.Popen(["python3", "web_server.py"])

counter.wait()
gpio.wait()
flask.wait()
