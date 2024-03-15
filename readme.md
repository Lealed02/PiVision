In order to run the project on the pi5 you must follow these steps and use a minimum 5V 3.1A power supply
1. sudo apt get python-opencv -> Not pip
2. python3 -m venv --system-site-packages venv
3. source ~venv/bin/activate
4. pip install ultralytics
5. python3 camera_inference.py or python3 camera_inference_FPS.py