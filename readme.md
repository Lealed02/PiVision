In order to run the project on the pi5 you must follow these steps and use a minimum 5V 3.1A power supply

1. python3 -m venv --system-site-packages venv
2. sudo apt get python-opencv -> Not pip
3. source ~venv/bin/activate
4. python3 camera_inference.py