import time

import torch
import numpy as np
from torchvision import models, transforms

import cv2
from PIL import Image

torch.backends.quantized.engine = 'qnnpack'

############################
# This project was originally made for the Pi4. Due to changes in the camera drivers
# this needs to be converted to work with picamera2
# See # OLD and # NEW tags for changes
 






# OLD
#cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
#cap.set(cv2.CAP_PROP_FPS, 36)


# NEW
from picamera2 import Picamera2
picam2 = Picamera2()

picam2.configure(picam2.create_preview_configuration(main={"format":"RGB888", "size" : (640,640)}))
picam2.start()



preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

net = models.quantization.mobilenet_v2(pretrained=True, quantize=True)
print(net)
exit()
# jit model to take it from ~20fps to ~30fps
net = torch.jit.script(net)

started = time.time()
last_logged = time.time()
frame_count = 0

with torch.no_grad():
    while True:
        # read frame

        # OLD
        # ret, img = cap.read()

        # NEW
        image = picam2.capture_array()

        # MAYBE INCOMPATIBLE
        # convert opencv output from BGR to RGB
        image = image[:, :, [2, 1, 0]]
        permuted = image
        # MAY INCOMPATIBLE

        # preprocess
        input_tensor = preprocess(image)

        # create a mini-batch as expected by the model
        input_batch = input_tensor.unsqueeze(0)

        # run model
        output = net(input_batch)
        # do something with output ...

        # log model performance
        frame_count += 1
        now = time.time()
        if now - last_logged > 1:
            print(f"{frame_count / (now-last_logged)} fps")
            last_logged = now
            frame_count = 0
