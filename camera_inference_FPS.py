import cv2
from ultralytics import YOLO
from picamera2 import Picamera2
import time

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format":"RGB888", "size" : (640,640)}))
picam2.start()

model = YOLO("100A.pt") # Default


def predict(chosen_model, img, classes=[], conf=0.5):
    
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
        exit()
    else:
        results = chosen_model.predict(img, conf=conf)

    return results


def predict_and_detect(chosen_model, img, classes=[], conf=0.5):
    results = predict(chosen_model, img, classes, conf=conf)

    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), 2)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
    return img, results

model = YOLO('100A.pt') # Override default, 100A sometimes performs better than 300A


# Set up FPS counter
started = time.time()
last_logged = time.time()
frame_count = 0

while True:
    # Capture images
    img = picam2.capture_array()

    # Run inference on the image and draw boxes
    result_img, _ = predict_and_detect(model, img, classes=[], conf=0.5)
    
    # Display image
    cv2.imshow('YOLO V8 Detection', result_img)        

    # Count fps
    frame_count += 1
    now = time.time()
    if now - last_logged > 1:
        print(f"{frame_count / (now-last_logged)} fps")
        last_logged = now
        frame_count = 0

    if cv2.waitKey(0) & 0xFF == ord(' '):
        break

cv2.destroyAllWindows()
