import cv2
from ultralytics import YOLO
from picamera2 import Picamera2

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

while True:
    img = picam2.capture_array()
    
    result_img, _ = predict_and_detect(model, img, classes=[], conf=0.5)
    
    cv2.imshow('YOLO V8 Detection', result_img)     

    if cv2.waitKey(0) & 0xFF == ord(' '):
        break

cv2.destroyAllWindows()
