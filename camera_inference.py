import cv2
from ultralytics import YOLO
model = YOLO("350A.pt")


def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
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

model = YOLO('350A.pt')
cap = cv2.VideoCapture(0)
cap.set(3, 650)
cap.set(4, 650)
_, img = cap.read()

#result_img = predict_and_detect(model, img, classes=[], conf=0.5)

while True:
    _, img = cap.read()
    img = cv2.imread("DSC_2716-min.jpg")
    result_img, _ = predict_and_detect(model, img, classes=[], conf=0.5)
    cv2.imshow('YOLO V8 Detection', result_img)     

    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()