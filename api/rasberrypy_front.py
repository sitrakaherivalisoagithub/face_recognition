import cv2
import requests
import time
import base64
import asyncio

API_ENDPOINT = "http://127.0.0.1:8001/recognize" #"http://<ip-serveur>:<port>/api/face_recognition"
CAMERA_ID = "cam_0"


def encode_image(img):
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

def capture_and_send():
    cap = cv2.VideoCapture(0)  # Cam USB ou PiCam

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            _, img_encoded = cv2.imencode('.jpg', frame)
            print(face.shape)
            response = requests.post(API_ENDPOINT, files={"image": img_encoded.tobytes()})
            print(response.json())  # Résultat de l’API : {id, nom, autorisé, etc.}

            time.sleep(5)  # Pause pour éviter les multiples envois

        cv2.imshow("Pointage facial", frame)

        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_send()
