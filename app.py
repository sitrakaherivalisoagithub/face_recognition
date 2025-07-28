
from flask import Flask, render_template, request, Response
import os
import json
import cv2
from datetime import datetime
from utils.face_utils import model, compare_embeddings

app = Flask(__name__)
DB_PATH = "employees.json"

# Charger les embeddings enregistrés
if os.path.exists(DB_PATH):
    with open(DB_PATH, "r") as f:
        known_faces = json.load(f)
else:
    known_faces = {}

def save_database():
    with open(DB_PATH, "w") as f:
        json.dump(known_faces, f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/register", methods=["POST"])
def register():
    name = request.form["name"]
    file = request.files["image"]
    path = f"static/images/{name}.jpg"
    file.save(path)

    img = cv2.imread(path)
    faces = model.get(img)
    if len(faces) > 0:
        embedding = faces[0].embedding
        known_faces[name] = embedding.tolist()
        save_database()
        return f"{name} enregistré avec succès !"
    return "Échec de l'enregistrement, aucun visage détecté.", 400

@app.route("/check", methods=["POST"])
def check():
    file = request.files["image"]
    path = "static/temp.jpg"
    file.save(path)
    
    img = cv2.imread(path)
    faces = model.get(img)
    
    if len(faces) == 0:
        return "Aucun visage détecté"

    test_embedding = faces[0].embedding

    for name, emb in known_faces.items():
        match, score = compare_embeddings(test_embedding, emb)
        if match:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return f"{name} reconnu. Pointage à {now} (similarité: {round(score, 2)})"
    return "Visage inconnu"

@app.route("/realtime")
def realtime():
    return render_template("realtime.html")


def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            faces = model.get(frame)
            for face in faces:
                box = face.bbox.astype(int)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                
                test_embedding = face.embedding
                match_name = "Inconnu"
                for name, emb in known_faces.items():
                    match, score = compare_embeddings(test_embedding, emb)
                    if match:
                        match_name = f"{name} ({round(score, 2)})"
                        break
                cv2.putText(frame, match_name, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
