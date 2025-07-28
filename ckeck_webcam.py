# check_webcam.py
import cv2
import numpy as np
import insightface
import os
import json
from utils.face_utils import compare_embeddings

# Charger le modèle
model = insightface.app.FaceAnalysis(name='buffalo_l')
model.prepare(ctx_id=0, det_size=(640, 640))

# Charger les embeddings enregistrés
DB_PATH = "employees.json"
if os.path.exists(DB_PATH):
    with open(DB_PATH, "r") as f:
        known_faces = json.load(f)
else:
    known_faces = {}

# Convertir les listes en np.array
known_faces = {k: np.array(v) for k, v in known_faces.items()}

# Initialiser la webcam
cap = cv2.VideoCapture(0)
print("Appuyez sur Q pour quitter.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = model.get(frame)
    for face in faces:
        bbox = face.bbox.astype(int)
        embedding = face.embedding

        label = "Visage inconnu"
        for name, ref_emb in known_faces.items():
            match, score = compare_embeddings(embedding, ref_emb)
            if match:
                label = f"{name} ({round(score, 2)})"
                break

        # Dessiner la bbox + label
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)
        cv2.putText(frame, label, (bbox[0], bbox[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    cv2.imshow("Pointage facial", frame)

    # Quitter avec Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
