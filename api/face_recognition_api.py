# 📦 Requirements: fastapi, uvicorn, pymongo (ou json), numpy, pydantic
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
from insightface.app import FaceAnalysis
import numpy as np
import cv2
import time
import os
import json
from datetime import datetime
from io import BytesIO
import base64

# 🔧 Configuration
USE_JSON = True  # Sinon MongoDB
EMBEDDINGS_FILE = "known_faces.json" # "known_faces.json"
THRESHOLD = 0.45

if not USE_JSON:
    from pymongo import MongoClient
    client = MongoClient("mongodb://localhost:27017")
    db = client["face_db"]
    collection = db["embeddings"]

# 🧠 Initialiser InsightFace
app_model = FaceAnalysis(name="buffalo_l")
app_model.prepare(ctx_id=0)

app = FastAPI()


# 📦 Base JSON si nécessaire
def load_json_embeddings():
    if not os.path.exists(EMBEDDINGS_FILE):
        return []
    with open(EMBEDDINGS_FILE, 'r') as f:
        return json.load(f)


def save_json_embeddings(data):
    with open(EMBEDDINGS_FILE, 'w') as f:
        json.dump(data, f)


# 📏 Cosine similarity
def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# 📍 Endpoint: Enregistrer une nouvelle personne
class Registration(BaseModel):
    name: str


class RecognitionRequest(BaseModel):
    image: str
    device_id: str
    timestamp: float

@app.get("/")
def home():
    return {"response": "OK"}

@app.post("/register")
async def register_person(name: str, image: UploadFile = File(...)):
    image_data = await image.read()
    np_img = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    faces = app_model.get(frame)
    if not faces:
        return {"error": "Aucun visage détecté."}

    embedding = faces[0].embedding.tolist()
    new_entry = {"name": name, "embedding": embedding}

    if USE_JSON:
        data = load_json_embeddings()
        data.append(new_entry)
        save_json_embeddings(data)
    else:
        collection.insert_one(new_entry)

    return {"message": f"{name} enregistré avec succès."}


# 📍 Endpoint: Reconnaissance faciale
@app.post("/recognize")
async def recognize(image: UploadFile = File(...)):
    image_data = await image.read()
    np_img = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    print(f'shape frame {frame.shape}')

    faces = app_model.get(frame)
    if not faces:
        return {"recognized": False, "message": "Aucun visage détecté."}

    face_emb = faces[0].embedding

    if USE_JSON:
        database = load_json_embeddings()
    else:
        database = list(collection.find({}, {"_id": 0}))

    best_score = 0
    best_match = None
    for person in database:
        score = cosine_similarity(np.array(person["embedding"]), face_emb)
        if score > best_score and score > THRESHOLD:
            best_score = score
            best_match = person["name"]

    result = {
        "recognized": bool(best_match),
        "name": best_match,
        "timestamp": datetime.utcnow().isoformat(),
        "score": best_score
    }
    return result


@app.post("/recognize_image")
def recognize_image(request: RecognitionRequest):
    try:
        img_data = base64.b64decode(request.image)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        faces = app_model.get(img)
        if not faces:
            return {"recognized": False, "message": "Aucun visage détecté."}

        face_emb = faces[0].embedding

        if USE_JSON:
            database = load_json_embeddings()
        else:
            database = list(collection.find({}, {"_id": 0}))

        best_score = 0
        best_match = None
        for person in database:
            score = cosine_similarity(np.array(person["embedding"]), face_emb)
            if score > best_score and score > THRESHOLD:
                best_score = score
                best_match = person["name"]

        result = {
            "recognized": bool(best_match),
            "name": best_match,
            "timestamp": request.timestamp,
            "score": best_score,
            "device": request.device_id
        }
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# uvicorn face_recognition_api:app --reload
