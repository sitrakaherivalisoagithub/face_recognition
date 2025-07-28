# utils/face_utils.py
import numpy as np
import cv2
import insightface
# from deepface import DeepFace

model = insightface.app.FaceAnalysis(name='buffalo_l')# , root='./.insightface')
model.prepare(ctx_id=0, det_size=(640, 640))

def get_embedding_insight(img_path):
    img = cv2.imread(img_path)
    faces = model.get(img)
    if len(faces) == 0:
        return None
    return faces[0].embedding


def compare_embeddings(emb1, emb2, threshold=0.45):
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity >= threshold, similarity
