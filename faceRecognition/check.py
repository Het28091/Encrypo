import pickle
import numpy as np
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN
import cv2
import time

encoding_file = "trained_dataset.pkl"
with open(encoding_file, 'rb') as f:
    known_face_encodings = pickle.load(f)

detector = MTCNN()
embedder = FaceNet()

cap = cv2.VideoCapture(0)

def is_same_face(face_encoding):
    distances = np.linalg.norm(np.array(known_face_encodings) - face_encoding, axis=1)
    threshold = 0.75
    matches = distances <= threshold
    return any(matches)

start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = detector.detect_faces(rgb_frame)

    if faces:
        x, y, width, height = faces[0]['box']
        face_image = rgb_frame[y:y + height, x:x + width]
        face_encoding = embedder.embeddings([face_image])[0]
        if is_same_face(face_encoding):
            print("Face matched! Quitting camera.")
            break
        else:
            print("No match found in the database.")
    cv2.imshow("Real-time Face Detection", frame)

    elapsed_time = time.time() - start_time
    if elapsed_time > 40:
        print("Time limit exceeded. Quitting camera.")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
