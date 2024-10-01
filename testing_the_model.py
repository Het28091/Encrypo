import pickle
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from PIL import Image

encoding_file = 'trained_data_captured_images_HET_PRAJAPATI.pkl'

with open(encoding_file, 'rb') as f:
    known_face_encodings = pickle.load(f)

print("Loaded face encodings from file.")

detector = MTCNN()
embedder = FaceNet()


def is_same_face(new_face_path):
    image = Image.open(new_face_path)
    image = np.array(image)

    faces = detector.detect_faces(image)

    if not faces:
        raise ValueError(f"No face detected in the new face image: {new_face_path}")

    x, y, width, height = faces[0]['box']
    face_image = image[y:y + height, x:x + width]

    new_face_encoding = embedder.embeddings([face_image])[0]

    distances = np.linalg.norm(known_face_encodings - new_face_encoding, axis=1)

    threshold = 0.75
    matches = distances <= threshold

    return any(matches)


new_face_path = 'new_face.jpg.jpg'  # Path to the new face image
try:
    is_match = is_same_face(new_face_path)
    print(f"Is the new face the same as the trained face? {'Yes' if is_match else 'No'}")
except ValueError as e:
    print(e)
