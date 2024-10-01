import os
import pickle
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from PIL import Image
from capture_data import fileName

save_dir = f"captured_images_of_{fileName}"
data_dir = save_dir
encoding_file = f"trained_dataset.pkl"

detector = MTCNN()
embedder = FaceNet()

if os.path.exists(encoding_file):
    with open(encoding_file, 'rb') as f:
        known_face_encodings = pickle.load(f)
    print("Loaded face encodings from file.")
else:
    known_face_encodings = []

    for image_file in os.listdir(data_dir):
        image_path = os.path.join(data_dir, image_file)

        image = Image.open(image_path)
        image = np.array(image)

        faces = detector.detect_faces(image)

        if faces:
            x, y, width, height = faces[0]['box']
            face_image = image[y:y + height, x:x + width]

            face_encoding = embedder.embeddings([face_image])

            known_face_encodings.append(face_encoding[0])
            print(f"Face detected and encoded from image: {image_file}")
        else:
            print(f"No face detected in image: {image_file}")

    with open(encoding_file, 'wb') as f:
        pickle.dump(known_face_encodings, f)
    print("Face encodings prepared and saved to file.")
