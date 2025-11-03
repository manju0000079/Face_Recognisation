# train_lbph.py
import os
import json
import cv2
import numpy as np

dataset_path = "dataset"
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

faces = []
labels = []
label_id = {}
current_id = 0

# iterate people
for person in sorted(os.listdir(dataset_path)):
    person_dir = os.path.join(dataset_path, person)
    if not os.path.isdir(person_dir):
        continue
    print("Processing:", person)
    if person not in label_id:
        label_id[person] = current_id
        current_id += 1
    id_ = label_id[person]
    for fname in sorted(os.listdir(person_dir)):
        fpath = os.path.join(person_dir, fname)
        img = cv2.imread(fpath)
        if img is None:
            print(" Warning: could not read", fpath)
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(rects) == 0:
            print("  No face found in", fpath)
            continue
        # choose the largest face
        x,y,w,h = max(rects, key=lambda r: r[2]*r[3])
        roi = gray[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi, (200,200))
        faces.append(roi_resized)
        labels.append(id_)

if len(faces) == 0:
    print("No faces found in any images. Please re-run capture_dataset and check images.")
    raise SystemExit(1)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))

os.makedirs("models", exist_ok=True)
recognizer.write("models/lbph_model.yml")

with open("models/labels.json", "w") as f:
    json.dump(label_id, f)

print("Training complete.")
print(f" Trained on {len(faces)} faces for {len(label_id)} people.")
print(" Saved: models/lbph_model.yml and models/labels.json")
