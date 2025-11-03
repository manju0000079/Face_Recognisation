# train_tweak.py - attempt to recover faces with different detect params
import os, json, cv2, numpy as np

dataset_path = "dataset"
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

faces = []
labels = []
label_id = {}
cur = 0

for person in sorted(os.listdir(dataset_path)):
    pd = os.path.join(dataset_path, person)
    if not os.path.isdir(pd): continue
    if person not in label_id:
        label_id[person] = cur; cur += 1
    id_ = label_id[person]
    for f in sorted(os.listdir(pd)):
        if not f.lower().endswith(".jpg"): continue
        p = os.path.join(pd, f)
        img = cv2.imread(p)
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # try permissive parameters first
        rects = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(30,30))
        if len(rects) == 0:
            # fallback: original settings
            rects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(rects) == 0:
            print("No face found in", p)
            continue
        x,y,w,h = max(rects, key=lambda r: r[2]*r[3])
        roi = cv2.resize(gray[y:y+h, x:x+w], (200,200))
        faces.append(roi); labels.append(id_)

if len(faces) == 0:
    print("No faces found at all. You must re-capture images with better framing.")
    raise SystemExit(1)

rec = cv2.face.LBPHFaceRecognizer_create()
rec.train(faces, np.array(labels))
os.makedirs("models", exist_ok=True)
rec.write("models/lbph_model.yml")
with open("models/labels.json","w") as f:
    json.dump(label_id, f)
print("Training done. Faces:", len(faces))
