# show_failures.py
import cv2, os

dataset = "dataset"
person = None
# find the person folder automatically if only one
people = [d for d in os.listdir(dataset) if os.path.isdir(os.path.join(dataset,d))]
if len(people) == 1:
    person = people[0]
else:
    person = input("Person folder name: ").strip()

p_dir = os.path.join(dataset, person)
print("Scanning:", p_dir)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

for fname in sorted(os.listdir(p_dir)):
    if not fname.lower().endswith(".jpg"):
        continue
    path = os.path.join(p_dir, fname)
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(rects) == 0:
        print("No face:", fname)
        # show image so you can judge
        cv2.imshow("NO FACE FOUND - press key to continue", img)
        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            break
    else:
        # draw boxes and show
        for (x,y,w,h) in rects:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imshow("Found (press key)", img)
        cv2.waitKey(200)  # short pause
cv2.destroyAllWindows()
