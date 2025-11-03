import cv2
import os

# your dataset path
dataset_path = "dataset/ak"  # change 'ak' if your name folder is different

for filename in sorted(os.listdir(dataset_path)):
    if not filename.lower().endswith(".jpg"):
        continue

    path = os.path.join(dataset_path, filename)
    img = cv2.imread(path)
    if img is None:
        continue

    cv2.imshow("Press 'y' to delete, any other key to keep", img)
    key = cv2.waitKey(0) & 0xFF

    if key == ord('y'):
        os.remove(path)
        print(f"🗑 Deleted {filename}")
    else:
        print(f"✅ Kept {filename}")

cv2.destroyAllWindows()
print("Cleanup finished!")
