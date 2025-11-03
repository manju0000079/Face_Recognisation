# capture_dataset.py
import cv2
import os

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    name = input("Enter person name (no spaces): ").strip()
    if not name:
        print("Name required.")
        return
    outdir = os.path.join("dataset", name)
    ensure_dir(outdir)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return
    count = 0
    print("Press SPACE to capture an image, ESC to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        display = frame.copy()
        h, w = display.shape[:2]
        cv2.putText(display, f"Person: {name}  Count: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow("Capture (SPACE to save, ESC to quit)", display)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        if key == 32:  # SPACE
            fname = os.path.join(outdir, f"{count:03d}.jpg")
            cv2.imwrite(fname, frame)
            print("Saved", fname)
            count += 1
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
