from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog, local_binary_pattern
import pytesseract
import shutil
import os

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)

# 1-4: Image processing, segmentation, descriptors and face detection

def image_lab(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError('Cannot open image')
    image = cv2.resize(image, (640, 480))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)
    edges = cv2.Canny(enhanced, 70, 150)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Segment connected foreground regions using the threshold mask.
    mask = 255 - binary if np.mean(binary) > 127 else binary
    _, labels = cv2.connectedComponents(mask)
    segmented = np.uint8((labels * 47) % 255)
    features, hog_view = hog(gray, pixels_per_cell=(16,16), cells_per_block=(2,2), visualize=True)
    lbp = local_binary_pattern(gray, 8, 1, method='uniform')
    print('HOG feature count:', len(features))
    print('LBP histogram:', np.bincount(lbp.astype(int).ravel(), minlength=10))
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = image.copy()
    for x,y,w,h in cascade.detectMultiScale(gray, 1.1, 5):
        cv2.rectangle(faces, (x,y), (x+w,y+h), (0,255,0), 2)
    panels = {'Resized':image, 'Grayscale':gray, 'Enhanced':enhanced, 'Edges':edges,
              'Threshold':binary, 'Segments':segmented, 'HOG':hog_view, 'LBP':lbp, 'Faces':faces}
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    for ax, (name, img) in zip(axes.flat, panels.items()):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img, cmap='gray')
        ax.set_title(name)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(OUT / 'image_dashboard.png')
    plt.show()

# 5: YOLO object detection

def detect(path):
    from ultralytics import YOLO
    result = YOLO('yolo11n.pt')(str(path), verbose=False)[0]
    frame = result.plot()
    cv2.imwrite(str(OUT / 'objects.jpg'), frame)
    for box in result.boxes:
        print(result.names[int(box.cls.item())], round(box.conf.item(), 2))
    cv2.imshow('YOLO detection', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 8: OCR (shared by image and video modes)

def ocr(frame):
    exe = os.getenv('TESSERACT_CMD') or shutil.which('tesseract')
    if not exe:
        candidate = Path(r'C:\Program Files\Tesseract-OCR\tesseract.exe')
        if candidate.exists(): exe = str(candidate)
    if not exe:
        raise RuntimeError('Tesseract OCR is required; see README.')
    pytesseract.pytesseract.tesseract_cmd = exe
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return pytesseract.image_to_string(binary, config='--psm 11').strip()

# 6-7, 9: Motion, optical flow, tracking and live dashboard

def video(source, tracking=False, text=False):
    model = None
    if tracking:
        from ultralytics import YOLO
        model = YOLO('yolo11n.pt')
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError('Cannot open video')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    writer = cv2.VideoWriter(str(OUT / 'processed_video.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (1280,960))
    bg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    prev, count = None, 0
    log = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.resize(frame, (640,480))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask = bg.apply(frame)
            motion = frame.copy()
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if cv2.contourArea(c) > 500:
                    x,y,w,h = cv2.boundingRect(c)
                    cv2.rectangle(motion,(x,y),(x+w,y+h),(0,255,255),2)
            flow_view = np.zeros_like(frame)
            if prev is not None:
                flow = cv2.calcOpticalFlowFarneback(prev,gray,None,.5,3,15,3,5,1.2,0)
                mag, ang = cv2.cartToPolar(flow[...,0],flow[...,1])
                hsv = np.zeros_like(frame)
                hsv[...,0], hsv[...,1] = ang*90/np.pi, 255
                hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
                flow_view = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
            prev = gray
            tracked = model.track(frame, persist=True, tracker='bytetrack.yaml', verbose=False)[0].plot() if model else frame.copy()
            if text and count % 30 == 0:
                log.append(f'Frame {count}: {ocr(frame)}')
            panels = [tracked, motion, cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR), flow_view]
            for panel, title in zip(panels,['YOLO tracking' if tracking else 'Original','Motion','Foreground','Optical flow']):
                cv2.putText(panel,title,(12,30),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,255,0),2)
            dashboard = np.vstack([np.hstack(panels[:2]),np.hstack(panels[2:])])
            writer.write(dashboard)
            cv2.imshow('Surveillance dashboard - Q to exit', cv2.resize(dashboard,(960,720)))
            count += 1
            if cv2.waitKey(1) & 255 == ord('q'): break
    finally:
        cap.release(); writer.release(); cv2.destroyAllWindows()
        if text: (OUT / 'video_ocr.txt').write_text('\n'.join(log),encoding='utf-8')
    print('Processed frames:', count)

if __name__ == '__main__':
    while True:
        print('\n1. Image laboratory  2. YOLO detection  3. Video motion/tracking  4. OCR  0. Exit')
        choice = input('Select module: ').strip()
        try:
            if choice == '1': image_lab(input('Image path (Enter for sample): ') or ROOT/'inputs/astronaut.jpg')
            elif choice == '2': detect(input('Image path (Enter for sample): ') or ROOT/'inputs/cat.jpg')
            elif choice == '3':
                path = input('Video path, 0 for webcam, Enter for sample: ').strip()
                video(0 if path == '0' else path or str(ROOT/'inputs/motion_demo.mp4'),
                      input('Enable YOLO tracking? (y/n): ').lower() == 'y',
                      input('Enable video OCR? (y/n): ').lower() == 'y')
            elif choice == '4':
                frame = cv2.imread(str(input('Image path (Enter for sample): ') or ROOT/'inputs/ocr_sign.png'))
                if frame is None: raise ValueError('Cannot open image')
                result = ocr(frame); print(result)
                (OUT/'ocr.txt').write_text(result,encoding='utf-8')
            elif choice == '0': break
        except Exception as e: print('Error:', e)
