# AI-Powered Smart Surveillance and Object Detection System

Compact Computer Vision assignment with nine modules in one Python file.

## Run in VS Code (Windows)
Use Python 3.10 or 3.11 and open this folder in VS Code.

```bash
python -m pip install -r requirements.txt
python main.py
```

For OCR only, install the Tesseract engine: https://tesseract-ocr.github.io/tessdoc/Installation.html . The program checks PATH and the default Windows installation directory. Other modules do not require it. First YOLO use downloads pretrained weights; internet is needed then.

## Demo
1. Image laboratory: resize, grayscale, histogram equalization, Canny, Otsu, connected-component segmentation, HOG, LBP and Haar face detection. A nine-panel dashboard is saved in outputs.
2. YOLO: detect and classify objects in the supplied cat photo or an authorized image.
3. Video: motion detection (MOG2), Farneback optical flow and optional YOLO ByteTrack tracking and OCR. Enter 0 for webcam; press Q to close. The synthetic sample demonstrates motion, not necessarily recognizable YOLO objects. For tracking use an authorized video of recognizable moving objects. Video dashboard is saved to outputs.
4. OCR: extract text from the supplied sign or your own image.

Use only authorized media. Do not publish private surveillance recordings or personal text. Sample photographs are from scikit-image (astronaut and chelsea); see https://scikit-image.org/docs/stable/api/skimage.data.html for image credits and redistribution terms. Sign and motion video are synthetic.

The assignment is an educational prototype, not a validated security product. Processing speed depends on hardware.
