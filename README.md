# 👁️ Computer Vision AI System

A comprehensive multi-task computer vision system with object detection, face recognition, image classification, and OCR capabilities.

## 🎯 Features

- **Object Detection**: YOLOv5-based real-time object detection
- **Face Detection & Recognition**: MTCNN + FaceNet for face analysis
- **Image Classification**: ResNet50 on ImageNet (1000 classes)
- **OCR**: Tesseract-based text extraction
- **Interactive UI**: Streamlit web interface
- **Batch Processing**: Process multiple images
- **Export Results**: JSON and annotated images

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Tesseract OCR installed
- Webcam (optional, for live detection)

### Installation

```bash
# Clone repository
git clone <your-repo-url>
cd cv-ai-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Tesseract
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr

# macOS:
brew install tesseract

# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
```

### Run Application

```bash
streamlit run app.py
```

## 💻 Usage

### Web Interface

1. Launch the app with `streamlit run app.py`
2. Select desired tasks from sidebar
3. Upload an image
4. View results in real-time

### Programmatic Usage

```python
from app import CVAISystem
import cv2

# Initialize system
cv_system = CVAISystem()

# Load image
image = cv2.imread('image.jpg')

# Process with multiple tasks
results = cv_system.process_image(
    image,
    tasks=['object_detection', 'face_detection', 'classification', 'ocr']
)

# Access results
print(results['object_detection']['detections'])
print(results['classification']['predictions'])
print(results['ocr']['text'])
```

## 🛠️ Individual Components

### Object Detection

```python
from app import ObjectDetector

detector = ObjectDetector()
annotated_image, detections = detector.detect(image, confidence=0.5)

for det in detections:
    print(f"Found {det['class']} with {det['confidence']:.2%} confidence")
```

### Face Recognition

```python
from app import FaceRecognizer

recognizer = FaceRecognizer()
annotated_image, faces = recognizer.detect_faces(image)

print(f"Detected {len(faces)} faces")
```

### Image Classification

```python
from app import ImageClassifier

classifier = ImageClassifier()
predictions = classifier.classify(image, top_k=5)

for pred in predictions:
    print(f"{pred['class']}: {pred['confidence']:.2%}")
```

### OCR

```python
from app import OCREngine

ocr = OCREngine()
result = ocr.extract_text(image, lang='eng')

print(f"Extracted text: {result['text']}")
print(f"Confidence: {result['confidence']:.2f}%")
```

## 📊 Supported Tasks

| Task | Model | Performance |
|------|-------|-------------|
| Object Detection | YOLOv5s | ~45 FPS (GPU) |
| Face Detection | MTCNN | ~30 FPS |
| Classification | ResNet50 | ~100 FPS (GPU) |
| OCR | Tesseract | Varies by text |

## 🎨 Examples

### Detect Objects

```python
results = cv_system.process_image(image, ['object_detection'])
# Returns: person, car, dog, etc. with bounding boxes
```

### Extract Text from Document

```python
results = cv_system.process_image(document_image, ['ocr'])
print(results['ocr']['text'])
```

### Classify Scene

```python
results = cv_system.process_image(photo, ['classification'])
# Returns: beach, sunset, mountain, etc. with confidence
```

## ⚙️ Configuration

Edit parameters in the code:

```python
# Object detection confidence
detector.detect(image, confidence=0.7)

# Top K classifications
classifier.classify(image, top_k=10)

# OCR language
ocr.extract_text(image, lang='spa')  # Spanish
```

## 📁 Project Structure

```
cv-ai-system/
├── app.py                 # Main application
├── requirements.txt       # Dependencies
├── README.md             # Documentation
├── models/               # Model weights (auto-downloaded)
├── samples/              # Sample images
├── outputs/              # Processed results
└── tests/                # Unit tests
```

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Test individual components
python -c "from app import ObjectDetector; print('OK')"
```

## 🚀 Advanced Usage

### Batch Processing

```python
import glob

cv_system = CVAISystem()

for img_path in glob.glob('images/*.jpg'):
    image = cv2.imread(img_path)
    results = cv_system.process_image(image, ['object_detection'])
    
    # Save annotated image
    cv2.imwrite(f'output/{Path(img_path).name}', 
                results['object_detection']['image'])
```

### Custom Object Detection Classes

```python
# Filter specific classes
detector.model.classes = [0, 2, 3]  # person, car, motorcycle
```

### Face Database

```python
# Add known faces
recognizer.face_database['person_1'] = embedding_1
recognizer.face_database['person_2'] = embedding_2

# Match faces
distance = np.linalg.norm(embedding_1 - embedding_2)
is_same = distance < 0.6
```

## 📈 Performance Optimization

### GPU Acceleration

```python
# Use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
```

### Batch Inference

```python
# Process multiple images at once
images = [img1, img2, img3]
batch_results = [cv_system.process_image(img, tasks) for img in images]
```

### Model Optimization

```python
# Use TorchScript for faster inference
model = torch.jit.script(model)
```

## 🔧 Troubleshooting

**Issue: Tesseract not found**
```bash
# Set tesseract path manually
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
```

**Issue: CUDA out of memory**
```python
# Reduce batch size or use CPU
device = 'cpu'
```

**Issue: Slow inference**
- Use GPU acceleration
- Reduce image resolution
- Use smaller models (yolov5n)

## 📱 Mobile Deployment

Convert models to mobile formats:

```python
# Export to ONNX
torch.onnx.export(model, dummy_input, "model.onnx")

# Export to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
```

## 🤝 Contributing

Contributions welcome! Areas to improve:

- [ ] Add more object detection models
- [ ] Implement face recognition matching
- [ ] Support video processing
- [ ] Add pose estimation
- [ ] Implement segmentation
- [ ] Create REST API
- [ ] Add model benchmarking

## 📝 License

MIT License

## 🙏 Acknowledgments

- Ultralytics for YOLOv5
- timesler for facenet-pytorch
- PyTorch team
- Tesseract OCR team

## 📧 Contact

For questions, open an issue on GitHub.
