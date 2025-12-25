"""
Computer Vision AI System
Multi-task CV system: Object Detection, Face Recognition, Image Classification, OCR
"""

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import pytesseract
from facenet_pytorch import MTCNN, InceptionResnetV1
import streamlit as st
from typing import List, Tuple, Dict, Any
import io
import json
from pathlib import Path


class ObjectDetector:
    """YOLO-based object detection"""
    
    def __init__(self):
        """Initialize object detector"""
        # Load YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model.eval()
        
    def detect(self, image: np.ndarray, confidence: float = 0.5) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect objects in image
        
        Args:
            image: Input image (BGR format)
            confidence: Detection confidence threshold
            
        Returns:
            Annotated image and detection results
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run detection
        results = self.model(image_rgb)
        
        # Filter by confidence
        detections = results.pandas().xyxy[0]
        detections = detections[detections['confidence'] >= confidence]
        
        # Draw boxes on image
        annotated_image = image.copy()
        
        detection_list = []
        for _, det in detections.iterrows():
            x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
            label = f"{det['name']} {det['confidence']:.2f}"
            
            # Draw rectangle
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            cv2.putText(annotated_image, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            detection_list.append({
                'class': det['name'],
                'confidence': float(det['confidence']),
                'bbox': [x1, y1, x2, y2]
            })
        
        return annotated_image, detection_list


class FaceRecognizer:
    """Face detection and recognition"""
    
    def __init__(self):
        """Initialize face recognizer"""
        self.mtcnn = MTCNN(keep_all=True, device='cpu')
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        self.face_database = {}
        
    def detect_faces(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect faces in image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Annotated image and face information
        """
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Detect faces
        boxes, probs = self.mtcnn.detect(pil_image)
        
        annotated_image = image.copy()
        face_list = []
        
        if boxes is not None:
            for i, (box, prob) in enumerate(zip(boxes, probs)):
                if prob > 0.9:
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                    
                    # Draw rectangle
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    # Draw label
                    label = f"Face {i+1} ({prob:.2f})"
                    cv2.putText(annotated_image, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
                    face_list.append({
                        'face_id': i + 1,
                        'confidence': float(prob),
                        'bbox': [x1, y1, x2, y2]
                    })
        
        return annotated_image, face_list
    
    def extract_embedding(self, image: np.ndarray, box: List[int]) -> np.ndarray:
        """Extract face embedding"""
        x1, y1, x2, y2 = box
        face = image[y1:y2, x1:x2]
        
        # Convert to PIL
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        pil_face = Image.fromarray(face_rgb)
        
        # Get aligned face
        face_tensor = self.mtcnn(pil_face)
        
        if face_tensor is not None:
            # Get embedding
            with torch.no_grad():
                embedding = self.resnet(face_tensor.unsqueeze(0))
            return embedding.cpu().numpy()
        
        return None


class ImageClassifier:
    """Image classification using ResNet"""
    
    def __init__(self):
        """Initialize classifier"""
        self.model = models.resnet50(pretrained=True)
        self.model.eval()
        
        # Load ImageNet labels
        self.labels = self._load_labels()
        
        # Image preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def _load_labels(self) -> List[str]:
        """Load ImageNet class labels"""
        # Simplified label list (in production, load from file)
        return [f"class_{i}" for i in range(1000)]
    
    def classify(self, image: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Classify image
        
        Args:
            image: Input image (BGR format)
            top_k: Number of top predictions to return
            
        Returns:
            List of predictions with class and confidence
        """
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Preprocess
        input_tensor = self.preprocess(pil_image)
        input_batch = input_tensor.unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_batch)
        
        # Get probabilities
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # Get top k predictions
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            predictions.append({
                'class': self.labels[idx.item()] if idx.item() < len(self.labels) else f"class_{idx.item()}",
                'confidence': float(prob.item())
            })
        
        return predictions


class OCREngine:
    """Optical Character Recognition"""
    
    def __init__(self):
        """Initialize OCR engine"""
        # Configure tesseract path if needed
        # pytesseract.pytesseract.tesseract_cmd = r'path/to/tesseract'
        pass
    
    def extract_text(self, image: np.ndarray, lang: str = 'eng') -> Dict[str, Any]:
        """
        Extract text from image
        
        Args:
            image: Input image (BGR format)
            lang: Language code for OCR
            
        Returns:
            Extracted text and metadata
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Extract text
        text = pytesseract.image_to_string(binary, lang=lang)
        
        # Get detailed data
        data = pytesseract.image_to_data(binary, output_type=pytesseract.Output.DICT)
        
        # Draw boxes on image
        annotated_image = image.copy()
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            if int(data['conf'][i]) > 60:  # Confidence threshold
                (x, y, w, h) = (data['left'][i], data['top'][i], 
                               data['width'][i], data['height'][i])
                cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        return {
            'text': text.strip(),
            'annotated_image': annotated_image,
            'word_count': len([w for w in data['text'] if w.strip()]),
            'confidence': np.mean([int(c) for c in data['conf'] if int(c) > 0])
        }


class CVAISystem:
    """Complete Computer Vision AI System"""
    
    def __init__(self):
        """Initialize CV system"""
        self.object_detector = ObjectDetector()
        self.face_recognizer = FaceRecognizer()
        self.image_classifier = ImageClassifier()
        self.ocr_engine = OCREngine()
    
    def process_image(self, image: np.ndarray, tasks: List[str]) -> Dict[str, Any]:
        """
        Process image with selected tasks
        
        Args:
            image: Input image
            tasks: List of tasks to perform
            
        Returns:
            Results from all tasks
        """
        results = {}
        
        if 'object_detection' in tasks:
            annotated, detections = self.object_detector.detect(image)
            results['object_detection'] = {
                'image': annotated,
                'detections': detections
            }
        
        if 'face_detection' in tasks:
            annotated, faces = self.face_recognizer.detect_faces(image)
            results['face_detection'] = {
                'image': annotated,
                'faces': faces
            }
        
        if 'classification' in tasks:
            predictions = self.image_classifier.classify(image)
            results['classification'] = {
                'predictions': predictions
            }
        
        if 'ocr' in tasks:
            ocr_result = self.ocr_engine.extract_text(image)
            results['ocr'] = ocr_result
        
        return results


def main():
    """Streamlit UI"""
    st.set_page_config(page_title="CV AI System", page_icon="👁️", layout="wide")
    
    st.title("👁️ Computer Vision AI System")
    st.markdown("Multi-task computer vision: Object Detection, Face Recognition, Classification, OCR")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        tasks = st.multiselect(
            "Select Tasks",
            ['object_detection', 'face_detection', 'classification', 'ocr'],
            default=['object_detection']
        )
        
        confidence = st.slider("Detection Confidence", 0.0, 1.0, 0.5, 0.05)
        
        st.markdown("---")
        st.markdown("### 📤 Upload Image")
        uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
    
    # Initialize system
    if 'cv_system' not in st.session_state:
        with st.spinner("Loading models..."):
            st.session_state.cv_system = CVAISystem()
        st.success("Models loaded!")
    
    # Process image
    if uploaded_file is not None:
        # Load image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Display original
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Original Image")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Process
        with st.spinner("Processing..."):
            results = st.session_state.cv_system.process_image(image, tasks)
        
        with col2:
            st.subheader("🎯 Results")
            
            # Display results
            for task, result in results.items():
                st.markdown(f"**{task.replace('_', ' ').title()}**")
                
                if 'image' in result:
                    st.image(cv2.cvtColor(result['image'], cv2.COLOR_BGR2RGB))
                
                if 'detections' in result:
                    st.json(result['detections'])
                
                if 'faces' in result:
                    st.json(result['faces'])
                
                if 'predictions' in result:
                    st.json(result['predictions'])
                
                if 'text' in result:
                    st.text_area("Extracted Text", result['text'], height=200)
                    st.write(f"Word Count: {result['word_count']}")
                    st.write(f"Average Confidence: {result['confidence']:.2f}%")


if __name__ == "__main__":
    main()
