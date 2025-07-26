from flask import Flask, request, render_template

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2, ResNet50
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import requests
from PIL import Image
import io

print("App Declared...")
app = Flask(__name__)
model_path = "./best_performed_model.h5"
print("Loading model...")
model = load_model(model_path)

print("Model loaded successfully!")
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
# Define full names corresponding to the class names
full_names = [
    'Actinic keratosis',
    'Basal cell carcinoma',
    'Benign keratosis-like lesions',
    'Dermatofibroma',
    'Melanoma',
    'Melanocytic nevi',
    'Vascular lesions'
]
MobileNetV2_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
ResNet50_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
class MultiStageSkinLesionDetector:
    def __init__(self, class_names):

        self.lesion_model = model
        self.class_names = class_names
        self.skin_detector = MobileNetV2_model
        self.anomaly_detector = ResNet50_model
        
    def detect_skin_region(self, file, confidence_threshold=0.6):
        """
        Stage 1: Detect if image contains skin
        
        Returns:
            is_skin: Boolean indicating if skin is detected
            confidence: Confidence score
            details: Additional information
        """
        try:
            # Load and preprocess image
            img_bytes = file.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return False, 0.0, {"error": "Cannot read image"}
            
            # Method 1: Color-based skin detection
            skin_confidence, skin_details = self._color_based_skin_detection(img)
            
            # Method 2: Deep learning features (backup method)
            feature_confidence = self._feature_based_skin_detection(file)
            
            # Method 3: Texture analysis
            texture_confidence = self._texture_based_skin_detection(img)
            
            # Combine confidences
            combined_confidence = (
                skin_confidence * 0.5 + 
                feature_confidence * 0.3 + 
                texture_confidence * 0.2
            )
            
            is_skin = combined_confidence >= confidence_threshold
            
            details = {
                "color_confidence": skin_confidence,
                "feature_confidence": feature_confidence,
                "texture_confidence": texture_confidence,
                "combined_confidence": combined_confidence,
                **skin_details
            }
            
            return is_skin, combined_confidence, details
            
        except Exception as e:
            return False, 0.0, {"error": str(e)}
    
    def _color_based_skin_detection(self, img):
        """Detect skin using color analysis"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        
        # Define skin color ranges in HSV
        lower_hsv = np.array([0, 20, 70])
        upper_hsv = np.array([20, 255, 255])
        mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
        
        # Define skin color ranges in YCrCb
        lower_ycrcb = np.array([0, 135, 85])
        upper_ycrcb = np.array([255, 180, 135])
        mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
        
        # Combine masks
        skin_mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
        
        # Calculate skin pixel ratio
        skin_pixels = np.sum(skin_mask > 0)
        total_pixels = img.shape[0] * img.shape[1]
        skin_ratio = skin_pixels / total_pixels
        
        # Additional checks for skin-like properties
        mean_hue = np.mean(hsv[:, :, 0][skin_mask > 0]) if skin_pixels > 0 else 0
        mean_saturation = np.mean(hsv[:, :, 1][skin_mask > 0]) if skin_pixels > 0 else 0
        
        # Calculate confidence based on skin ratio and color properties
        confidence = min(skin_ratio * 2, 1.0)  # Cap at 1.0
        
        details = {
            "skin_pixel_ratio": skin_ratio,
            "mean_hue": mean_hue,
            "mean_saturation": mean_saturation
        }
        
        return confidence, details
    
    def _feature_based_skin_detection(self, file):
        """Use pretrained model features for skin detection"""
        try:
            # Load and preprocess for MobileNetV2
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes))
            img = img.convert('RGB').resize((224, 224))
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
            
            # Extract features
            features = self.skin_detector.predict(img_array, verbose=0)
            
            # Simple heuristic: skin images tend to have certain feature patterns
            # This is a simplified approach - in practice, you'd train a proper classifier
            feature_variance = np.var(features)
            
            # Normalize variance to confidence (0-1)
            confidence = min(feature_variance / 1000, 1.0)
            
            return confidence
            
        except Exception:
            return 0.0
    
    def _texture_based_skin_detection(self, img):
        """Analyze texture to detect skin"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate texture features
        # 1. Local Binary Pattern (simplified)
        lbp_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 2. Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (img.shape[0] * img.shape[1])
        
        # Skin typically has moderate texture variation
        # Too smooth = artificial, too rough = non-skin
        if 100 < lbp_var < 2000 and 0.05 < edge_density < 0.25:
            confidence = 0.8
        elif 50 < lbp_var < 3000 and 0.02 < edge_density < 0.35:
            confidence = 0.6
        else:
            confidence = 0.2
            
        return confidence
    
    def detect_skin_lesions(self, file, lesion_threshold=0.7):
        """
        Stage 2: Detect if skin has lesions/scars/abnormalities
        
        Returns:
            has_lesions: Boolean indicating if lesions are detected
            confidence: Confidence score
            details: Additional information
        """
        try:
            img_bytes = file.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return False, 0.0, {"error": "Cannot read image"}
            
            # Method 1: Analyze color uniformity
            uniformity_score = self._analyze_color_uniformity(img)
            
            # Method 2: Detect irregular shapes/boundaries
            irregularity_score = self._detect_irregularities(img)
            
            # Method 3: Analyze texture variation
            texture_variation = self._analyze_texture_variation(img)
            
            # Method 4: Use deep features for anomaly detection
            anomaly_score = self._deep_anomaly_detection(file)
            
            # Combine scores (higher = more likely to have lesions)
            combined_score = (
                (1 - uniformity_score) * 0.25 +  # Less uniform = more lesions
                irregularity_score * 0.3 +
                texture_variation * 0.25 +
                anomaly_score * 0.2
            )
            
            has_lesions = combined_score >= lesion_threshold
            
            details = {
                "uniformity_score": uniformity_score,
                "irregularity_score": irregularity_score,
                "texture_variation": texture_variation,
                "anomaly_score": anomaly_score,
                "combined_score": combined_score
            }
            
            return has_lesions, combined_score, details
            
        except Exception as e:
            return False, 0.0, {"error": str(e)}
    
    def _analyze_color_uniformity(self, img):
        """Analyze how uniform the skin color is"""
        # Convert to LAB color space for better color analysis
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        # Calculate standard deviation of each channel
        l_std = np.std(lab[:, :, 0])
        a_std = np.std(lab[:, :, 1])
        b_std = np.std(lab[:, :, 2])
        
        # Combine standard deviations
        total_std = l_std + a_std + b_std
        
        # Normalize to 0-1 (lower = more uniform)
        uniformity = max(0, 1 - (total_std / 100))
        
        return uniformity
    
    def _detect_irregularities(self, img):
        """Detect irregular shapes and boundaries"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur and find contours
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return 0.0
        
        irregularity_scores = []
        
        for contour in contours:
            if cv2.contourArea(contour) < 100:  # Skip small contours
                continue
                
            # Calculate perimeter and area
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            
            if area == 0:
                continue
                
            # Circularity: 4π*area/perimeter² (1 = perfect circle, <1 = irregular)
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            irregularity = 1 - circularity
            
            irregularity_scores.append(irregularity)
        
        return np.mean(irregularity_scores) if irregularity_scores else 0.0
    
    def _analyze_texture_variation(self, img):
        """Analyze texture variation in the image"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate local standard deviation
        kernel = np.ones((9, 9)) / 81
        mean_img = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        sqr_img = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
        std_img = np.sqrt(sqr_img - mean_img**2)
        
        # Calculate variation in texture
        texture_variation = np.std(std_img) / 50  # Normalize
        
        return min(texture_variation, 1.0)
    
    def _deep_anomaly_detection(self, file):
        """Use deep features for anomaly detection"""
        try:
            # Load and preprocess for ResNet50
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes))
            img = img.convert('RGB').resize((224, 224))
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
            
            # Extract features
            features = self.anomaly_detector.predict(img_array, verbose=0)
            
            # Simple anomaly scoring based on feature activation patterns
            # Higher activation variance might indicate anomalies
            feature_variance = np.var(features)
            anomaly_score = min(feature_variance / 5000, 1.0)
            
            return anomaly_score
            
        except Exception:
            return 0.0
    
    def classify_lesion(self, file):
        try:
            # Load and preprocess for your model
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes))
            img = img.convert('RGB').resize((224, 224))
            img_array = np.array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make prediction
            predictions = self.lesion_model.predict(img_array, verbose=0)
            probabilities = predictions[0]
            
            predicted_class_idx = np.argmax(probabilities)
            confidence = np.max(probabilities)
            predicted_class = self.class_names[predicted_class_idx]
            
            return predicted_class, confidence, probabilities
            
        except Exception as e:
            return None, 0.0, None
    
    def simplified_pipeline_prediction(self, file, skin_threshold=0.6, classification_threshold=0.7):
        
        result = {
            "stage": "none",
            "is_skin": False,
            "lesion_type": None,
            "confidence": 0.0,
            "probabilities": None,
            "details": {},
            "message": "",
            "safe_to_classify": False
        }

        is_skin, skin_conf, skin_details = self.detect_skin_region(file, skin_threshold)
        
        result["stage"] = "skin_detection"
        result["is_skin"] = is_skin
        result["details"]["skin"] = skin_details
        
        if not is_skin:
            result["message"] = f"🚫 NOT SKIN: Confidence {skin_conf:.3f} < {skin_threshold}"
            return result
 

        lesion_type, class_conf, probabilities = self.classify_lesion(file)
        result["stage"] = "classification"
        result["lesion_type"] = lesion_type
        result["confidence"] = class_conf
        result["probabilities"] = probabilities.tolist() if probabilities is not None else None
        
        if class_conf < classification_threshold:
            result["message"] = f"⚠️  UNCERTAIN CLASSIFICATION: {lesion_type} (confidence: {class_conf:.3f})"
        else:
            result["message"] = f"🎯 CONFIDENT CLASSIFICATION: {lesion_type} (confidence: {class_conf:.3f})"
            result["safe_to_classify"] = True
        
        return result
    

# Standalone function for easy integration
def predict_skin_lesion(file, class_names, skin_threshold=0.6):
    detector = MultiStageSkinLesionDetector(class_names)
    return detector.simplified_pipeline_prediction(file, skin_threshold)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']
    
        if file:
            result = predict_skin_lesion(file, class_names)

            if not result['is_skin']:
                predicted_class = 'No Skin Detected'
                confidence = 1.0
            else:
                file.seek(0)
                img_bytes = file.read()
                img = Image.open(io.BytesIO(img_bytes))
                img = img.convert('RGB').resize((128, 128))
                img_array = np.array(img)
                img_array = img_array / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Make prediction
                predictions = model.predict(img_array, verbose=0)
                probabilities = predictions[0]
                
                predicted_class_idx = np.argmax(probabilities)
                confidence = np.max(probabilities)
                predicted_class = full_names[predicted_class_idx]

            return render_template('index.html', predicted_full_name=predicted_class, confidence=confidence)
    # Render the template without prediction for GET requests
    return render_template('index.html', predicted_full_name=None, confidence=None)

if __name__ == '__main__':
    app.run(debug=True)