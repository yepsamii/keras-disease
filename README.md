# 🔬 AI Skin Disease Classification System

A modern, web-based application that uses deep learning to classify skin lesions and diseases from uploaded images. Built with Flask, Keras, and featuring a sleek, animated user interface.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Flask](https://img.shields.io/badge/flask-v2.0+-green.svg)
![Keras](https://img.shields.io/badge/keras-v2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Disease Classifications](#disease-classifications)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Medical Disclaimer](#medical-disclaimer)

## 🎯 Overview

This application leverages state-of-the-art deep learning techniques to analyze skin lesion images and provide classification results with confidence scores. The system is designed for educational and research purposes, featuring a modern web interface with smooth animations and professional medical aesthetics.

### Key Capabilities

- **Real-time Image Analysis**: Upload and get instant classification results
- **7 Disease Categories**: Comprehensive coverage of common skin conditions
- **Confidence Scoring**: Transparent probability assessment for each prediction
- **Modern UI/UX**: Professional interface with smooth animations
- **Mobile Responsive**: Works seamlessly across all devices

## 🏥 Disease Classifications

The system can identify and classify the following skin conditions:

| Code      | Full Name                     | Description                                     |
| --------- | ----------------------------- | ----------------------------------------------- |
| **akiec** | Actinic Keratosis             | Pre-cancerous skin lesions caused by sun damage |
| **bcc**   | Basal Cell Carcinoma          | Most common type of skin cancer                 |
| **bkl**   | Benign Keratosis-like Lesions | Non-cancerous skin growths                      |
| **df**    | Dermatofibroma                | Benign fibrous skin nodules                     |
| **mel**   | Melanoma                      | Most dangerous form of skin cancer              |
| **nv**    | Melanocytic Nevi              | Common moles or beauty marks                    |
| **vasc**  | Vascular Lesions              | Blood vessel-related skin conditions            |

## 🛠 Technology Stack

### **Backend**

- **Flask 2.0+**: Lightweight Python web framework
- **Keras/TensorFlow**: Deep learning model framework
- **OpenCV**: Image processing and computer vision
- **NumPy**: Numerical computing and array operations

### **Frontend**

- **HTML5**: Semantic markup structure
- **CSS3**: Modern styling with animations and gradients
- **JavaScript**: Interactive functionality and form handling
- **Responsive Design**: Mobile-first approach

### **Development Tools**

- **Python 3.8+**: Programming language
- **Virtual Environment**: Dependency isolation
- **Git**: Version control

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yepsamii/keras-disease
cd keras-disease
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Pre-trained Model

```bash
# Place your trained model file as 'model.keras' in the root directory
# Or train your own model using the provided training scripts
```

### Step 5: Run the Application

```bash
python app.py
```

The application will be available at `http://localhost:5000`

## 🧮 Model Details

### Architecture

- **Base Model**: Convolutional Neural Network (CNN)
- **Input Size**: 224x224x3 RGB images
- **Output**: 7-class classification with softmax activation
- **Training Data**: HAM10000 dermatology dataset

### Preprocessing Pipeline

1. **Image Resizing**: Standardized to 224x224 pixels
2. **Normalization**: Pixel values scaled to [0, 1] range
3. **Batch Processing**: Single image prediction with batch dimension
4. **Format Handling**: Automatic BGR to RGB conversion

### Accessibility Features

- **High Contrast**: WCAG compliant color schemes
- **Focus States**: Keyboard navigation support
- **Screen Reader**: Semantic HTML structure
- **Mobile Optimization**: Touch-friendly interface

## 📚 Usage

### Basic Workflow

1. **Access Application**: Navigate to `http://localhost:5000`
2. **Upload Image**: Click "Choose skin image to analyze" button
3. **Select File**: Choose a clear image of the skin lesion
4. **Analyze**: Click "Analyze Image" to process
5. **View Results**: See classification and confidence score

### Image Requirements

- **Format**: JPG, PNG, or GIF
- **Quality**: High resolution for better accuracy
- **Content**: Clear view of the skin lesion
- **Size**: No strict limit (automatically resized)

### Best Practices

- Use well-lit, focused images
- Ensure the lesion is clearly visible
- Avoid blurry or distorted images
- Consider multiple angles for complex lesions

## 🔧 API Reference

### Endpoints

#### `GET /`

Returns the main application interface.

**Response**: HTML page with upload form

#### `POST /`

Processes uploaded image and returns classification.

**Parameters**:

- `file`: Image file (multipart/form-data)

**Response**: HTML page with prediction results

### Example cURL Request

```bash
curl -X POST \
  http://localhost:5000/ \
  -F "file=@skin_lesion.jpg"
```

### How to Contribute

1. **Fork the Repository**
2. **Create Feature Branch**: `git checkout -b feature/amazing-feature`
3. **Commit Changes**: `git commit -m 'Add amazing feature'`
4. **Push to Branch**: `git push origin feature/amazing-feature`
5. **Open Pull Request**

### Development Guidelines

- Follow PEP 8 Python style guide
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation as needed
- Ensure mobile responsiveness

### Areas for Improvement

- **Model Accuracy**: Enhanced training with larger datasets
- **Additional Features**: Batch processing, comparison tools
- **Performance**: Optimization for faster inference
- **Accessibility**: Enhanced screen reader support
- **Internationalization**: Multi-language support

## ⚠️ Medical Disclaimer

**IMPORTANT: This application is for educational and research purposes only.**

- **Not a Medical Device**: This tool is not intended for medical diagnosis
- **Consult Professionals**: Always seek advice from qualified dermatologists
- **No Treatment Recommendations**: Results should not guide medical decisions
- **Accuracy Limitations**: AI predictions may contain errors or biases
- **Emergency Situations**: Seek immediate medical attention for urgent concerns

_Last updated: July 2025_
