# Smart ANPR and Vehicle Classification System

## Project Title
Smart ANPR and Video Incident Detection System for Vehicle Classification and Traffic Analytics

## Project Objectives
- Develop a working prototype for vehicle detection and classification using deep learning
- Implement OpenCV preprocessing to improve detection accuracy under varying lighting conditions
- Build a CNN-based classifier (MobileNetV2) for vehicle type classification
- Enable license plate detection and character extraction using OCR
- Provide traffic analytics and incident detection capabilities
- Deliver a user-friendly web interface using Streamlit for real-time monitoring

## Features and Modules

### 1. Vehicle Detection
- YOLOv8-based vehicle detection for Cars, Buses, and Trucks
- Real-time detection from images and videos
- Clear labeling of YOLO predictions vs. MobileNetV2 classifications

### 2. Vehicle Classification
- MobileNetV2 transfer learning model for vehicle type classification
- Distinguishes between Cars, Buses, and Trucks
- Can be trained on custom dataset

### 3. Image Preprocessing
- CLAHE (Contrast Limited Adaptive Histogram Equalization) for brightness/contrast enhancement
- Resizing and normalization
- Grayscale conversion

### 4. License Plate Detection & OCR
- EasyOCR integration for license plate character extraction
- Basic color-based plate detection prototype
- Clearly marked as feature under development

### 5. Traffic Analytics
- Vehicle counting and classification summaries
- Traffic density estimation
- Incident detection algorithms
- Historical data tracking

### 6. Export functionality
- Export detection results as CSV
- Export analytics data for further processing

### 7. Streamlit Dashboard
- Image and video upload interface
- Real-time annotated frame display
- Vehicle count metrics and breakdown
## Technology Stack

- **Python 3.11**: Core programming language
- **OpenCV**: Image preprocessing and computer vision operations
- **Ultralytics YOLO**: Vehicle detection framework
- **TensorFlow/Keras**: MobileNetV2-based vehicle classification
- **EasyOCR**: License plate character recognition
- **Streamlit**: Web application interface
- **Pandas, NumPy, Scikit-learn**: Data processing and analytics

## Project Architecture

```
smart-anpr-vehicle-classification/
├── app.py                                              # Main Streamlit application
├── requirements.txt                                    # Python dependencies
├── .gitignore                                         # Git ignore file
├── README.md                                          # Project documentation
├── venv/                                              # Virtual environment
├── detectors/
│   ├── __init__.py                                    # Package init
│   └── yolo_detector.py                               # YOLO vehicle detection
├── classifiers/
│   ├── __init__.py                                    # Package init
│   └── mobilnet_classifier.py                         # MobileNetV2 classification
├── preprocessor/
│   ├── __init__.py                                    # Package init
│   └── openpreprocessor.py                            # OpenCV preprocessing
├── utils/
│   ├── __init__.py                                    # Package init
│   └── analytics.py                                   # Vehicle counting and traffic analytics
├── exports/
│   ├── __init__.py                                    # Package init
│   └── csv_export.py                                  # CSV export functionality
## Installation Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/Diyapratap22/smart-anpr-vehicle-classification.git
   cd smart-anpr-vehicle-classification
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Access the application**
   Open your web browser and navigate to `http://localhost:8501`

## How to Run the Application

After installation, use the following commands:

- **Phase 1 (Vehicle Detection)**: `streamlit run app.py`
  - Upload images or videos for vehicle detection
  - YOLO-based detection with vehicle counts displayed
  - Results can be exported as CSV

- **Phase 2 (Classification)**: After training MobileNetV2
  - Replace `model_path=None` with trained model path
  - Vehicle classifications will be updated from YOLO to MobileNetV2

- **Phase 3 (ANPR)**: After OCR model setup
  - License plate detection and extraction will be available
  - Registration numbers will be extracted and displayed

- **Phase 4 (Incidents)**: Traffic incident detection enabled
  - Real-time incident alerts will be displayed
  - Analytics dashboard will show traffic density and incidents

## Screenshots

![Dashboard Screenshot](docs/screenshots/dashboard.png)
*(Screenshots to be added after full implementation)*

## Future Enhancements

- Train MobileNetV2 on custom vehicle dataset for improved classification accuracy
- Implement dedicated license plate detection model (e.g., using plate-specific architecture)
- Add real-time video streaming support
- Implement speed estimation and vehicle tracking
- Add integration with traffic management systems
- Develop mobile application version
- Add JSON export in addition to CSV
- Implement video summarization and highlight generation

## Author and GitHub Profile

**Diyapratap22**
- GitHub: https://github.com/Diyapratap22
- Project: Smart ANPR and Vehicle Classification System
- Institution: Kavyan ITS Solutions Pvt. Ltd. - AI & ML Internship