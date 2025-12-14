# Real-Time Hand Gesture Recognition using MediaPipe and Neural Networks

This project implements a real-time hand gesture recognition system based on hand landmark features extracted using MediaPipe Hands and classified using a TensorFlow neural network (MLP).

The system follows a complete pipeline:
data collection → feature extraction → neural network training → real-time inference.

---

## Project Description

Instead of using raw images, this system relies on geometric hand landmarks to achieve fast and stable gesture recognition.
A fully connected neural network is trained to classify gestures from a 69-dimensional feature vector derived from hand joint positions.

This approach is lightweight, interpretable, and suitable for real-time execution on CPU or GPU.

---

## System Architecture

Webcam  
↓  
MediaPipe Hands (21 landmarks)  
↓  
69-D Feature Extraction (geometric + distances)  
↓  
Neural Network Classifier (MLP – TensorFlow)  
↓  
Gesture Prediction  

---

## Repository Structure

.
├── collect_data_hybrid.py  
├── extract_features.py  
├── train_tf.py  
├── run_realtime_tf.py  
│  
├── dataset/  
│   ├── gesture_name.csv  
│   └── images/  
│  
├── features/  
│   └── features.csv  
│  
├── models/  
│   ├── tf_gesture_model.keras  
│   └── labels.txt  
│  
├── requirements.txt  
└── README.md  

---

## Neural Network Usage

The neural network is used only in the following files:

train_tf.py  
Defines and trains a fully connected neural network (MLP).

run_realtime_tf.py  
Loads the trained model and performs real-time inference.

Model architecture:
Input (69)
→ Dense(128) + ReLU
→ Dense(64) + ReLU
→ Dense(N) + Softmax

---

## Feature Representation

Each hand sample is represented by:
- 63 normalized landmark coordinates (21 points × x, y, z)
- 6 geometric distances between key joints

---

## Installation

Clone the repository:
git clone https://github.com/your-username/gesture-recognition.git
cd gesture-recognition

Create a virtual environment:
python -m venv venv
source venv/bin/activate
venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt

---

## requirements.txt

opencv-python
mediapipe
numpy
pandas
scikit-learn
tensorflow
keras

---

## Data Collection

python collect_data_hybrid.py

---

## Dataset Preparation

python extract_features.py

Output:
features/features.csv

---

## Model Training

python train_tf.py

Outputs:
models/tf_gesture_model.keras  
models/labels.txt  

---

## Real-Time Inference

python run_realtime_tf.py

Press Q to exit.

---

## Performance Characteristics

Input Features: 69  
Model Type: Fully Connected Neural Network (MLP)  
Latency: Real-time  
Hand Detector: MediaPipe Hands  
Hardware: CPU / GPU  

---

## Applications

Human–Computer Interaction  
Touchless interfaces  
Assistive technologies  
Smart systems  
Academic research  

---

## Future Improvements

CNN and landmark hybrid architecture  
Temporal modeling with LSTM or Transformers  
Gesture sequence recognition  
Model optimization using ONNX or TensorRT  
Mobile deployment  

---

## License

This project is intended for research and educational use.
