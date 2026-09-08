# 🔥 Wildfire Detection System

An AI-based wildfire detection system that uses deep learning to classify images as **smoke** or **no-smoke**. It is a part of a larger wildfire detection system that combines computer vision, environmental sensors, embedded systems, and automated alerts. 

## 🚀 Features

- Image classification using a fine-tuned ResNet-18 model
- Dataset preparation and train/validation/test splitting
- Fine tuning of ResNet's final convolutional layer and classification layer
- Model training and evaluation
- Confusion Matrix
- False positive analysis
- Image prediction with confidence scores
- SMS alert functionality using Twilio

## 💻 Technologies Used

- Python
- PyTorch
- Torchvision
- ResNet-18
- Scikit-learn
- Twilio
- OpenCV
- Raspberry Pi
- Arduino UNO

## 📊 Training Pipeline 

```text
Image Dataset
      ↓
Dataset Splitting
      ↓
Train / Validation / Test Sets
      ↓
Image Preprocessing
(Resize → 224×224 → Tensor)
      ↓
Fine-Tuned ResNet-18
      ↓
Model Training
      ↓
Validation
      ↓
Independent Testing
      ↓
Accuracy + Confusion Matrix
      ↓
False Positive Analysis
```


## 📁 Project Structure

```text
wildfire-detection/
├── src/
│   ├── wildfire_train.py         # Trains and evaluates the ResNet-18 model
│   ├── predict.py                # Runs image predictions
│   ├── split_dataset.py          # Splits the dataset into 3 subsets: Training, Validation, Testing
│   └── SMS.py                    # Relays or communicates with the local authorities based on the results of the model
├── models/
├── requirements.txt
└── .gitignore
```

## 🔭 Larger System

```text
Environmental Sensors
        ↓
      Arduino
        ↓
   Raspberry Pi
      ↙       ↘
Sensor Data   Camera
                 ↓
          AI Smoke Detection
                 ↓
           Alert System
                 ↓
             SMS Alert

```

## Machine Learning Model

The project uses transfer learning with a pretrained ResNet-18 convolutional neural network. The model was adapted for binary image classification to distinguish between smoke and non-smoke images.

During training, most of the pretrained network was frozen while later layers and the final classification layer were fine-tuned for the wildfire detection task.

## 🔮 Future Work

- Integrate real-time camera input 
- Deploy the model on raspberry pi hardware
- Integrate environmental sensors
- Trigger automated emergency alerts based on detection results  


